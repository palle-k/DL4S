//
//  GPUImplicitConvolution.swift
//  DL4S
//
//  Created by Palle Klewitz on 29.09.26.
//  Copyright (c) 2026 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation
import Metal
import Synchronization

/// Convolutions as implicit matrix products on the GPU, see `convolution.metal`.
///
/// The kernel gathers the windows of the images when it loads its tiles, so no window matrix is written to memory.
/// The forward pass is one product. The data gradient is one product for every phase of the stride.
struct GPUImplicitConvolution {
    let batchSize: Int
    let inputChannels: Int
    let height: Int
    let width: Int
    let outputChannels: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let outputHeight: Int
    let outputWidth: Int
    let padding: Int
    let stride: Int

    init(inputShape: [Int], filterShape: [Int], outputShape: [Int], padding: Int, stride: Int) {
        (batchSize, inputChannels, height, width) = (inputShape[0], inputShape[1], inputShape[2], inputShape[3])
        (outputChannels, kernelHeight, kernelWidth) = (filterShape[0], filterShape[2], filterShape[3])
        (outputHeight, outputWidth) = (outputShape[2], outputShape[3])
        self.padding = padding
        self.stride = stride
    }

    /// Records the forward pass: `result = convolution(input, filters) + bias`.
    func forward(input: GPUBuffer, filters: GPUBuffer, bias: GPUBuffer?, result: GPUBuffer) {
        let windowSize = inputChannels * kernelHeight * kernelWidth
        let tables = GPUImplicitConvolutionTables.tables(for: TableKey(pass: .forward, convolution: self, phaseY: 0, phaseX: 0)) {
            let columns = (0 ..< windowSize).map { k -> SIMD4<Int32> in
                let (channel, tap) = k.quotientAndRemainder(dividingBy: kernelHeight * kernelWidth)
                let (i, j) = tap.quotientAndRemainder(dividingBy: kernelWidth)
                return SIMD4(Int32(channel * height * width + i * width + j), Int32(i), Int32(j), 0)
            }
            return GPUImplicitConvolutionTables.Tables(columns: columns, filterColumns: nil)
        }
        let parameters = ImplicitGemmParameters(
            m: outputChannels, n: batchSize * outputHeight * outputWidth, k: windowSize, aRowStride: windowSize,
            heightN: outputHeight, widthN: outputWidth,
            bBatch: inputChannels * height * width, bStrideY: stride, bOffsetY: -padding, bStrideX: stride, bOffsetX: -padding, bHeight: height, bWidth: width,
            cRowStride: outputHeight * outputWidth, cBatch: outputChannels * outputHeight * outputWidth, cStrideY: 1, cOffsetY: 0, cStrideX: 1, cOffsetX: 0, cWidth: outputWidth,
            accumulate: false, hasBias: bias != nil,
        )
        Self.encode(parameters, a: filters, b: input, c: result, columnTable: tables.columns, bias: bias)
    }

    /// Records the gradient of the input, which is added to the accumulated gradient when `accumulate` is true.
    func dataGradient(outputGradient: GPUBuffer, filters: GPUBuffer, inputGradient: GPUBuffer, accumulate: Bool) {
        for phaseY in 0 ..< stride {
            for phaseX in 0 ..< stride {
                // The rows y of the input with (y + padding) % stride == phaseY: y = stride * uy + phaseY - padding.
                let (firstRow, lastRow) = (Self.ceilingQuotient(padding - phaseY, stride), Self.floorQuotient(height - 1 + padding - phaseY, stride))
                let (firstColumn, lastColumn) = (Self.ceilingQuotient(padding - phaseX, stride), Self.floorQuotient(width - 1 + padding - phaseX, stride))
                guard lastRow >= firstRow, lastColumn >= firstColumn else {
                    continue
                }
                // The taps i with i % stride == phaseY reach these rows, from the output row uy - (i - phaseY) / stride.
                let (tapsY, tapsX) = ((kernelHeight - phaseY + stride - 1) / stride, (kernelWidth - phaseX + stride - 1) / stride)
                let taps = outputChannels * Swift.max(tapsY, 0) * Swift.max(tapsX, 0)
                let tables = GPUImplicitConvolutionTables.tables(for: TableKey(pass: .dataGradient, convolution: self, phaseY: phaseY, phaseX: phaseX)) {
                    var filterColumns: [Int32] = [], columns: [SIMD4<Int32>] = []
                    for channel in 0 ..< outputChannels {
                        for ty in 0 ..< tapsY {
                            for tx in 0 ..< tapsX {
                                let (i, j) = (phaseY + stride * ty, phaseX + stride * tx)
                                filterColumns.append(Int32(channel * inputChannels * kernelHeight * kernelWidth + i * kernelWidth + j))
                                columns.append(SIMD4(Int32(channel * outputHeight * outputWidth - ty * outputWidth - tx), Int32(-ty), Int32(-tx), 0))
                            }
                        }
                    }
                    return GPUImplicitConvolutionTables.Tables(columns: columns.isEmpty ? [.zero] : columns, filterColumns: filterColumns.isEmpty ? [0] : filterColumns)
                }
                let (rows, columns) = (lastRow - firstRow + 1, lastColumn - firstColumn + 1)
                // The filters of the phase are written as a contiguous [inputChannels, taps] matrix, which the product loads with
                // vector loads. It has at most as many elements as the filters.
                let phaseFilters = taps > 0 ? GPUKernels.temporary(count: inputChannels * taps) : filters
                if taps > 0 {
                    Self.gatherColumns(filters, rowStride: kernelHeight * kernelWidth, table: tables.filterColumns!, rows: inputChannels, columns: taps, into: phaseFilters)
                }
                let parameters = ImplicitGemmParameters(
                    m: inputChannels, n: batchSize * rows * columns, k: taps, aRowStride: taps,
                    heightN: rows, widthN: columns,
                    bBatch: outputChannels * outputHeight * outputWidth, bStrideY: 1, bOffsetY: firstRow, bStrideX: 1, bOffsetX: firstColumn, bHeight: outputHeight, bWidth: outputWidth,
                    cRowStride: height * width, cBatch: inputChannels * height * width,
                    cStrideY: stride, cOffsetY: stride * firstRow + phaseY - padding, cStrideX: stride, cOffsetX: stride * firstColumn + phaseX - padding, cWidth: width,
                    accumulate: accumulate, hasBias: false,
                )
                Self.encode(parameters, a: phaseFilters, b: outputGradient, c: inputGradient, columnTable: tables.columns, bias: nil)
            }
        }
    }

    private static func encode(_ parameters: ImplicitGemmParameters, a: GPUBuffer, b: GPUBuffer, c: GPUBuffer, columnTable: GPUBuffer, bias: GPUBuffer?) {
        let reading = [a, b, columnTable] + (bias.map { [$0] } ?? []) + (parameters.accumulate != 0 ? [c] : [])
        // The taller tile needs at least as many rows, or it computes rows that do not exist.
        let tileHeight = parameters.m >= 128 ? 128 : 64
        let threadgroups = MTLSize(width: (Int(parameters.n) + 63) / 64, height: (Int(parameters.m) + tileHeight - 1) / tileHeight, depth: 1)
        GPUContext.current.compute(GPUKernels.pipeline("implicit_gemm_\(tileHeight)", in: .convolution), reading: reading, writing: [c]) { arguments in
            arguments.buffer(a)
            arguments.buffer(b)
            arguments.buffer(c)
            arguments.buffer(columnTable)
            arguments.buffer(bias ?? a)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: threadgroups, threadgroup: MTLSize(width: tileHeight * 2, height: 1, depth: 1))
        }
    }

    private static func gatherColumns(_ source: GPUBuffer, rowStride: Int, table: GPUBuffer, rows: Int, columns: Int, into result: GPUBuffer) {
        let parameters = SIMD3<Int32>(Int32(rows), Int32(columns), Int32(rowStride))
        GPUContext.current.compute(GPUKernels.pipeline("gather_columns", in: .convolution), reading: [source, table], writing: [result]) { arguments in
            arguments.buffer(source)
            arguments.buffer(table)
            arguments.buffer(result)
            arguments.value(parameters)
            arguments.dispatch(threads: MTLSize(width: columns, height: rows, depth: 1), threadgroup: MTLSize(width: 32, height: 8, depth: 1))
        }
    }

    private static func floorQuotient(_ n: Int, _ d: Int) -> Int {
        n >= 0 ? n / d : -((-n + d - 1) / d)
    }

    private static func ceilingQuotient(_ n: Int, _ d: Int) -> Int {
        -floorQuotient(-n, d)
    }

    fileprivate enum Pass: Hashable {
        case forward
        case dataGradient
    }

    fileprivate struct TableKey: Hashable {
        let pass: Pass
        let inputChannels: Int, height: Int, width: Int, outputChannels: Int, kernelHeight: Int, kernelWidth: Int
        let outputHeight: Int, outputWidth: Int, padding: Int, stride: Int
        let phaseY: Int, phaseX: Int

        init(pass: Pass, convolution c: GPUImplicitConvolution, phaseY: Int, phaseX: Int) {
            self.pass = pass
            (inputChannels, height, width, outputChannels, kernelHeight, kernelWidth) = (c.inputChannels, c.height, c.width, c.outputChannels, c.kernelHeight, c.kernelWidth)
            (outputHeight, outputWidth, padding, stride) = (c.outputHeight, c.outputWidth, c.padding, c.stride)
            (self.phaseY, self.phaseX) = (phaseY, phaseX)
        }
    }
}

/// The offset tables of the implicit convolutions, computed once for every geometry and phase. The tables do not depend on the batch size.
private enum GPUImplicitConvolutionTables {
    struct Tables: @unchecked Sendable {
        // `@unchecked Sendable`: The buffers are written once, when they are created, and only read afterwards.

        /// Entries of the rows of B, four integers each, see `convolution.metal`
        let columns: GPUBuffer
        /// Columns of the filters of a phase of the data gradient, which `gather_columns` writes as a contiguous matrix
        let filterColumns: GPUBuffer?

        init(columns: [SIMD4<Int32>], filterColumns: [Int32]?) {
            self.columns = Self.upload(columns.flatMap { [$0.x, $0.y, $0.z, $0.w] })
            self.filterColumns = filterColumns.map(Self.upload)
        }

        private static func upload(_ values: [Int32]) -> GPUBuffer {
            let buffer = GPUMemoryOperators.allocateBuffer(withCapacity: values.count, type: Int32.self)
            values.withUnsafeBufferPointer { GPUMemoryOperators.assign(from: $0, to: buffer, count: values.count) }
            return buffer.memory
        }
    }

    private static let cache = Mutex<[GPUImplicitConvolution.TableKey: Tables]>([:])

    static func tables(for key: GPUImplicitConvolution.TableKey, create: () -> Tables) -> Tables {
        if let tables = cache.withLock({ $0[key] }) {
            return tables
        }
        let tables = create()
        cache.withLock { $0[key] = tables }
        return tables
    }
}

private struct ImplicitGemmParameters {
    var m: Int32, n: Int32, k: Int32
    var aRowStride: Int32
    var heightN: Int32, widthN: Int32
    var bBatch: Int32, bStrideY: Int32, bOffsetY: Int32, bStrideX: Int32, bOffsetX: Int32, bHeight: Int32, bWidth: Int32
    var cRowStride: Int32, cBatch: Int32, cStrideY: Int32, cOffsetY: Int32, cStrideX: Int32, cOffsetX: Int32, cWidth: Int32
    var accumulate: Int32, hasBias: Int32

    init(
        m: Int, n: Int, k: Int, aRowStride: Int, heightN: Int, widthN: Int,
        bBatch: Int, bStrideY: Int, bOffsetY: Int, bStrideX: Int, bOffsetX: Int, bHeight: Int, bWidth: Int,
        cRowStride: Int, cBatch: Int, cStrideY: Int, cOffsetY: Int, cStrideX: Int, cOffsetX: Int, cWidth: Int,
        accumulate: Bool, hasBias: Bool,
    ) {
        (self.m, self.n, self.k, self.aRowStride, self.heightN, self.widthN) = (Int32(m), Int32(n), Int32(k), Int32(aRowStride), Int32(heightN), Int32(widthN))
        (self.bBatch, self.bStrideY, self.bOffsetY, self.bStrideX, self.bOffsetX, self.bHeight, self.bWidth) = (Int32(bBatch), Int32(bStrideY), Int32(bOffsetY), Int32(bStrideX), Int32(bOffsetX), Int32(bHeight), Int32(bWidth))
        (self.cRowStride, self.cBatch, self.cStrideY, self.cOffsetY, self.cStrideX, self.cOffsetX, self.cWidth) = (Int32(cRowStride), Int32(cBatch), Int32(cStrideY), Int32(cOffsetY), Int32(cStrideX), Int32(cOffsetX), Int32(cWidth))
        (self.accumulate, self.hasBias) = (accumulate ? 1 : 0, hasBias ? 1 : 0)
    }
}
#endif

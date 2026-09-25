//
//  GPUMatrixMultiplication.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
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
import MetalPerformanceShaders
import Synchronization

/// Records matrix products of floats: with Metal Performance Shaders for large products, with the kernel of the package for the others.
///
/// A Metal Performance Shaders kernel reaches the highest throughput for large matrices, but it costs about 15 µs of host time
/// per product and opens its own encoder. The kernel of the package costs about 1 µs of host time and supports batches with any strides.
enum GPUMatrixMultiplication {
    /// Number of multiplications from which a product uses Metal Performance Shaders.
    static let largeProductMultiplications = 1 << 31

    private struct KernelKey: Hashable {
        var transposeLeft: Bool
        var transposeRight: Bool
        var rows: Int
        var columns: Int
        var inner: Int
        var alpha: Float
        var beta: Float
    }

    private struct UncheckedKernel: @unchecked Sendable {
        // `@unchecked Sendable`: The kernel has no state that changes when it encodes.
        let kernel: MPSMatrixMultiplication
    }

    private static let kernels = Mutex<[KernelKey: UncheckedKernel]>([:])
    /// Number of Metal Performance Shaders kernels that the cache keeps.
    private static let maximumCachedKernels = 256

    private struct GemmParameters {
        var rows: Int32
        var columns: Int32
        var inner: Int32
        var lda: Int32
        var ldb: Int32
        var ldc: Int32
        var transposeA: Int32
        var transposeB: Int32
        var alpha: Float
        var beta: Float
        var batchStrideA: Int64
        var batchStrideB: Int64
        var batchStrideC: Int64
        var splits: Int32 = 1
        var splitLength: Int32 = 0
    }

    /// Records the product `result = alpha * op(lhs) * op(rhs) + beta * result` of matrices of floats.
    ///
    /// - Parameters:
    ///   - lhsShape: Shape of the left matrix as stored, before the transposition.
    ///   - rhsShape: Shape of the right matrix as stored, before the transposition.
    ///   - rows: Number of rows of the result.
    ///   - columns: Number of columns of the result.
    ///   - inner: Number of columns of `op(lhs)`.
    static func encode(lhs: GPUBuffer, lhsShape: [Int], rhs: GPUBuffer, rhsShape: [Int], result: GPUBuffer, rows: Int, columns: Int, inner: Int, alpha: Float, beta: Float, transposeFirst: Bool, transposeSecond: Bool) {
        guard rows > 0, columns > 0 else {
            return
        }
        if !GPUContext.current.supportsMatrixKernels || rows * columns * inner >= largeProductMultiplications && rows >= 64 && columns >= 64 {
            encodeLarge(lhs: lhs, lhsShape: lhsShape, rhs: rhs, rhsShape: rhsShape, result: result, rows: rows, columns: columns, inner: inner, alpha: alpha, beta: beta, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
            return
        }
        encodeBatch(
            lhs: lhs, rhs: rhs, result: result, count: 1, rows: rows, columns: columns, inner: inner,
            lhsColumns: lhsShape[1], rhsColumns: rhsShape[1], strides: (0, 0, 0),
            alpha: alpha, beta: beta, transposeFirst: transposeFirst, transposeSecond: transposeSecond,
        )
    }

    /// Records the products of a batch of matrices with constant strides between the matrices of an operand.
    ///
    /// - Parameters:
    ///   - lhsColumns: Number of columns of every left matrix as stored.
    ///   - rhsColumns: Number of columns of every right matrix as stored.
    ///   - strides: Number of elements between neighboring matrices of the left operand, the right operand, and the result.
    ///     The stride 0 uses the same matrix for all products.
    static func encodeBatch(lhs: GPUBuffer, rhs: GPUBuffer, result: GPUBuffer, count: Int, rows: Int, columns: Int, inner: Int, lhsColumns: Int, rhsColumns: Int, strides: (lhs: Int, rhs: Int, result: Int), alpha: Float, beta: Float, transposeFirst: Bool, transposeSecond: Bool) {
        guard count > 0, rows > 0, columns > 0 else {
            return
        }
        let context = GPUContext.current
        guard context.supportsMatrixKernels else {
            let lhsShape = transposeFirst ? [inner, rows] : [rows, inner]
            let rhsShape = transposeSecond ? [columns, inner] : [inner, columns]
            for index in 0 ..< count {
                func matrix(_ buffer: GPUBuffer, stride: Int, elements: Int) -> GPUBuffer {
                    GPUBuffer(storage: buffer.storage, byteOffset: buffer.byteOffset + index * stride * 4, byteCount: elements * 4)
                }
                encodeLarge(
                    lhs: matrix(lhs, stride: strides.lhs, elements: rows * inner), lhsShape: lhsShape,
                    rhs: matrix(rhs, stride: strides.rhs, elements: inner * columns), rhsShape: rhsShape,
                    result: matrix(result, stride: strides.result, elements: rows * columns),
                    rows: rows, columns: columns, inner: inner, alpha: alpha, beta: beta, transposeFirst: transposeFirst, transposeSecond: transposeSecond,
                )
            }
            return
        }
        let parameters = GemmParameters(
            rows: Int32(rows), columns: Int32(columns), inner: Int32(inner),
            lda: Int32(lhsColumns), ldb: Int32(rhsColumns), ldc: Int32(columns),
            transposeA: transposeFirst ? 1 : 0, transposeB: transposeSecond ? 1 : 0,
            alpha: alpha, beta: beta,
            batchStrideA: Int64(strides.lhs), batchStrideB: Int64(strides.rhs), batchStrideC: Int64(strides.result),
            splitLength: Int32(inner),
        )
        if rows == 1 {
            // A product with one row is a vector-matrix product, which is limited by the reads of the matrix.
            let pipeline = GPUKernels.pipeline(transposeSecond ? "gemv_t" : "gemv_n", in: .matrix)
            context.compute(pipeline, reading: [lhs, rhs, result], writing: [result]) { arguments in
                arguments.buffer(lhs)
                arguments.buffer(rhs)
                arguments.buffer(result)
                arguments.value(parameters)
                if transposeSecond {
                    arguments.dispatch(threadgroups: MTLSize(width: (columns + 7) / 8, height: count, depth: 1), threadgroup: MTLSize(width: 256, height: 1, depth: 1))
                } else {
                    arguments.dispatch(threads: MTLSize(width: columns, height: count, depth: 1), threadgroup: MTLSize(width: min(columns, 256), height: 1, depth: 1))
                }
            }
            return
        }
        // Tiles of 32 x 32 elements give four times as many threadgroups, so that small products occupy all GPU cores.
        let largeTiles = ((rows + 63) / 64) * ((columns + 63) / 64) * count
        let tile = largeTiles >= 64 ? 64 : 32
        let tiles = ((rows + tile - 1) / tile) * ((columns + tile - 1) / tile) * count
        let name = "gemm_\(transposeFirst ? "t" : "n")\(transposeSecond ? "t" : "n")_\(tile)"
        let pipeline = GPUKernels.pipeline(name, in: .matrix)

        // A product with few tiles and a long inner axis, such as the weight gradient of a convolution, splits the inner axis
        // into parts, so that more threadgroups run. A second kernel adds the products of the parts.
        let splits = min(max(1, 128 / tiles), inner / 512)
        guard splits > 1 else {
            context.compute(pipeline, reading: [lhs, rhs, result], writing: [result]) { arguments in
                arguments.buffer(lhs)
                arguments.buffer(rhs)
                arguments.buffer(result)
                arguments.value(parameters)
                arguments.dispatch(
                    threadgroups: MTLSize(width: (columns + tile - 1) / tile, height: (rows + tile - 1) / tile, depth: count),
                    threadgroup: MTLSize(width: 128, height: 1, depth: 1),
                )
            }
            return
        }
        // Parts are multiples of the depth of a tile of the inner axis.
        let splitLength = ((inner + splits - 1) / splits + 31) / 32 * 32
        let splitCount = (inner + splitLength - 1) / splitLength
        let partial = GPUKernels.temporary(count: count * splitCount * rows * columns)
        var splitParameters = parameters
        splitParameters.alpha = 1
        splitParameters.beta = 0
        splitParameters.ldc = Int32(columns)
        splitParameters.batchStrideC = Int64(rows * columns)
        splitParameters.splits = Int32(splitCount)
        splitParameters.splitLength = Int32(splitLength)
        context.compute(pipeline, reading: [lhs, rhs], writing: [partial]) { arguments in
            arguments.buffer(lhs)
            arguments.buffer(rhs)
            arguments.buffer(partial)
            arguments.value(splitParameters)
            arguments.dispatch(
                threadgroups: MTLSize(width: (columns + tile - 1) / tile, height: (rows + tile - 1) / tile, depth: count * splitCount),
                threadgroup: MTLSize(width: 128, height: 1, depth: 1),
            )
        }
        var sumParameters = parameters
        sumParameters.splits = Int32(splitCount)
        let sum = GPUKernels.pipeline("gemm_split_sum", in: .matrix)
        context.compute(sum, reading: [partial, result], writing: [result]) { arguments in
            arguments.buffer(partial)
            arguments.buffer(result)
            arguments.value(sumParameters)
            arguments.dispatch(threads: MTLSize(width: columns, height: rows, depth: count), threadgroup: MTLSize(width: min(columns, 32), height: min(rows, 8), depth: 1))
        }
    }

    private static func encodeLarge(lhs: GPUBuffer, lhsShape: [Int], rhs: GPUBuffer, rhsShape: [Int], result: GPUBuffer, rows: Int, columns: Int, inner: Int, alpha: Float, beta: Float, transposeFirst: Bool, transposeSecond: Bool) {
        let context = GPUContext.current
        let key = KernelKey(transposeLeft: transposeFirst, transposeRight: transposeSecond, rows: rows, columns: columns, inner: inner, alpha: alpha, beta: beta)
        let kernel = kernels.withLock { kernels in
            if let kernel = kernels[key] {
                return kernel
            }
            // Shapes that change in every step, such as the lengths of padded sequences, would let the cache grow without a limit.
            if kernels.count >= maximumCachedKernels {
                kernels.removeAll()
            }
            // The creation autoreleases descriptors, see ``GPUContext``.
            let kernel = autoreleasepool { UncheckedKernel(kernel: MPSMatrixMultiplication(
                device: context.device,
                transposeLeft: transposeFirst,
                transposeRight: transposeSecond,
                resultRows: rows,
                resultColumns: columns,
                interiorColumns: inner,
                alpha: Double(alpha),
                beta: Double(beta),
            )) }
            kernels[key] = kernel
            return kernel
        }
        func matrix(_ buffer: GPUBuffer, rows: Int, columns: Int) -> MPSMatrix {
            let descriptor = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns * MemoryLayout<Float>.stride, dataType: .float32)
            return MPSMatrix(buffer: buffer.storage.buffer, offset: buffer.byteOffset, descriptor: descriptor)
        }
        context.commands(reading: [lhs, rhs, result], writing: [result]) { commandBuffer in
            // The matrices are created while the stream is locked, because a new storage can still replace its buffer.
            let left = matrix(lhs, rows: lhsShape[0], columns: lhsShape[1])
            let right = matrix(rhs, rows: rhsShape[0], columns: rhsShape[1])
            let target = matrix(result, rows: rows, columns: columns)
            kernel.kernel.encode(commandBuffer: commandBuffer, leftMatrix: left, rightMatrix: right, resultMatrix: target)
        }
    }
}
#endif

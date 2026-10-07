//
//  CPUKernels.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
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

import Foundation

/// Helpers for the fused kernels of the CPU.
///
/// The kernels process tensors in blocks or rows that fit into the L1 cache. Each block goes through the
/// accelerated primitives of ``CPUNumeric`` and through element-wise loops, so intermediate values do not leave the cache.
/// Scratch buffers are allocated for one call of a kernel and released at its end.
enum CPUKernels {
    // Some kernels keep a few scratch buffers of this size, which together fit into the L1 cache.
    /// Number of elements of a block.
    static let blockSize = 4096

    /// Upper bound for the number of elements of the scratch matrices of convolutions.
    static let maximumColumnElements = 1 << 20

    /// Calls `body` with the offset and the length of consecutive blocks of `count` elements.
    @inline(__always)
    static func forEachBlock(count: Int, blockSize: Int = blockSize, _ body: (_ offset: Int, _ length: Int) -> Void) {
        var offset = 0
        while offset < count {
            let length = Swift.min(blockSize, count - offset)
            body(offset, length)
            offset += length
        }
    }

    /// Calls `body` with the first row and the number of rows of consecutive blocks of rows.
    ///
    /// A block has at least one row and at most ``blockSize`` elements, unless a single row is longer.
    @inline(__always)
    static func forEachRowBlock(rows: Int, rowLength: Int, _ body: (_ firstRow: Int, _ rowCount: Int) -> Void) {
        let rowsPerBlock = Swift.max(1, blockSize / Swift.max(rowLength, 1))
        forEachBlock(count: rows, blockSize: rowsPerBlock, body)
    }

    /// Computes an element-wise function in blocks.
    ///
    /// `body` receives the input and the result of a block, offset to the block, and the length of the block.
    @inline(__always)
    static func map<N: NumericType>(_ input: ShapedBuffer<N, CPU>, into result: MutableShapedBuffer<N, CPU>, _ body: (_ x: UnsafePointer<N>, _ y: UnsafeMutablePointer<N>, _ length: Int) -> Void) {
        let (x, y) = (input.elementPointer, result.elementPointer)
        forEachBlock(count: input.count) { offset, length in
            body(x + offset, y + offset, length)
        }
    }

    @inline(__always)
    static func fill<N: NumericType>(_ pointer: UnsafeMutablePointer<N>, with value: N, count: Int) {
        N.fill(value: value, result: UnsafeMutableBufferPointer(start: pointer, count: count), count: count)
    }

    @inline(__always)
    static func sum<N: NumericType>(_ pointer: UnsafePointer<N>, count: Int) -> N {
        N.sum(val: UnsafeBufferPointer(start: pointer, count: count), count: count)
    }

    @inline(__always)
    static func dot<N: NumericType>(_ lhs: UnsafePointer<N>, _ rhs: UnsafePointer<N>, count: Int) -> N {
        N.dot(lhs: UnsafeBufferPointer(start: lhs, count: count), rhs: UnsafeBufferPointer(start: rhs, count: count), count: count)
    }

    @inline(__always)
    static func maximum<N: NumericType>(_ pointer: UnsafePointer<N>, count: Int) -> N {
        N.argmax(values: UnsafeBufferPointer(start: pointer, count: count), count: count).1
    }

    @inline(__always)
    static func exp<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        N.exp(val: UnsafeBufferPointer(start: values, count: count), result: UnsafeMutableBufferPointer(start: result, count: count), count: count)
    }

    @inline(__always)
    static func log<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        N.log(val: UnsafeBufferPointer(start: values, count: count), result: UnsafeMutableBufferPointer(start: result, count: count), count: count)
    }

    @inline(__always)
    static func tanh<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        N.tanh(val: UnsafeBufferPointer(start: values, count: count), result: UnsafeMutableBufferPointer(start: result, count: count), count: count)
    }

    @inline(__always)
    static func sqrt<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        N.sqrt(val: UnsafeBufferPointer(start: values, count: count), result: UnsafeMutableBufferPointer(start: result, count: count), count: count)
    }

    // The sigmoid follows from it without an overflow for large magnitudes, and kernels that use the sigmoid in a further
    // loop apply the last step there.
    /// Computes `tanh(scale * x / 2)`, from which `sigmoid(scale * x) = tanh(scale * x / 2) / 2 + 1 / 2` follows.
    ///
    /// The input and the result can be the same memory.
    @inline(__always)
    static func tanhOfHalf<N: NumericType>(_ x: UnsafePointer<N>, scale: N = 1, into result: UnsafeMutablePointer<N>, count: Int) {
        let factor = scale * N(0.5)
        for i in 0 ..< count {
            result[i] = x[i] * factor
        }
        tanh(result, into: result, count: count)
    }

    /// Computes `sigmoid(x)` as `tanh(x / 2) / 2 + 1 / 2`, which does not overflow for large magnitudes.
    ///
    /// The input and the result can be the same memory.
    @inline(__always)
    static func sigmoid<N: NumericType>(_ x: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        let half = N(0.5)
        tanhOfHalf(x, into: result, count: count)
        for i in 0 ..< count {
            result[i] = result[i] * half + half
        }
    }

    /// Computes the softmax of every row: `exp(x - max(x)) / sum(exp(x - max(x)))`.
    ///
    /// `scratch` holds `rows * rowLength` elements. The values and the scratch buffer can be the same memory.
    @inline(__always)
    static func softmaxRows<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, scratch: UnsafeMutablePointer<N>, rows: Int, rowLength: Int) {
        subtractRowMaxima(values, into: scratch, rows: rows, rowLength: rowLength)
        exp(scratch, into: result, count: rows * rowLength)
        for row in 0 ..< rows {
            let rowResult = result + row * rowLength
            let inverseSum = 1 / sum(rowResult, count: rowLength)
            for j in 0 ..< rowLength {
                rowResult[j] *= inverseSum
            }
        }
    }

    /// Computes the gradient of the softmax of every row, `scale * output * (outputGradient - sum(outputGradient * output))`.
    ///
    /// `scratch` holds `rowLength` elements. The gradient of the result and the result can be the same memory.
    @inline(__always)
    static func softmaxRowsBackward<N: NumericType>(
        output: UnsafePointer<N>,
        outputGradient: UnsafePointer<N>,
        scale: N = 1,
        into result: UnsafeMutablePointer<N>,
        scratch: UnsafeMutablePointer<N>,
        rows: Int,
        rowLength: Int,
    ) {
        for row in 0 ..< rows {
            let start = row * rowLength
            let (y, g, dx) = (output + start, outputGradient + start, result + start)
            for j in 0 ..< rowLength {
                scratch[j] = g[j] * y[j]
            }
            let product = sum(scratch, count: rowLength)
            for j in 0 ..< rowLength {
                dx[j] = y[j] * (g[j] - product) * scale
            }
        }
    }

    /// Subtracts the maximum of every row from the row, so that the exponentials do not overflow.
    @inline(__always)
    static func subtractRowMaxima<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, rows: Int, rowLength: Int) {
        for row in 0 ..< rows {
            let (source, target) = (values + row * rowLength, result + row * rowLength)
            let maximum = maximum(source, count: rowLength)
            for j in 0 ..< rowLength {
                target[j] = source[j] - maximum
            }
        }
    }

    /// Computes `target = values + beta * target` for a beta of 0 or 1. With beta 0, the target need not be initialized.
    @inline(__always)
    static func store<N: NumericType>(_ values: UnsafePointer<N>, into target: UnsafeMutablePointer<N>, beta: N, count: Int) {
        if beta == 0 {
            target.update(from: values, count: count)
        } else {
            for i in 0 ..< count {
                target[i] += values[i]
            }
        }
    }

    /// Copies rows between the layouts [first, second, length] and [second, first, length], and computes `target = values + beta * target` for a beta of 0 or 1.
    ///
    /// With offsets, every row of the first index `i` gets `offsets[i]` added. Convolutions use it to move between the layout of
    /// a matrix product, [channels, images, pixels], and the layout of images, [images, channels, pixels].
    @inline(__always)
    static func swapLeadingAxes<N: NumericType>(_ values: UnsafePointer<N>, first: Int, second: Int, length: Int, adding offsets: UnsafePointer<N>? = nil, into target: UnsafeMutablePointer<N>, beta: N = 0) {
        // The rows of the target are written in order.
        for j in 0 ..< second {
            for i in 0 ..< first {
                let (source, destination) = (values + (i * second + j) * length, target + (j * first + i) * length)
                guard let offsets else {
                    store(source, into: destination, beta: beta, count: length)
                    continue
                }
                let offset = offsets[i]
                if beta == 0 {
                    for k in 0 ..< length {
                        destination[k] = source[k] + offset
                    }
                } else {
                    for k in 0 ..< length {
                        destination[k] += source[k] + offset
                    }
                }
            }
        }
    }

    /// Repeats the values along the axes that broadcast the shape of the values to the shape of the result.
    /// The shape of the values must broadcast to the shape of the result.
    static func broadcast<N: NumericType>(_ values: ShapedBuffer<N, CPU>, into result: MutableShapedBuffer<N, CPU>) {
        precondition(ShapeUtil.broadcasts(values.shape, to: result.shape), "The values do not broadcast to the result.")
        let shape = result.shape
        let sourceShape = Array(repeating: 1, count: shape.count - values.dim) + values.shape
        // A broadcast axis has the stride 0, so every position along it reads the same element.
        // The last axis is copied or filled as a row.
        let denseStrides = MemoryOps.strides(from: sourceShape)
        let sourceStrides = zip(sourceShape, denseStrides).map { $0 == 1 ? 0 : $1 }
        let (source, target) = (values.elementPointer, result.elementPointer)
        let rowLength = shape.last ?? 1
        let rowStride = sourceStrides.last ?? 0
        StridedIteration.forEachOffset(shape: Array(shape.dropLast()), strides: Array(sourceStrides.dropLast()), Array(MemoryOps.strides(from: shape).dropLast())) { sourceOffset, targetOffset in
            if rowStride == 0 {
                fill(target + targetOffset, with: source[sourceOffset], count: rowLength)
            } else {
                (target + targetOffset).update(from: source + sourceOffset, count: rowLength)
            }
        }
    }

    /// Multiplies two row-major matrices: `result = alpha * op(lhs) × op(rhs) + beta * result`.
    @inline(__always)
    static func gemm<N: NumericType>(
        _ lhs: UnsafePointer<N>,
        shape lhsShape: (Int, Int),
        lhsTransposed transposeFirst: Bool = false,
        _ rhs: UnsafePointer<N>,
        shape rhsShape: (Int, Int),
        rhsTransposed transposeSecond: Bool = false,
        into result: UnsafeMutablePointer<N>,
        alpha: N = 1,
        beta: N = 0,
    ) {
        let resultShape = (transposeFirst ? lhsShape.1 : lhsShape.0, transposeSecond ? rhsShape.0 : rhsShape.1)
        N.gemm(
            lhs: UnsafeBufferPointer(start: lhs, count: lhsShape.0 * lhsShape.1),
            rhs: UnsafeBufferPointer(start: rhs, count: rhsShape.0 * rhsShape.1),
            result: UnsafeMutableBufferPointer(start: result, count: resultShape.0 * resultShape.1),
            lhsShape: lhsShape,
            rhsShape: rhsShape,
            resultShape: resultShape,
            alpha: alpha,
            beta: beta,
            transposeFirst: transposeFirst,
            transposeSecond: transposeSecond,
        )
    }
}

extension GradientBuffer where Device == CPU {
    /// The elements of the gradient and the factor of their current values, for a kernel that writes every element once:
    /// `elements = gradient + beta * elements`.
    func elementsToWrite() -> (elements: UnsafeMutablePointer<Element>, beta: Element) {
        (values.elementPointer, beta)
    }

    /// The elements of the gradient for a kernel that adds to them in several steps. A gradient that is not added to starts at 0.
    func elementsToAddTo() -> UnsafeMutablePointer<Element> {
        if !adds {
            CPUKernels.fill(values.elementPointer, with: 0, count: values.count)
        }
        return values.elementPointer
    }

    /// Writes the given values into the gradient, or adds them.
    func write(_ gradient: UnsafePointer<Element>) {
        CPUKernels.store(gradient, into: values.elementPointer, beta: beta, count: values.count)
    }

    /// Writes the gradient in blocks of ``CPUKernels/blockSize`` elements.
    ///
    /// `body` receives the offset and the length of a block and writes the gradient of the elements of the block to the given memory.
    /// The helper adds it to the elements, or stores it.
    @inline(__always)
    func writeBlocks(_ body: (_ offset: Int, _ length: Int, _ block: UnsafeMutablePointer<Element>) -> Void) {
        let elements = values.elementPointer
        guard adds else {
            CPUKernels.forEachBlock(count: values.count) { offset, length in
                body(offset, length, elements + offset)
            }
            return
        }
        let block = UnsafeMutablePointer<Element>.allocate(capacity: CPUKernels.blockSize)
        defer {
            block.deallocate()
        }
        CPUKernels.forEachBlock(count: values.count) { offset, length in
            body(offset, length, block)
            CPUKernels.store(block, into: elements + offset, beta: 1, count: length)
        }
    }
}

// The fused operations do not accept empty buffers, see `FusedOperationsType`. Every kernel reads its buffers through
// `elementPointer`, so this is the one place that checks it.

extension ShapedBuffer where Device == CPU {
    /// Pointer to the elements of the buffer. The buffer must not be empty.
    var elementPointer: UnsafePointer<Element> {
        precondition(count > 0, "The fused operations do not accept empty buffers.")
        return immutable.baseAddress!
    }
}

extension MutableShapedBuffer where Device == CPU {
    /// Pointer to the elements of the buffer. The buffer must not be empty.
    var elementPointer: UnsafeMutablePointer<Element> {
        precondition(count > 0, "The fused operations do not accept empty buffers.")
        return pointer.baseAddress!
    }
}

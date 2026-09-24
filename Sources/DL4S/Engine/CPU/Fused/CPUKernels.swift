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
enum CPUKernels {
    /// Number of elements of a block. Some kernels keep a few scratch buffers of this size, which together fit into the L1 cache.
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

    /// Allocates an uninitialized scratch buffer for the duration of `body`.
    @inline(__always)
    static func withScratch<N, Result>(_ type: N.Type, count: Int, _ body: (UnsafeMutablePointer<N>) -> Result) -> Result {
        let scratch = UnsafeMutablePointer<N>.allocate(capacity: Swift.max(count, 1))
        defer {
            scratch.deallocate()
        }
        return body(scratch)
    }

    /// Creates a tensor without context and returns it together with a pointer to its uninitialized elements.
    ///
    /// The pointer is valid while the tensor is alive.
    @inline(__always)
    static func makeTensor<N: NumericType>(shape: [Int]) -> (Tensor<N, CPU>, UnsafeMutablePointer<N>) {
        let buffer = CPU.Memory.allocateBuffer(withShape: shape, type: N.self)
        return (Tensor(using: buffer, context: nil), buffer.pointer.baseAddress!)
    }

    /// Creates a tensor without context whose elements are 0.
    @inline(__always)
    static func makeZeroTensor<N: NumericType>(shape: [Int]) -> (Tensor<N, CPU>, UnsafeMutablePointer<N>) {
        let (tensor, pointer) = makeTensor(shape: shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        fill(pointer, with: 0, count: tensor.count)
        return (tensor, pointer)
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

extension Tensor where Device == CPU {
    /// Pointer to the elements of the tensor. It is valid while the tensor is alive.
    var elementPointer: UnsafePointer<Element> {
        UnsafePointer(handle.values.pointer.baseAddress!)
    }
}

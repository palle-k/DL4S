//
//  CPUFusedSoftmax.swift
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

// The kernels normalize along the last axis, so every row is contiguous. They process blocks of rows:
// the exponentials of a block go through one vectorized call, and the reductions run per row while the row is in the cache.
// Other axes use the default implementations.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softmax<N: NumericType>(input: ShapedBuffer<N, CPU>, axis: Int, result: MutableShapedBuffer<N, CPU>) {
        // The kernel supports the last axis.
        guard axis == input.dim - 1 else {
            DefaultFusedOperations<CPU>.softmax(input: input, axis: axis, result: result)
            return
        }
        let rowLength = input.shape[axis]
        let (x, y) = (input.elementPointer, result.elementPointer)
        let shifted = UnsafeMutablePointer<N>.allocate(capacity: Swift.max(CPUKernels.blockSize, rowLength))
        defer {
            shifted.deallocate()
        }
        CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
            let offset = firstRow * rowLength
            CPUKernels.softmaxRows(x + offset, into: y + offset, scratch: shifted, rows: rowCount, rowLength: rowLength)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softmaxBackward<N: NumericType>(output: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, axis: Int, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        // The kernel supports the last axis.
        guard axis == output.dim - 1 else {
            DefaultFusedOperations<CPU>.softmaxBackward(output: output, outputGradient: outputGradient, axis: axis, inputGradient: inputGradient)
            return
        }
        let rowLength = output.shape[axis]
        let (y, g) = (output.elementPointer, outputGradient.elementPointer)
        let (dx, beta) = inputGradient.elementsToWrite()
        let products = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        let rowGradient = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        defer {
            products.deallocate()
            rowGradient.deallocate()
        }
        for row in 0 ..< output.count / rowLength {
            let start = row * rowLength
            let target = beta == 0 ? dx + start : rowGradient
            CPUKernels.softmaxRowsBackward(output: y + start, outputGradient: g + start, into: target, scratch: products, rows: 1, rowLength: rowLength)
            if beta != 0 {
                CPUKernels.store(rowGradient, into: dx + start, beta: 1, count: rowLength)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func logSoftmax<N: NumericType>(input: ShapedBuffer<N, CPU>, axis: Int, result: MutableShapedBuffer<N, CPU>) {
        // The kernel supports the last axis.
        guard axis == input.dim - 1 else {
            DefaultFusedOperations<CPU>.logSoftmax(input: input, axis: axis, result: result)
            return
        }
        let rowLength = input.shape[axis]
        let (x, y) = (input.elementPointer, result.elementPointer)
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: Swift.max(CPUKernels.blockSize, rowLength))
        defer {
            exponentials.deallocate()
        }
        CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
            let offset = firstRow * rowLength
            CPUKernels.subtractRowMaxima(x + offset, into: y + offset, rows: rowCount, rowLength: rowLength)
            CPUKernels.exp(y + offset, into: exponentials, count: rowCount * rowLength)
            for row in 0 ..< rowCount {
                let values = y + offset + row * rowLength
                let logSum = CPUKernels.sum(exponentials + row * rowLength, count: rowLength).log()
                for j in 0 ..< rowLength {
                    values[j] -= logSum
                }
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func logSoftmaxBackward<N: NumericType>(output: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, axis: Int, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        // The kernel supports the last axis.
        guard axis == output.dim - 1 else {
            DefaultFusedOperations<CPU>.logSoftmaxBackward(output: output, outputGradient: outputGradient, axis: axis, inputGradient: inputGradient)
            return
        }
        let rowLength = output.shape[axis]
        let (y, g) = (output.elementPointer, outputGradient.elementPointer)
        let (dx, beta) = inputGradient.elementsToWrite()
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: Swift.max(CPUKernels.blockSize, rowLength))
        defer {
            exponentials.deallocate()
        }
        CPUKernels.forEachRowBlock(rows: output.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
            let offset = firstRow * rowLength
            CPUKernels.exp(y + offset, into: exponentials, count: rowCount * rowLength)
            for row in 0 ..< rowCount {
                let start = offset + row * rowLength
                let (gr, er, target) = (g + start, exponentials + row * rowLength, dx + start)
                let gradientSum = CPUKernels.sum(gr, count: rowLength)
                // dx = g - exp(y) * sum(g)
                if beta == 0 {
                    for j in 0 ..< rowLength {
                        target[j] = gr[j] - er[j] * gradientSum
                    }
                } else {
                    for j in 0 ..< rowLength {
                        target[j] += gr[j] - er[j] * gradientSum
                    }
                }
            }
        }
    }
}

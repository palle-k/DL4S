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
    static func softmax<N: NumericType>(input: Tensor<N, CPU>, axis: Int) -> Tensor<N, CPU> {
        guard input.count > 0, axis == input.dim - 1 else {
            return DefaultFusedOperations<CPU>.softmax(input: input, axis: axis)
        }
        let rowLength = input.shape[axis]
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        CPUKernels.withScratch(N.self, count: Swift.max(CPUKernels.blockSize, rowLength)) { t in
            CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                let offset = firstRow * rowLength
                subtractRowMaxima(x + offset, into: t, rows: rowCount, rowLength: rowLength)
                CPUKernels.exp(t, into: y + offset, count: rowCount * rowLength)
                for row in 0 ..< rowCount {
                    let values = y + offset + row * rowLength
                    let inverseSum = 1 / CPUKernels.sum(values, count: rowLength)
                    for j in 0 ..< rowLength {
                        values[j] *= inverseSum
                    }
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softmaxBackward<N: NumericType>(output: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, axis: Int, accumulating gradient: inout Tensor<N, CPU>?) {
        guard output.count > 0, axis == output.dim - 1, output.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.softmaxBackward(output: output, outputGradient: outputGradient, axis: axis, accumulating: &gradient)
            return
        }
        let rowLength = output.shape[axis]
        let (result, dx) = CPUKernels.makeTensor(shape: output.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let y = output.elementPointer
        let g = outputGradient.elementPointer
        CPUKernels.withScratch(N.self, count: Swift.max(CPUKernels.blockSize, rowLength)) { t in
            CPUKernels.forEachRowBlock(rows: output.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                let offset = firstRow * rowLength
                for i in 0 ..< rowCount * rowLength {
                    t[i] = g[offset + i] * y[offset + i]
                }
                for row in 0 ..< rowCount {
                    let start = offset + row * rowLength
                    let product = CPUKernels.sum(t + row * rowLength, count: rowLength)
                    for j in start ..< start + rowLength {
                        dx[j] = y[j] * (g[j] - product)
                    }
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func logSoftmax<N: NumericType>(input: Tensor<N, CPU>, axis: Int) -> Tensor<N, CPU> {
        guard input.count > 0, axis == input.dim - 1 else {
            return DefaultFusedOperations<CPU>.logSoftmax(input: input, axis: axis)
        }
        let rowLength = input.shape[axis]
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        CPUKernels.withScratch(N.self, count: Swift.max(CPUKernels.blockSize, rowLength)) { e in
            CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                let offset = firstRow * rowLength
                subtractRowMaxima(x + offset, into: y + offset, rows: rowCount, rowLength: rowLength)
                CPUKernels.exp(y + offset, into: e, count: rowCount * rowLength)
                for row in 0 ..< rowCount {
                    let values = y + offset + row * rowLength
                    let logSum = CPUKernels.sum(e + row * rowLength, count: rowLength).log()
                    for j in 0 ..< rowLength {
                        values[j] -= logSum
                    }
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func logSoftmaxBackward<N: NumericType>(output: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, axis: Int, accumulating gradient: inout Tensor<N, CPU>?) {
        guard output.count > 0, axis == output.dim - 1, output.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.logSoftmaxBackward(output: output, outputGradient: outputGradient, axis: axis, accumulating: &gradient)
            return
        }
        let rowLength = output.shape[axis]
        let (result, dx) = CPUKernels.makeTensor(shape: output.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let y = output.elementPointer
        let g = outputGradient.elementPointer
        CPUKernels.withScratch(N.self, count: Swift.max(CPUKernels.blockSize, rowLength)) { e in
            CPUKernels.forEachRowBlock(rows: output.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                let offset = firstRow * rowLength
                CPUKernels.exp(y + offset, into: e, count: rowCount * rowLength)
                for row in 0 ..< rowCount {
                    let start = offset + row * rowLength
                    let gradientSum = CPUKernels.sum(g + start, count: rowLength)
                    for j in 0 ..< rowLength {
                        dx[start + j] = g[start + j] - e[row * rowLength + j] * gradientSum
                    }
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }
}

extension CPUFusedOperations {
    /// Subtracts the maximum of every row from the row, so that the exponentials do not overflow.
    @inline(__always)
    static func subtractRowMaxima<N: NumericType>(_ values: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, rows: Int, rowLength: Int) {
        for row in 0 ..< rows {
            let start = row * rowLength
            let maximum = CPUKernels.maximum(values + start, count: rowLength)
            for j in start ..< start + rowLength {
                result[j] = values[j] - maximum
            }
        }
    }
}

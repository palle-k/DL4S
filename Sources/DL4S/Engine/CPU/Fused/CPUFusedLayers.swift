//
//  CPUFusedLayers.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.09.26.
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

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func linear<N: NumericType>(input: ShapedBuffer<N, CPU>, weights: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, result: MutableShapedBuffer<N, CPU>) {
        precondition(input.dim == 2 && weights.dim == 2, "The input and the weights must be matrices.")
        precondition(input.shape[1] == weights.shape[0], "The input must have one column for every row of the weights.")
        precondition(bias.map { $0.shape == [weights.shape[1]] } ?? true, "The bias must have one element for every column of the weights.")
        let (rows, inputSize, outputSize) = (input.shape[0], weights.shape[0], weights.shape[1])
        let y = result.elementPointer
        // Every row of the result starts with the bias, and the product is added to it.
        if let bias {
            let b = bias.elementPointer
            for row in 0 ..< rows {
                (y + row * outputSize).update(from: b, count: outputSize)
            }
        }
        CPUKernels.gemm(input.elementPointer, shape: (rows, inputSize), weights.elementPointer, shape: (inputSize, outputSize), into: y, beta: bias == nil ? 0 : 1)
    }
}

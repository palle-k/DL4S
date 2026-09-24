//
//  CPUFusedOptimizers.swift
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

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func adamUpdate<N: NumericType>(
        parameter: Tensor<N, CPU>,
        gradient: Tensor<N, CPU>,
        firstMoment: inout Tensor<N, CPU>,
        secondMoment: inout Tensor<N, CPU>,
        secondMomentMax: inout Tensor<N, CPU>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
    ) -> Tensor<N, CPU> {
        let shape = parameter.shape
        guard parameter.count > 0, gradient.shape == shape, firstMoment.shape == shape, secondMoment.shape == shape, secondMomentMax.map({ $0.shape == shape }) ?? true else {
            return DefaultFusedOperations<CPU>.adamUpdate(
                parameter: parameter, gradient: gradient, firstMoment: &firstMoment, secondMoment: &secondMoment, secondMomentMax: &secondMomentMax,
                learningRate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon, beta1Power: beta1Power, beta2Power: beta2Power,
            )
        }
        let (result, updated) = CPUKernels.makeTensor(shape: shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (p, g) = (parameter.elementPointer, gradient.elementPointer)
        // The optimizer owns the moments, so the writes do not copy them.
        let m = firstMoment.mutableValues.pointer.baseAddress!
        let v = secondMoment.mutableValues.pointer.baseAddress!
        var maximum: UnsafeMutablePointer<N>?
        if secondMomentMax != nil {
            maximum = secondMomentMax!.mutableValues.pointer.baseAddress!
        }
        let (firstRate, secondRate) = (1 - beta1, 1 - beta2)
        let firstCorrection = 1 / (1 - beta1Power)
        let secondCorrection = 1 / (1 - beta2Power)

        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (corrected, root) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: parameter.count) { offset, length in
                let (pb, gb, mb, vb, ub) = (p + offset, g + offset, m + offset, v + offset, updated + offset)
                for i in 0 ..< length {
                    let gradient = gb[i]
                    mb[i] = beta1 * mb[i] + firstRate * gradient
                    vb[i] = beta2 * vb[i] + secondRate * gradient * gradient
                }
                if let maximumBlock = maximum.map({ $0 + offset }) {
                    for i in 0 ..< length {
                        let largest = Swift.max(maximumBlock[i], vb[i])
                        maximumBlock[i] = largest
                        corrected[i] = largest * secondCorrection
                    }
                } else {
                    for i in 0 ..< length {
                        corrected[i] = vb[i] * secondCorrection
                    }
                }
                CPUKernels.sqrt(corrected, into: root, count: length)
                for i in 0 ..< length {
                    ub[i] = pb[i] - learningRate / (root[i] + epsilon) * (mb[i] * firstCorrection)
                }
            }
        }
        return result
    }
}

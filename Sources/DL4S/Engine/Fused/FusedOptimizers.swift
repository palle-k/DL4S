//
//  FusedOptimizers.swift
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

// MARK: Default implementations

public extension FusedOperationsType {
    static func adamUpdate<N: NumericType>(
        parameter: Tensor<N, Device>,
        gradient: Tensor<N, Device>,
        firstMoment: inout Tensor<N, Device>,
        secondMoment: inout Tensor<N, Device>,
        secondMomentMax: inout Tensor<N, Device>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
    ) -> Tensor<N, Device> {
        let gradient = gradient.detached()
        firstMoment = firstMoment.detached() * Tensor(beta1) + gradient * Tensor(1 - beta1)
        secondMoment = secondMoment.detached() * Tensor(beta2) + gradient * gradient * Tensor(1 - beta2)

        var normalizer = secondMoment
        if let maximum = secondMomentMax {
            normalizer = Tensor.max(maximum.detached(), secondMoment)
            secondMomentMax = normalizer
        }
        let correctedFirstMoment = firstMoment / Tensor(1 - beta1Power)
        let correctedSecondMoment = normalizer / Tensor(1 - beta2Power)
        return parameter.detached() - Tensor(learningRate) / (correctedSecondMoment.sqrt() + Tensor(epsilon)) * correctedFirstMoment
    }
}

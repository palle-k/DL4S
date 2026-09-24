//
//  FusedLayers.swift
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
    static func linear<N: NumericType>(input: Tensor<N, Device>, weights: Tensor<N, Device>, bias: Tensor<N, Device>?) -> Tensor<N, Device> {
        let product = input.detached().matrixMultiplied(with: weights.detached())
        guard let bias else {
            return product
        }
        return product + bias.detached()
    }

    static func linearBackward<N: NumericType>(input: Tensor<N, Device>, weights: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, accumulating gradients: inout (input: Tensor<N, Device>?, weights: Tensor<N, Device>?, bias: Tensor<N, Device>?)) {
        let outputGradient = outputGradient.detached()
        // The products are added to the accumulated gradients in place, so a weight that is used several times needs no temporary gradient.
        if input.requiresGradient {
            Tensor.accumulateProduct(outputGradient, weights.detached(), transposeRhs: true, into: &gradients.input)
        }
        if weights.requiresGradient {
            Tensor.accumulateProduct(input.detached(), outputGradient, transposeLhs: true, into: &gradients.weights)
        }
        if bias?.requiresGradient ?? false {
            Tensor.accumulate(outputGradient.reduceSum(along: [0]), into: &gradients.bias)
        }
    }

    static func dropout<N: NumericType>(input: Tensor<N, Device>, rate: Float) -> (output: Tensor<N, Device>, mask: Tensor<N, Device>) {
        let mask = Tensor<N, Device>(bernoulliDistributedWithShape: input.shape, probability: 1 - rate)
        return (input.detached() * mask, mask)
    }

    static func dropoutBackward<N: NumericType>(mask: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradient: inout Tensor<N, Device>?) {
        Tensor.accumulate(
            outputGradient.detached() * mask.detached(),
            into: &gradient,
        )
    }
}

// MARK: Composed gradients

extension Composed {
    static func linearGradients<N, Device>(
        input: Tensor<N, Device>,
        weights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesInput: Bool,
        computesWeights: Bool,
        computesBias: Bool,
    ) -> (input: Tensor<N, Device>?, weights: Tensor<N, Device>?, bias: Tensor<N, Device>?) {
        let inputGradient = computesInput ? outputGradient.matrixMultiplied(with: weights, transposeOther: true) : nil
        let weightGradient = computesWeights ? input.matrixMultiplied(with: outputGradient, transposeSelf: true) : nil
        let biasGradient = computesBias ? outputGradient.reduceSum(along: [0]) : nil
        return (inputGradient, weightGradient, biasGradient)
    }
}

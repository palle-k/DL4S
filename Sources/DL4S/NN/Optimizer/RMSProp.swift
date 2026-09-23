//
//  RMSProp.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
//  Copyright (c) 2019 - 2026 - Palle Klewitz
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

/// Root mean square optimizer
///
/// Unpublished, proposed by [Geoffrey Hinton - Neural Networks for Machine Learning](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
public struct RMSProp<Element: NumericType, Device: DeviceType>: Optimizer, Sendable {
    public typealias ParamTensor = Tensor<Element, Device>

    /// Learning rate scaling factor
    public var learningRate: ParamTensor

    /// Exponential decay rate for gradient history
    public var gamma: ParamTensor

    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor

    private var gradientSums: [ParamTensor] = []

    /// Root mean square optimizer
    ///
    /// Unpublished, proposed by [Geoffrey Hinton - Neural Networks for Machine Learning](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    /// - Parameters:
    ///   - learningRate: Learning rate scaling factor
    ///   - gamma: Exponential decay rate for gradient history
    ///   - epsilon: Normalization scalar added to divisors
    public init(learningRate: ParamTensor = 0.001, gamma: ParamTensor = 0.9, epsilon: ParamTensor = 1e-8) {
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
    }

    public mutating func reset() {
        gradientSums = []
    }

    /// Reports the moving average of squared gradients of the weight at position `i` as `gradientSums.i`.
    public mutating func visitTensors(_ visitor: inout TensorVisitor<Element, Device>) {
        visitor.frozen(&gradientSums, named: "gradientSums")
    }

    /// Creates the gradient sums of the layout.
    public mutating func adoptLayout(_ layout: TensorLayout) {
        gradientSums = Self.zeroState(for: layout.children(of: "gradientSums"))
    }

    public mutating func update(_ parameters: inout [ParamTensor], along gradients: [ParamTensor]) {
        Self.validateGradients(gradients, against: parameters)
        Self.initializeStateIfNeeded(&gradientSums, for: parameters)

        for index in parameters.indices {
            let grad = gradients[index].detached()

            let addedToGradientSum = (1 - gamma) * (grad * grad)
            gradientSums[index] = gamma * gradientSums[index] + addedToGradientSum

            let a = learningRate / sqrt(gradientSums[index] + epsilon)
            let delta = a * grad

            parameters[index] -= delta
            parameters[index].discardContext()
        }
    }
}

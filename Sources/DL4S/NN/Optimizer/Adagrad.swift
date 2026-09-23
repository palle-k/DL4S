//
//  Adagrad.swift
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

/// Adagrad optimizer
///
/// Follows [Duchi et al - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
public struct Adagrad<Element: NumericType, Device: DeviceType>: Optimizer, Sendable {
    public typealias ParamTensor = Tensor<Element, Device>

    /// Learning rate scaling factor
    public var learningRate: ParamTensor

    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor

    private var gradientSums: [ParamTensor] = []

    /// Adagrad optimizer
    ///
    /// Follows [Duchi et al - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    /// - Parameters:
    ///   - learningRate: Learning rate scaling factor
    ///   - epsilon: Normalization scalar added to divisors
    public init(learningRate: ParamTensor, epsilon: ParamTensor = 1e-8) {
        self.learningRate = learningRate
        self.epsilon = epsilon
    }

    public mutating func reset() {
        gradientSums = []
    }

    /// Reports the sum of squared gradients of the weight at position `i` as `gradientSums.i`.
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

            gradientSums[index] += grad * grad

            let adaptiveLearningRate = learningRate / sqrt(gradientSums[index] + epsilon)

            parameters[index] -= adaptiveLearningRate * grad
            parameters[index].discardContext()
        }
    }
}

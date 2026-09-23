//
//  SGD.swift
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

/// 'Vanilla' stochastic gradient descent optimizer
public struct SGD<Element: NumericType, Device: DeviceType>: Optimizer, Sendable {
    public typealias ParamTensor = Tensor<Element, Device>

    /// Learning rate with which to move along the gradient
    public var learningRate: ParamTensor

    /// 'Vanilla' stochastic gradient descent optimizer
    /// - Parameter learningRate: Learning rate with which to move along the gradient
    public init(learningRate: ParamTensor) {
        self.learningRate = learningRate
    }

    public mutating func update(_ parameters: inout [ParamTensor], along gradients: [ParamTensor]) {
        Self.validateGradients(gradients, against: parameters)
        for index in parameters.indices {
            parameters[index] -= learningRate * gradients[index].detached()
            parameters[index].discardContext()
        }
    }

    /// Stochastic gradient descent has no state, so this method does nothing.
    public mutating func reset() {}

    /// Stochastic gradient descent has no state, so this method reports no tensors.
    public mutating func visitTensors(_ visitor: inout TensorVisitor<Element, Device>) {}
}

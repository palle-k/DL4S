//
//  Momentum.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.09.20.
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

/// Gradient descent optimizer with momentum
public struct Momentum<Element: NumericType, Device: DeviceType>: Optimizer, Sendable {
    public typealias ParamTensor = Tensor<Element, Device>

    /// Learning rate with which to move along the gradient
    public var learningRate: ParamTensor

    /// Decay rate of momentum that is built up, when subsequent gradient updates move in the same direction
    public var momentum: ParamTensor

    private var velocities: [ParamTensor] = []

    /// Gradient descent optimizer with momentum
    /// - Parameters:
    ///   - learningRate: Learning rate with which to move along the gradient
    ///   - momentum: Decay rate of momentum that is built up, when subsequent gradient updates move in the same direction
    public init(learningRate: ParamTensor, momentum: ParamTensor = 0.8) {
        self.learningRate = learningRate
        self.momentum = momentum
    }

    public mutating func reset() {
        velocities = []
    }

    public mutating func update(_ parameters: inout [ParamTensor], along gradients: [ParamTensor]) {
        Self.validateGradients(gradients, against: parameters)
        Self.initializeStateIfNeeded(&velocities, for: parameters)

        for index in parameters.indices {
            velocities[index] = velocities[index] * momentum + learningRate * gradients[index].detached()
            parameters[index] -= velocities[index]
            parameters[index].discardContext()
        }
    }
}

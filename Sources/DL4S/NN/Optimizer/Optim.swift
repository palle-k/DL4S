//
//  Optim.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
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

/// An optimizer that moves the weights of a model along their gradients.
///
/// An optimizer does not know the model. It receives the trainable weights as an array from
/// ``LayerType/update(_:)`` and changes them in place. State such as moments or step counters is created on
/// the first step from the shapes of the weights and matched to the weights by position on every later step.
/// Because the state is `Codable`, a training run can be saved and resumed.
///
/// ```swift
/// var optimizer = Adam<Float, CPU>(learningRate: 0.001)
/// let loss = categoricalNegativeLogLikelihood(expected: labels, actual: model(images))
/// model.update { parameters in
///     optimizer.update(&parameters, along: loss.gradients(of: parameters))
/// }
/// ```
public protocol Optimizer: Codable {
    /// Element type of the weights
    associatedtype Element: NumericType

    /// Device type of the weights
    associatedtype Device: DeviceType

    /// Moves the weights along their gradients.
    ///
    /// The gradients must have the same count and order as the weights. The count and the shapes of the
    /// weights must match the state of the optimizer; call ``reset()`` after the set of trainable weights
    /// changed, for example after ``LayerType/freeze()`` or ``LayerType/unfreeze()``.
    ///
    /// - Parameters:
    ///   - parameters: Trainable weights of the model, changed in place.
    ///   - gradients: Gradient of the loss with respect to each weight.
    mutating func update(_ parameters: inout [Tensor<Element, Device>], along gradients: [Tensor<Element, Device>])

    /// Discards the state of the optimizer. The next step creates new state from the weights it receives.
    mutating func reset()
}

extension Optimizer {
    /// Creates the state tensors on the first step and verifies they still match in later steps.
    /// - Parameters:
    ///   - state: One state tensor per weight, or an empty array.
    ///   - parameters: Weights updated during the current step.
    static func initializeStateIfNeeded(_ state: inout [Tensor<Element, Device>], for parameters: [Tensor<Element, Device>]) {
        if state.isEmpty {
            state = parameters.map { Tensor(repeating: 0, shape: $0.shape) }
            return
        }
        precondition(state.count == parameters.count, "\(Self.self) has state for \(state.count) weights but received \(parameters.count). Call reset() after the set of trainable weights changed.")
        for index in state.indices {
            precondition(state[index].shape == parameters[index].shape, "\(Self.self) has state with shape \(state[index].shape) for weight \(index), but the weight has shape \(parameters[index].shape). Call reset() after the shapes of the trainable weights changed.")
        }
    }

    /// Checks that there is one gradient per weight.
    static func validateGradients(_ gradients: [Tensor<Element, Device>], against parameters: [Tensor<Element, Device>]) {
        precondition(gradients.count == parameters.count, "\(Self.self) received \(gradients.count) gradients for \(parameters.count) weights.")
    }
}

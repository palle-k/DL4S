//
//  LearningRate.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.05.20.
//  Copyright (c) 2020 - Palle Klewitz
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

/// A schedule that gives the learning rate for each step of a training run.
///
/// Set the learning rate of the optimizer to the value of the schedule before each step.
public protocol LearningRateScheduler: Sendable {
    /// Returns the learning rate at a step.
    /// - Parameter step: Number of the training step. The first step is 1.
    /// - Returns: The learning rate at the step.
    func learningRate<Element: NumericType>(atStep step: Int) -> Element
}

/// The learning rate schedule of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The learning rate is `modelDim^(-0.5) * min(step^(-0.5), step * warmupSteps^(-1.5))`. It increases linearly for
/// the first `warmupSteps` steps and then decreases proportionally to the inverse square root of the step.
public struct NoamScheduler: LearningRateScheduler {
    /// Number of steps in which the learning rate increases
    public let warmupSteps: Int

    /// Size of the hidden states of the model
    public let modelDim: Int

    /// Creates a learning rate schedule with warmup and inverse square root decay.
    /// - Parameters:
    ///   - warmupSteps: Number of steps in which the learning rate increases
    ///   - modelDim: Size of the hidden states of the model
    public init(warmupSteps: Int, modelDim: Int) {
        self.warmupSteps = warmupSteps
        self.modelDim = modelDim
    }

    /// Returns the learning rate at a step.
    /// - Parameter step: Number of the training step. The first step is 1. At step 0, the learning rate is 0.
    /// - Returns: The learning rate at the step.
    public func learningRate<Element: NumericType>(atStep step: Int) -> Element {
        let step = Float(step)
        let warmupSteps = Float(warmupSteps)
        let modelDim = Float(modelDim)

        return Element(1 / sqrt(modelDim) * min(1 / sqrt(step), step * pow(warmupSteps, -1.5)))
    }
}

//
//  Adam.swift
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

/// Adam optimizer (Adaptive moment estimation)
///
/// Follows [Kingma et al. - Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
public struct Adam<Element: NumericType, Device: DeviceType>: Optimizer, Sendable {
    public typealias ParamTensor = Tensor<Element, Device>

    /// Whether the maximum of the past second moments normalizes the step, as in AMSGrad
    public let useAMSGrad: Bool

    /// Learning rate scaling factor
    public var learningRate: ParamTensor

    /// Exponential decay rate for first moment
    public var beta1: ParamTensor

    /// Exponential decay rate for second moment
    public var beta2: ParamTensor

    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor

    private var beta1t: ParamTensor
    private var beta2t: ParamTensor

    private var firstMoments: [ParamTensor] = []
    private var secondMoments: [ParamTensor] = []
    private var secondMomentMax: [ParamTensor] = []

    /// Adam optimizer (Adaptive moment estimation)
    ///
    /// Follows [Kingma et al. - Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
    /// - Parameters:
    ///   - learningRate: Learning rate scaling factor
    ///   - useAMSGrad: Whether the maximum of the past second moments normalizes the step, as in AMSGrad
    ///   - beta1: Exponential decay rate for first moment
    ///   - beta2: Exponential decay rate for second moment
    ///   - epsilon: Normalization scalar added to divisors
    public init(learningRate: ParamTensor, useAMSGrad: Bool = false, beta1: ParamTensor = 0.9, beta2: ParamTensor = 0.999, epsilon: ParamTensor = 1e-8) {
        self.useAMSGrad = useAMSGrad

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2

        beta1t = beta1
        beta2t = beta2

        self.epsilon = epsilon
    }

    public mutating func reset() {
        beta1t = beta1
        beta2t = beta2

        firstMoments = []
        secondMoments = []
        secondMomentMax = []
    }

    /// Reports the moments of the weight at position `i` as `firstMoments.i`, `secondMoments.i`, and, with
    /// AMSGrad, `secondMomentMax.i`, and the decayed rates as `beta1t` and `beta2t`.
    public mutating func visitTensors(_ visitor: inout TensorVisitor<Element, Device>) {
        visitor.frozen(&firstMoments, named: "firstMoments")
        visitor.frozen(&secondMoments, named: "secondMoments")
        if useAMSGrad {
            visitor.frozen(&secondMomentMax, named: "secondMomentMax")
        }
        visitor.frozen(&beta1t, named: "beta1t")
        visitor.frozen(&beta2t, named: "beta2t")
    }

    /// Creates the moments of the layout. Without AMSGrad, the maximum second moments are not created.
    public mutating func adoptLayout(_ layout: TensorLayout) {
        firstMoments = Self.zeroState(for: layout.children(of: "firstMoments"))
        secondMoments = Self.zeroState(for: layout.children(of: "secondMoments"))
        secondMomentMax = useAMSGrad ? Self.zeroState(for: layout.children(of: "secondMomentMax")) : []
    }

    public mutating func update(_ parameters: inout [ParamTensor], along gradients: [ParamTensor]) {
        Self.validateGradients(gradients, against: parameters)
        Self.initializeStateIfNeeded(&firstMoments, for: parameters)
        Self.initializeStateIfNeeded(&secondMoments, for: parameters)
        if useAMSGrad {
            Self.initializeStateIfNeeded(&secondMomentMax, for: parameters)
        }

        for index in parameters.indices {
            let grad = gradients[index].detached()

            let addedToFirstMoment = grad * (1 - beta1)
            firstMoments[index] = firstMoments[index] * beta1 + addedToFirstMoment

            let addedToSecondMoment = (grad * grad) * (1 - beta2)
            secondMoments[index] = secondMoments[index] * beta2 + addedToSecondMoment

            let v_t_norm: ParamTensor
            if useAMSGrad {
                secondMomentMax[index] = Tensor.max(secondMomentMax[index], secondMoments[index])
                v_t_norm = secondMomentMax[index]
            } else {
                v_t_norm = secondMoments[index]
            }

            let m_car_t = firstMoments[index] / (1 - beta1t)
            let v_car_t = v_t_norm / (1 - beta2t)

            let delta = learningRate / (v_car_t.sqrt() + epsilon) * m_car_t
            parameters[index] -= delta
            parameters[index].discardContext()
        }

        beta1t *= beta1
        beta2t *= beta2
    }
}

//
//  ComposedActivation.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.09.26.
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

// MARK: Composed gradients

extension Composed {
    static func tanhBackward<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add((1 - output * output) * outputGradient)
    }

    static func reluBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(input.heaviside() * outputGradient)
    }

    static func sigmoidBackward<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(output * (1 - output) * outputGradient)
    }

    static func softmaxBackward<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(softmaxGradient(output: output, outputGradient: outputGradient, axis: axis))
    }

    /// The gradient of the softmax along an axis, `output * (outputGradient - sum(outputGradient * output, axis))`.
    static func softmaxGradient<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        output * (outputGradient - (outputGradient * output).reduceSum(along: [axis]).unsqueezed(at: axis))
    }

    static func logSoftmaxBackward<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(outputGradient - output.exp() * outputGradient.reduceSum(along: [axis]).unsqueezed(at: axis))
    }

    static func leakyReluBackward<N, Device>(
        input: Tensor<N, Device>,
        leakage: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inputGradient: inout GradientAccumulator<N, Device>,
        leakageGradient: inout GradientAccumulator<N, Device>,
    ) {
        if inputGradient.isRequested {
            // The slope at 0 is 0, as for the rectified linear unit.
            inputGradient.add(((input.heaviside() + leakage * (-input).heaviside()) * outputGradient).reducingBroadcast(to: input.shape))
        }
        if leakageGradient.isRequested {
            leakageGradient.add((-(-input).rectifiedLinear() * outputGradient).reducingBroadcast(to: leakage.shape))
        }
    }

    static func geluBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let s = (input * 1.702).sigmoid()
        inputGradient.add((s + 1.702 * input * s * (1 - s)) * outputGradient)
    }

    static func swishBackward<N, Device>(
        input: Tensor<N, Device>,
        beta: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inputGradient: inout GradientAccumulator<N, Device>,
        betaGradient: inout GradientAccumulator<N, Device>,
    ) {
        let s = (beta * input).sigmoid()
        let sigmoidSlope = s * (1 - s)
        if inputGradient.isRequested {
            inputGradient.add(((s + beta * input * sigmoidSlope) * outputGradient).reducingBroadcast(to: input.shape))
        }
        if betaGradient.isRequested {
            betaGradient.add((input * input * sigmoidSlope * outputGradient).reducingBroadcast(to: beta.shape))
        }
    }

    static func mishBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let t = (1 + input.exp()).log().tanh()
        // The derivative of log(1 + exp(x)) is sigmoid(x).
        inputGradient.add((t + input * (1 - t * t) * input.sigmoid()) * outputGradient)
    }

    static func lishtBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let t = input.tanh()
        inputGradient.add((t + input * (1 - t * t)) * outputGradient)
    }

    static func eluBackward<N, Device>(
        input: Tensor<N, Device>,
        alpha: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inputGradient: inout GradientAccumulator<N, Device>,
        alphaGradient: inout GradientAccumulator<N, Device>,
    ) {
        let exponential = (-(-input).rectifiedLinear()).exp()
        if inputGradient.isRequested {
            let positive = input.heaviside()
            inputGradient.add(((positive + (1 - positive) * alpha * exponential) * outputGradient).reducingBroadcast(to: input.shape))
        }
        if alphaGradient.isRequested {
            alphaGradient.add(((exponential - 1) * outputGradient).reducingBroadcast(to: alpha.shape))
        }
    }

    static func softplusBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(input.sigmoid() * outputGradient)
    }

    static func squareplusBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add((1 + input / (input * input + 4).sqrt()) / 2 * outputGradient)
    }
}

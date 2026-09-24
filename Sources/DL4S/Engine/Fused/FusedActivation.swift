//
//  FusedActivation.swift
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
    static func tanhBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.tanhGradient(output: output.detached(), outputGradient: outputGradient.detached())
    }

    static func reluBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.reluGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }

    static func sigmoid<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        0.5 * (input.detached() * 0.5).tanh() + 0.5
    }

    static func sigmoidBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.sigmoidGradient(output: output.detached(), outputGradient: outputGradient.detached())
    }

    static func softmax<N: NumericType>(input: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        let input = input.detached()
        let normalizer = input.reduceMax(along: [axis]).unsqueezed(at: axis)
        let exponentiated = (input - normalizer).exp()
        return exponentiated / exponentiated.reduceSum(along: [axis]).unsqueezed(at: axis)
    }

    static func softmaxBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        Composed.softmaxGradient(output: output.detached(), outputGradient: outputGradient.detached(), axis: axis)
    }

    static func logSoftmax<N: NumericType>(input: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        let input = input.detached()
        let normalized = input - input.reduceMax(along: [axis]).unsqueezed(at: axis)
        let logSumExp = normalized.exp().reduceSum(along: [axis]).log().unsqueezed(at: axis)
        return normalized - logSumExp
    }

    static func logSoftmaxBackward<N: NumericType>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        Composed.logSoftmaxGradient(output: output.detached(), outputGradient: outputGradient.detached(), axis: axis)
    }

    static func leakyRelu<N: NumericType>(input: Tensor<N, Device>, leakage: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return input.rectifiedLinear() - leakage.detached() * (-input).rectifiedLinear()
    }

    static func leakyReluBackward<N: NumericType>(input: Tensor<N, Device>, leakage: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> (input: Tensor<N, Device>?, leakage: Tensor<N, Device>?) {
        Composed.leakyReluGradients(
            input: input.detached(),
            leakage: leakage.detached(),
            outputGradient: outputGradient.detached(),
            computesInput: input.requiresGradient,
            computesLeakage: leakage.requiresGradient,
        )
    }

    static func gelu<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return input * (input * 1.702).sigmoid()
    }

    static func geluBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.geluGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }

    static func swish<N: NumericType>(input: Tensor<N, Device>, beta: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return input * (beta.detached() * input).sigmoid()
    }

    static func swishBackward<N: NumericType>(input: Tensor<N, Device>, beta: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> (input: Tensor<N, Device>?, beta: Tensor<N, Device>?) {
        Composed.swishGradients(
            input: input.detached(),
            beta: beta.detached(),
            outputGradient: outputGradient.detached(),
            computesInput: input.requiresGradient,
            computesBeta: beta.requiresGradient,
        )
    }

    static func mish<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return input * (1 + input.exp()).log().tanh()
    }

    static func mishBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.mishGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }

    static func lisht<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return input * input.tanh()
    }

    static func lishtBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.lishtGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }

    static func elu<N: NumericType>(input: Tensor<N, Device>, alpha: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        // exp(min(input, 0)) does not overflow for large inputs, and its exponential part is 0 for positive inputs.
        return input.rectifiedLinear() + alpha.detached() * ((-(-input).rectifiedLinear()).exp() - 1)
    }

    static func eluBackward<N: NumericType>(input: Tensor<N, Device>, alpha: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> (input: Tensor<N, Device>?, alpha: Tensor<N, Device>?) {
        Composed.eluGradients(
            input: input.detached(),
            alpha: alpha.detached(),
            outputGradient: outputGradient.detached(),
            computesInput: input.requiresGradient,
            computesAlpha: alpha.requiresGradient,
        )
    }

    static func softplus<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        (input.detached().exp() + 1).log()
    }

    static func softplusBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.softplusGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }

    static func squareplus<N: NumericType>(input: Tensor<N, Device>) -> Tensor<N, Device> {
        let input = input.detached()
        return (input + (input * input + 4).sqrt()) / 2
    }

    static func squareplusBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        Composed.squareplusGradient(input: input.detached(), outputGradient: outputGradient.detached())
    }
}

// MARK: Composed gradients

extension Composed {
    static func tanhGradient<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        (1 - output * output) * outputGradient
    }

    static func reluGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        input.heaviside() * outputGradient
    }

    static func sigmoidGradient<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        output * (1 - output) * outputGradient
    }

    static func softmaxGradient<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        output * (outputGradient - (outputGradient * output).reduceSum(along: [axis]).unsqueezed(at: axis))
    }

    static func logSoftmaxGradient<N, Device>(output: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axis: Int) -> Tensor<N, Device> {
        outputGradient - output.exp() * outputGradient.reduceSum(along: [axis]).unsqueezed(at: axis)
    }

    static func leakyReluGradients<N, Device>(
        input: Tensor<N, Device>,
        leakage: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesInput: Bool,
        computesLeakage: Bool,
    ) -> (input: Tensor<N, Device>?, leakage: Tensor<N, Device>?) {
        // The slope at 0 is 0, as for the rectified linear unit.
        let inputGradient = computesInput ? ((input.heaviside() + leakage * (-input).heaviside()) * outputGradient).reducingBroadcast(to: input.shape) : nil
        let leakageGradient = computesLeakage ? (-(-input).rectifiedLinear() * outputGradient).reducingBroadcast(to: leakage.shape) : nil
        return (inputGradient, leakageGradient)
    }

    static func geluGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        let s = (input * 1.702).sigmoid()
        return (s + 1.702 * input * s * (1 - s)) * outputGradient
    }

    static func swishGradients<N, Device>(
        input: Tensor<N, Device>,
        beta: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesInput: Bool,
        computesBeta: Bool,
    ) -> (input: Tensor<N, Device>?, beta: Tensor<N, Device>?) {
        let s = (beta * input).sigmoid()
        let sigmoidSlope = s * (1 - s)
        let inputGradient = computesInput ? ((s + beta * input * sigmoidSlope) * outputGradient).reducingBroadcast(to: input.shape) : nil
        let betaGradient = computesBeta ? (input * input * sigmoidSlope * outputGradient).reducingBroadcast(to: beta.shape) : nil
        return (inputGradient, betaGradient)
    }

    static func mishGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        let t = (1 + input.exp()).log().tanh()
        // The derivative of log(1 + exp(x)) is sigmoid(x).
        return (t + input * (1 - t * t) * input.sigmoid()) * outputGradient
    }

    static func lishtGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        let t = input.tanh()
        return (t + input * (1 - t * t)) * outputGradient
    }

    static func eluGradients<N, Device>(
        input: Tensor<N, Device>,
        alpha: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesInput: Bool,
        computesAlpha: Bool,
    ) -> (input: Tensor<N, Device>?, alpha: Tensor<N, Device>?) {
        let positive = input.heaviside()
        let exponential = (-(-input).rectifiedLinear()).exp()
        let inputGradient = computesInput ? ((positive + (1 - positive) * alpha * exponential) * outputGradient).reducingBroadcast(to: input.shape) : nil
        let alphaGradient = computesAlpha ? ((exponential - 1) * outputGradient).reducingBroadcast(to: alpha.shape) : nil
        return (inputGradient, alphaGradient)
    }

    static func softplusGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        input.sigmoid() * outputGradient
    }

    static func squareplusGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        (1 + input / (input * input + 4).sqrt()) / 2 * outputGradient
    }
}

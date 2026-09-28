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

// The defaults compute in the buffers of their results where they can. A gradient that is added to an accumulated gradient
// goes through one intermediate buffer, see `BufferMath.write(_:_:)`.

public extension FusedOperationsType {
    static func tanhBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (1 - output * output) * outputGradient
        math.write(inputGradient) { dx in
            math.multiply(output, output, into: dx)
            math.subtract(1, dx, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func reluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.write(inputGradient) { dx in
            math.heaviside(input, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func sigmoid<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.sigmoid(input, into: result)
    }

    static func sigmoidBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // output * (1 - output) * outputGradient
        math.write(inputGradient) { dx in
            math.subtract(1, output, into: dx)
            math.multiply(dx, output, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func softmax<N: NumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.softmax(input, along: axis, into: result)
    }

    static func softmaxBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // output * (outputGradient - sum(outputGradient * output))
        math.write(inputGradient) { dx in
            let sums = math.temporary(ShapeUtil.keptShape(of: output.shape, along: [axis]))
            math.multiply(outputGradient, output, into: dx)
            math.sum(dx, along: [axis], into: sums)
            math.subtract(outputGradient, sums, into: dx)
            math.multiply(dx, output, into: dx)
        }
    }

    static func logSoftmax<N: NumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let reduced = math.temporary(ShapeUtil.keptShape(of: input.shape, along: [axis]))
        let exponentials = math.temporary(input.shape)
        math.maximum(input, along: axis, into: reduced)
        math.subtract(input, reduced, into: result)
        math.exp(result, into: exponentials)
        math.sum(exponentials, along: [axis], into: reduced)
        math.log(reduced, into: reduced)
        math.subtract(result, reduced, into: result)
    }

    static func logSoftmaxBackward<N: NumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // outputGradient - exp(output) * sum(outputGradient)
        math.write(inputGradient) { dx in
            let sums = math.temporary(ShapeUtil.keptShape(of: output.shape, along: [axis]))
            math.sum(outputGradient, along: [axis], into: sums)
            math.exp(output, into: dx)
            math.multiply(dx, sums, into: dx)
            math.subtract(outputGradient, dx, into: dx)
        }
    }

    static func leakyRelu<N: NumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // relu(input) - leakage * relu(-input)
        let negativePart = math.temporary(input.shape)
        math.negate(input, into: negativePart)
        math.relu(negativePart, into: negativePart)
        math.multiply(negativePart, leakage, into: negativePart)
        math.relu(input, into: result)
        math.subtract(result, negativePart, into: result)
    }

    static func leakyReluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, leakageGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The slope at 0 is 0, as for the rectified linear unit: (heaviside(input) + leakage * heaviside(-input)) * outputGradient
        math.write(inputGradient) { dx in
            let positive = math.temporary(input.shape)
            math.negate(input, into: dx)
            math.heaviside(dx, into: dx)
            math.multiply(dx, leakage, into: dx)
            math.heaviside(input, into: positive)
            math.add(dx, positive, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
        if leakageGradient != nil {
            // -relu(-input) * outputGradient
            let products = math.temporary(input.shape)
            math.negate(input, into: products)
            math.relu(products, into: products)
            math.negate(products, into: products)
            math.multiply(products, outputGradient, into: products)
            math.writeSum(of: products, into: leakageGradient)
        }
    }

    static func gelu<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // input * sigmoid(1.702 * input)
        math.sigmoid(input, scale: N(1.702), into: result)
        math.multiply(result, input, into: result)
    }

    static func geluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (s + 1.702 * input * s * (1 - s)) * outputGradient with s = sigmoid(1.702 * input)
        math.write(inputGradient) { dx in
            let s = math.temporary(input.shape)
            math.sigmoid(input, scale: N(1.702), into: s)
            math.subtract(1, s, into: dx)
            math.multiply(dx, s, into: dx)
            math.multiply(dx, input, into: dx)
            math.multiply(dx, N(1.702), into: dx)
            math.add(dx, s, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func swish<N: NumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // input * sigmoid(beta * input)
        math.multiply(input, beta, into: result)
        math.sigmoid(result, into: result)
        math.multiply(result, input, into: result)
    }

    static func swishBackward<N: NumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, betaGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // With s = sigmoid(beta * input) and the slope s * (1 - s):
        // the input gradient is (s + beta * input * slope) * outputGradient, the beta gradient input * input * slope * outputGradient.
        let s = math.temporary(input.shape)
        let slope = math.temporary(input.shape)
        math.multiply(input, beta, into: s)
        math.sigmoid(s, into: s)
        math.subtract(1, s, into: slope)
        math.multiply(slope, s, into: slope)
        math.write(inputGradient) { dx in
            math.multiply(input, slope, into: dx)
            math.multiply(dx, beta, into: dx)
            math.add(dx, s, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
        if betaGradient != nil {
            math.multiply(slope, input, into: slope)
            math.multiply(slope, input, into: slope)
            math.multiply(slope, outputGradient, into: slope)
            math.writeSum(of: slope, into: betaGradient)
        }
    }

    static func mish<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // input * tanh(log(1 + exp(input)))
        math.exp(input, into: result)
        math.add(result, 1, into: result)
        math.log(result, into: result)
        math.tanh(result, into: result)
        math.multiply(result, input, into: result)
    }

    static func mishBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (t + input * (1 - t * t) * sigmoid(input)) * outputGradient with t = tanh(log(1 + exp(input))).
        // The derivative of log(1 + exp(x)) is sigmoid(x).
        math.write(inputGradient) { dx in
            let t = math.temporary(input.shape)
            let s = math.temporary(input.shape)
            math.exp(input, into: t)
            math.add(t, 1, into: t)
            math.log(t, into: t)
            math.tanh(t, into: t)
            math.sigmoid(input, into: s)
            math.multiply(t, t, into: dx)
            math.subtract(1, dx, into: dx)
            math.multiply(dx, input, into: dx)
            math.multiply(dx, s, into: dx)
            math.add(dx, t, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func lisht<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.tanh(input, into: result)
        math.multiply(result, input, into: result)
    }

    static func lishtBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (t + input * (1 - t * t)) * outputGradient with t = tanh(input)
        math.write(inputGradient) { dx in
            let t = math.temporary(input.shape)
            math.tanh(input, into: t)
            math.multiply(t, t, into: dx)
            math.subtract(1, dx, into: dx)
            math.multiply(dx, input, into: dx)
            math.add(dx, t, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func elu<N: NumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // relu(input) + alpha * (exp(min(input, 0)) - 1). exp(min(input, 0)) does not overflow for large inputs,
        // and its exponential part is 0 for positive inputs.
        let exponentialPart = math.temporary(input.shape)
        exponentialOfNegativePart(input, into: exponentialPart, math: math)
        math.add(exponentialPart, -1, into: exponentialPart)
        math.multiply(exponentialPart, alpha, into: exponentialPart)
        math.relu(input, into: result)
        math.add(result, exponentialPart, into: result)
    }

    static func eluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, alphaGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // With e = exp(min(input, 0)) and p = heaviside(input): the input gradient is (p + (1 - p) * alpha * e) * outputGradient,
        // the alpha gradient (e - 1) * outputGradient.
        let exponentials = math.temporary(input.shape)
        exponentialOfNegativePart(input, into: exponentials, math: math)
        math.write(inputGradient) { dx in
            let positive = math.temporary(input.shape)
            math.heaviside(input, into: positive)
            math.subtract(1, positive, into: dx)
            math.multiply(dx, alpha, into: dx)
            math.multiply(dx, exponentials, into: dx)
            math.add(dx, positive, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
        if alphaGradient != nil {
            math.add(exponentials, -1, into: exponentials)
            math.multiply(exponentials, outputGradient, into: exponentials)
            math.writeSum(of: exponentials, into: alphaGradient)
        }
    }

    static func softplus<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.exp(input, into: result)
        math.add(result, 1, into: result)
        math.log(result, into: result)
    }

    static func softplusBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.write(inputGradient) { dx in
            math.sigmoid(input, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func squareplus<N: NumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (input + sqrt(input * input + 4)) / 2
        let roots = math.temporary(input.shape)
        math.multiply(input, input, into: roots)
        math.add(roots, 4, into: roots)
        math.sqrt(roots, into: roots)
        math.add(input, roots, into: result)
        math.multiply(result, N(0.5), into: result)
    }

    static func squareplusBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (1 + input / sqrt(input * input + 4)) / 2 * outputGradient
        math.write(inputGradient) { dx in
            let roots = math.temporary(input.shape)
            math.multiply(input, input, into: roots)
            math.add(roots, 4, into: roots)
            math.sqrt(roots, into: roots)
            math.divide(input, roots, into: dx)
            math.add(dx, 1, into: dx)
            math.multiply(dx, N(0.5), into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }
}

extension FusedOperationsType {
    /// Computes `exp(min(input, 0))`, which does not overflow for large inputs.
    static func exponentialOfNegativePart<N: NumericType>(_ input: ShapedBuffer<N, Device>, into result: MutableShapedBuffer<N, Device>, math: BufferMath<N, Device>) {
        math.negate(input, into: result)
        math.relu(result, into: result)
        math.negate(result, into: result)
        math.exp(result, into: result)
    }
}

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
    static func tanhBackward<N: FloatingPointNumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // (1 - output * output) * outputGradient
        math.write(inputGradient) { dx in
            math.multiply(output, output, into: dx)
            math.subtract(1, dx, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func reluBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        math.write(inputGradient) { dx in
            math.heaviside(input, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func sigmoid<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        math.sigmoid(input, into: result)
    }

    static func sigmoidBackward<N: FloatingPointNumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // output * (1 - output) * outputGradient
        math.write(inputGradient) { dx in
            math.subtract(1, output, into: dx)
            math.multiply(dx, output, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func softmax<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        math.softmax(input, along: axis, into: result)
    }

    static func softmaxBackward<N: FloatingPointNumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // output * (outputGradient - sum(outputGradient * output))
        math.write(inputGradient) { dx in
            let sums = Device.Memory.allocateBuffer(withShape: ShapeUtil.keptShape(of: output.shape, along: [axis]), type: N.self)
            math.multiply(outputGradient, output, into: dx)
            math.sum(dx, along: [axis], into: sums)
            math.subtract(outputGradient, sums, into: dx)
            math.multiply(dx, output, into: dx)
        }
    }

    static func logSoftmax<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, axis: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        let reduced = Device.Memory.allocateBuffer(withShape: ShapeUtil.keptShape(of: input.shape, along: [axis]), type: N.self)
        let exponentials = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        math.maximum(input, along: axis, into: reduced)
        math.subtract(input, reduced, into: result)
        math.exp(result, into: exponentials)
        math.sum(exponentials, along: [axis], into: reduced)
        math.log(reduced, into: reduced)
        math.subtract(result, reduced, into: result)
    }

    static func logSoftmaxBackward<N: FloatingPointNumericType>(output: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axis: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // outputGradient - exp(output) * sum(outputGradient)
        math.write(inputGradient) { dx in
            let sums = Device.Memory.allocateBuffer(withShape: ShapeUtil.keptShape(of: output.shape, along: [axis]), type: N.self)
            math.sum(outputGradient, along: [axis], into: sums)
            math.exp(output, into: dx)
            math.multiply(dx, sums, into: dx)
            math.subtract(outputGradient, dx, into: dx)
        }
    }

    static func leakyRelu<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // relu(input) - leakage * relu(-input)
        let negativePart = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        math.negate(input, into: negativePart)
        math.relu(negativePart, into: negativePart)
        math.multiply(negativePart, leakage, into: negativePart)
        math.relu(input, into: result)
        math.subtract(result, negativePart, into: result)
    }

    static func leakyReluBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, leakage: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, leakageGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // The slope at 0 is 0, as for the rectified linear unit: (heaviside(input) + leakage * heaviside(-input)) * outputGradient
        math.write(inputGradient) { dx in
            let positive = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
            math.negate(input, into: dx)
            math.heaviside(dx, into: dx)
            math.multiply(dx, leakage, into: dx)
            math.heaviside(input, into: positive)
            math.add(dx, positive, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
        if leakageGradient != nil {
            // -relu(-input) * outputGradient
            let products = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
            math.negate(input, into: products)
            math.relu(products, into: products)
            math.negate(products, into: products)
            math.multiply(products, outputGradient, into: products)
            math.writeSum(of: products, into: leakageGradient)
        }
    }

    static func gelu<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // input * sigmoid(1.702 * input)
        math.sigmoid(input, scale: N(FusedConstants.geluSlope), into: result)
        math.multiply(result, input, into: result)
    }

    static func geluBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // (s + 1.702 * input * s * (1 - s)) * outputGradient with s = sigmoid(1.702 * input)
        math.write(inputGradient) { dx in
            let s = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
            math.sigmoid(input, scale: N(FusedConstants.geluSlope), into: s)
            math.subtract(1, s, into: dx)
            math.multiply(dx, s, into: dx)
            math.multiply(dx, input, into: dx)
            math.multiply(dx, N(FusedConstants.geluSlope), into: dx)
            math.add(dx, s, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func swish<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // input * sigmoid(beta * input)
        math.multiply(input, beta, into: result)
        math.sigmoid(result, into: result)
        math.multiply(result, input, into: result)
    }

    static func swishBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, beta: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, betaGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // With s = sigmoid(beta * input) and the slope s * (1 - s):
        // the input gradient is (s + beta * input * slope) * outputGradient, the beta gradient input * input * slope * outputGradient.
        let s = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        let slope = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
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

    static func mish<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // input * tanh(log(1 + exp(input)))
        math.exp(input, into: result)
        math.add(result, 1, into: result)
        math.log(result, into: result)
        math.tanh(result, into: result)
        math.multiply(result, input, into: result)
    }

    static func mishBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // (t + input * (1 - t * t) * sigmoid(input)) * outputGradient with t = tanh(log(1 + exp(input))).
        // The derivative of log(1 + exp(x)) is sigmoid(x).
        math.write(inputGradient) { dx in
            let t = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
            let s = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
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

    static func lisht<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        math.tanh(input, into: result)
        math.multiply(result, input, into: result)
    }

    static func lishtBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // (t + input * (1 - t * t)) * outputGradient with t = tanh(input)
        math.write(inputGradient) { dx in
            let t = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
            math.tanh(input, into: t)
            math.multiply(t, t, into: dx)
            math.subtract(1, dx, into: dx)
            math.multiply(dx, input, into: dx)
            math.add(dx, t, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func elu<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // relu(input) + alpha * (exp(min(input, 0)) - 1). exp(min(input, 0)) does not overflow for large inputs,
        // and its exponential part is 0 for positive inputs.
        let exponentialPart = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        exponentialOfNegativePart(input, into: exponentialPart, math: math)
        math.add(exponentialPart, -1, into: exponentialPart)
        math.multiply(exponentialPart, alpha, into: exponentialPart)
        math.relu(input, into: result)
        math.add(result, exponentialPart, into: result)
    }

    static func eluBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, alpha: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?, alphaGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // With e = exp(min(input, 0)) and p = heaviside(input): the input gradient is (p + (1 - p) * alpha * e) * outputGradient,
        // the alpha gradient (e - 1) * outputGradient.
        let exponentials = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        exponentialOfNegativePart(input, into: exponentials, math: math)
        math.write(inputGradient) { dx in
            let positive = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
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

    static func softplus<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // max(x, 0) + log(1 + exp(-|x|)), because exp(x) overflows for large x. -|x| = x - 2 max(x, 0).
        let positive = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        math.relu(input, into: positive)
        math.multiply(positive, -2, into: result)
        math.add(result, input, into: result)
        math.exp(result, into: result)
        math.add(result, 1, into: result)
        math.log(result, into: result)
        math.add(result, positive, into: result)
    }

    static func softplusBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        math.write(inputGradient) { dx in
            math.sigmoid(input, into: dx)
            math.multiply(dx, outputGradient, into: dx)
        }
    }

    static func squareplus<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // (input + sqrt(input * input + 4)) / 2
        let roots = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
        math.multiply(input, input, into: roots)
        math.add(roots, 4, into: roots)
        math.sqrt(roots, into: roots)
        math.add(input, roots, into: result)
        math.multiply(result, N(0.5), into: result)
    }

    static func squareplusBackward<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        // (1 + input / sqrt(input * input + 4)) / 2 * outputGradient
        math.write(inputGradient) { dx in
            let roots = Device.Memory.allocateBuffer(withShape: input.shape, type: N.self)
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
    static func exponentialOfNegativePart<N: FloatingPointNumericType>(_ input: ShapedBuffer<N, Device>, into result: MutableShapedBuffer<N, Device>, math: BufferMath<N, Device>) {
        math.negate(input, into: result)
        math.relu(result, into: result)
        math.negate(result, into: result)
        math.exp(result, into: result)
    }
}

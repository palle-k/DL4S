//
//  GPUFusedActivation.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation

// Every activation is one element-wise kernel for the forward pass and one for the backward pass. Parameters with a
// requested gradient, and parameters with other shapes than a scalar or the last axes of the input, use the default implementations.

public extension GPUFusedOperations {
    static func tanhBackward<N: NumericType>(output: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        guard GPUFused.runsKernel(N.self, elements: output.count, reading: [output.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.tanhBackward(output: output, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("tanh", input: output, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func reluBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.reluBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("relu", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func sigmoid<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.sigmoid(input: input, result: result)
            return
        }
        GPUFused.activation("sigmoid", input: input, result: result)
    }

    static func sigmoidBackward<N: NumericType>(output: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        guard GPUFused.runsKernel(N.self, elements: output.count, reading: [output.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.sigmoidBackward(output: output, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("sigmoid", input: output, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func leakyRelu<N: NumericType>(input: ShapedBuffer<N, GPU>, leakage: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(ShapeUtil.broadcasts(leakage.shape, to: input.shape), "The leakage must be broadcastable to the shape of the input.")
        guard let length = GPUFused.parameterLength(leakage, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, leakage.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.leakyRelu(input: input, leakage: leakage, result: result)
            return
        }
        GPUFused.activation("leaky_relu", input: input, parameter: leakage, parameterLength: length, result: result)
    }

    static func leakyReluBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, leakage: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?, leakageGradient: GradientBuffer<N, GPU>?) {
        precondition(ShapeUtil.broadcasts(leakage.shape, to: input.shape), "The leakage must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel does not compute the gradient of the leakage.
        guard leakageGradient == nil, let length = GPUFused.parameterLength(leakage, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, leakage.gpuBuffer, outputGradient.gpuBuffer])
        else {
            DefaultFusedOperations<GPU>.leakyReluBackward(input: input, leakage: leakage, outputGradient: outputGradient, inputGradient: inputGradient, leakageGradient: leakageGradient)
            return
        }
        if let inputGradient {
            GPUFused.activationBackward("leaky_relu", input: input, outputGradient: outputGradient, parameter: leakage, parameterLength: length, inputGradient: inputGradient)
        }
    }

    static func gelu<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.gelu(input: input, result: result)
            return
        }
        GPUFused.activation("gelu", input: input, result: result)
    }

    static func geluBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.geluBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("gelu", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func swish<N: NumericType>(input: ShapedBuffer<N, GPU>, beta: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(ShapeUtil.broadcasts(beta.shape, to: input.shape), "The beta must be broadcastable to the shape of the input.")
        guard let length = GPUFused.parameterLength(beta, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, beta.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.swish(input: input, beta: beta, result: result)
            return
        }
        GPUFused.activation("swish", input: input, parameter: beta, parameterLength: length, result: result)
    }

    static func swishBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, beta: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?, betaGradient: GradientBuffer<N, GPU>?) {
        precondition(ShapeUtil.broadcasts(beta.shape, to: input.shape), "The beta must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel does not compute the gradient of the beta.
        guard betaGradient == nil, let length = GPUFused.parameterLength(beta, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, beta.gpuBuffer, outputGradient.gpuBuffer])
        else {
            DefaultFusedOperations<GPU>.swishBackward(input: input, beta: beta, outputGradient: outputGradient, inputGradient: inputGradient, betaGradient: betaGradient)
            return
        }
        if let inputGradient {
            GPUFused.activationBackward("swish", input: input, outputGradient: outputGradient, parameter: beta, parameterLength: length, inputGradient: inputGradient)
        }
    }

    static func mish<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.mish(input: input, result: result)
            return
        }
        GPUFused.activation("mish", input: input, result: result)
    }

    static func mishBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.mishBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("mish", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func lisht<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.lisht(input: input, result: result)
            return
        }
        GPUFused.activation("lisht", input: input, result: result)
    }

    static func lishtBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.lishtBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("lisht", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func elu<N: NumericType>(input: ShapedBuffer<N, GPU>, alpha: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        precondition(ShapeUtil.broadcasts(alpha.shape, to: input.shape), "The alpha must be broadcastable to the shape of the input.")
        guard let length = GPUFused.parameterLength(alpha, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, alpha.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.elu(input: input, alpha: alpha, result: result)
            return
        }
        GPUFused.activation("elu", input: input, parameter: alpha, parameterLength: length, result: result)
    }

    static func eluBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, alpha: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?, alphaGradient: GradientBuffer<N, GPU>?) {
        precondition(ShapeUtil.broadcasts(alpha.shape, to: input.shape), "The alpha must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel does not compute the gradient of the alpha.
        guard alphaGradient == nil, let length = GPUFused.parameterLength(alpha, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, alpha.gpuBuffer, outputGradient.gpuBuffer])
        else {
            DefaultFusedOperations<GPU>.eluBackward(input: input, alpha: alpha, outputGradient: outputGradient, inputGradient: inputGradient, alphaGradient: alphaGradient)
            return
        }
        if let inputGradient {
            GPUFused.activationBackward("elu", input: input, outputGradient: outputGradient, parameter: alpha, parameterLength: length, inputGradient: inputGradient)
        }
    }

    static func softplus<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.softplus(input: input, result: result)
            return
        }
        GPUFused.activation("softplus", input: input, result: result)
    }

    static func softplusBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.softplusBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("softplus", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func squareplus<N: NumericType>(input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.squareplus(input: input, result: result)
            return
        }
        GPUFused.activation("squareplus", input: input, result: result)
    }

    static func squareplusBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.squareplusBackward(input: input, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("squareplus", input: input, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func dropout<N: NumericType>(input: ShapedBuffer<N, GPU>, rate: Float, result: MutableShapedBuffer<N, GPU>, mask: MutableShapedBuffer<N, GPU>) {
        let probability = 1 - rate
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.dropout(input: input, rate: rate, result: result, mask: mask)
            return
        }
        // The kernel would drop an element whose random number is UInt32.max, so a rate of 0 keeps every element without it.
        guard probability < 1 else {
            GPUKernels.fill(mask.gpuBuffer, word: Float(1).bitPattern, count: mask.count)
            GPUKernels.copyWords(from: input.gpuBuffer, to: result.gpuBuffer, byteCount: input.count * MemoryLayout<Float>.stride)
            return
        }
        // An element is kept when a uniform 32-bit random number is below the threshold, which happens with the given probability.
        let threshold = probability <= 0 ? 0 : UInt32(Swift.min(Double(probability) * 0x1p32, Double(UInt32.max)))
        var generator = WyHash()
        let seed = generator.next()
        let parameters = DropoutParameters(count: UInt32(input.count), threshold: threshold, seed: SIMD2(UInt32(truncatingIfNeeded: seed), UInt32(truncatingIfNeeded: seed >> 32)))
        let (x, y, m) = (input.gpuBuffer, result.gpuBuffer, mask.gpuBuffer)
        let pipeline = GPUKernels.pipeline("dropout_forward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x], writing: [y, m]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.buffer(m)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
    }

    static func dropoutBackward<N: NumericType>(mask: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == mask.shape, "The gradient of the values must have the shape of the mask.")
        guard GPUFused.runsKernel(N.self, elements: mask.count, reading: [mask.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.dropoutBackward(mask: mask, outputGradient: outputGradient, inputGradient: inputGradient)
            return
        }
        GPUFused.activationBackward("dropout", input: mask, outputGradient: outputGradient, inputGradient: inputGradient)
    }
}

private struct DropoutParameters {
    var count: UInt32
    var threshold: UInt32
    var seed: SIMD2<UInt32>
}
#endif

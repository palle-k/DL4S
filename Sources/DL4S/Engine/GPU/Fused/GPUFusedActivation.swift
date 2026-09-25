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

// Every activation is one element-wise kernel for the forward pass and one for the backward pass.
// Parameters with a gradient, and parameters with other shapes than a scalar or the last axes of the input, use the default implementations.

public extension GPUFusedOperations {
    static func tanhBackward<N: NumericType>(output: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard output.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: output.count, reading: [output, outputGradient]) else {
            DefaultFusedOperations<GPU>.tanhBackward(output: output, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("tanh", input: output, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func reluBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.reluBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("relu", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func sigmoid<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.sigmoid(input: input)
        }
        return GPUFused.activation("sigmoid", input: input)
    }

    static func sigmoidBackward<N: NumericType>(output: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard output.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: output.count, reading: [output, outputGradient]) else {
            DefaultFusedOperations<GPU>.sigmoidBackward(output: output, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("sigmoid", input: output, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func leakyRelu<N: NumericType>(input: Tensor<N, GPU>, leakage: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard let length = GPUFused.parameterLength(leakage, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input, leakage]) else {
            return DefaultFusedOperations<GPU>.leakyRelu(input: input, leakage: leakage)
        }
        return GPUFused.activation("leaky_relu", input: input, parameter: leakage, parameterLength: length)
    }

    static func leakyReluBackward<N: NumericType>(input: Tensor<N, GPU>, leakage: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradients: inout (input: Tensor<N, GPU>?, leakage: Tensor<N, GPU>?)) {
        guard !leakage.requiresGradient, input.shape == outputGradient.shape, let length = GPUFused.parameterLength(leakage, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, leakage, outputGradient])
        else {
            DefaultFusedOperations<GPU>.leakyReluBackward(input: input, leakage: leakage, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        if input.requiresGradient {
            GPUFused.activationBackward("leaky_relu", input: input, outputGradient: outputGradient, parameter: leakage, parameterLength: length, accumulating: &gradients.input)
        }
    }

    static func gelu<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.gelu(input: input)
        }
        return GPUFused.activation("gelu", input: input)
    }

    static func geluBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.geluBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("gelu", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func swish<N: NumericType>(input: Tensor<N, GPU>, beta: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard let length = GPUFused.parameterLength(beta, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input, beta]) else {
            return DefaultFusedOperations<GPU>.swish(input: input, beta: beta)
        }
        return GPUFused.activation("swish", input: input, parameter: beta, parameterLength: length)
    }

    static func swishBackward<N: NumericType>(input: Tensor<N, GPU>, beta: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradients: inout (input: Tensor<N, GPU>?, beta: Tensor<N, GPU>?)) {
        guard !beta.requiresGradient, input.shape == outputGradient.shape, let length = GPUFused.parameterLength(beta, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, beta, outputGradient])
        else {
            DefaultFusedOperations<GPU>.swishBackward(input: input, beta: beta, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        if input.requiresGradient {
            GPUFused.activationBackward("swish", input: input, outputGradient: outputGradient, parameter: beta, parameterLength: length, accumulating: &gradients.input)
        }
    }

    static func mish<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.mish(input: input)
        }
        return GPUFused.activation("mish", input: input)
    }

    static func mishBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.mishBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("mish", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func lisht<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.lisht(input: input)
        }
        return GPUFused.activation("lisht", input: input)
    }

    static func lishtBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.lishtBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("lisht", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func elu<N: NumericType>(input: Tensor<N, GPU>, alpha: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard let length = GPUFused.parameterLength(alpha, input: input), GPUFused.runsKernel(N.self, elements: input.count, reading: [input, alpha]) else {
            return DefaultFusedOperations<GPU>.elu(input: input, alpha: alpha)
        }
        return GPUFused.activation("elu", input: input, parameter: alpha, parameterLength: length)
    }

    static func eluBackward<N: NumericType>(input: Tensor<N, GPU>, alpha: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradients: inout (input: Tensor<N, GPU>?, alpha: Tensor<N, GPU>?)) {
        guard !alpha.requiresGradient, input.shape == outputGradient.shape, let length = GPUFused.parameterLength(alpha, input: input),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, alpha, outputGradient])
        else {
            DefaultFusedOperations<GPU>.eluBackward(input: input, alpha: alpha, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        if input.requiresGradient {
            GPUFused.activationBackward("elu", input: input, outputGradient: outputGradient, parameter: alpha, parameterLength: length, accumulating: &gradients.input)
        }
    }

    static func softplus<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.softplus(input: input)
        }
        return GPUFused.activation("softplus", input: input)
    }

    static func softplusBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.softplusBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("softplus", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func squareplus<N: NumericType>(input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.squareplus(input: input)
        }
        return GPUFused.activation("squareplus", input: input)
    }

    static func squareplusBackward<N: NumericType>(input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard input.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, outputGradient]) else {
            DefaultFusedOperations<GPU>.squareplusBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("squareplus", input: input, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func dropout<N: NumericType>(input: Tensor<N, GPU>, rate: Float) -> (output: Tensor<N, GPU>, mask: Tensor<N, GPU>) {
        let probability = 1 - rate
        guard probability < 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.dropout(input: input, rate: rate)
        }
        let output: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        let mask: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        // An element is kept when a uniform 32-bit random number is below the threshold, which happens with the given probability.
        let threshold = probability <= 0 ? 0 : UInt32(Swift.min(Double(probability) * 0x1p32, Double(UInt32.max)))
        var generator = WyHash()
        let seed = generator.next()
        let parameters = DropoutParameters(count: UInt32(input.count), threshold: threshold, seed: SIMD2(UInt32(truncatingIfNeeded: seed), UInt32(truncatingIfNeeded: seed >> 32)))
        let (x, y, m) = (input.gpuBuffer, output.gpuBuffer, mask.gpuBuffer)
        let pipeline = GPUKernels.pipeline("dropout_forward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x], writing: [y, m]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.buffer(m)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
        return (output, mask)
    }

    static func dropoutBackward<N: NumericType>(mask: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        guard mask.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: mask.count, reading: [mask, outputGradient]) else {
            DefaultFusedOperations<GPU>.dropoutBackward(mask: mask, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        GPUFused.activationBackward("dropout", input: mask, outputGradient: outputGradient, accumulating: &gradient)
    }
}

private struct DropoutParameters {
    var count: UInt32
    var threshold: UInt32
    var seed: SIMD2<UInt32>
}
#endif

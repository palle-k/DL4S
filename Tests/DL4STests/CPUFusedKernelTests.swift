//
//  CPUFusedKernelTests.swift
//  DL4STests
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

@testable import DL4S
import Testing

private typealias DoubleTensor = Tensor<Double, CPU>
private typealias Gradient = GradientAccumulator<Double, CPU>

/// Fused operations of the CPU, or the default implementations.
private typealias Operations = any FusedOperationsType<CPU>.Type

private func uniform(_ shape: [Int], min: Double = -2, max: Double = 2, seed: UInt64, requiresGradient: Bool = false) -> DoubleTensor {
    var generator = WyHash(seed: seed)
    return DoubleTensor(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient, using: &generator)
}

/// Initial values of the accumulated gradients that a kernel case passes to the backward requirements.
enum Accumulation: String, CaseIterable, Sendable {
    case empty
    case prefilled

    /// Returns nil, or a tensor with the shape `shape` that the backward requirement adds the gradient to.
    fileprivate func start(_ shape: [Int]) -> DoubleTensor? {
        self == .empty ? nil : uniform(shape, seed: 100)
    }

    /// Returns an accumulator for a gradient of the given shape that is requested.
    fileprivate func accumulator(_ shape: [Int]) -> Gradient {
        Gradient(isRequested: true, shape: shape, value: start(shape))
    }

    /// Returns an accumulator for the gradient of a source, which is requested when the source requires a gradient.
    fileprivate func accumulator(for source: DoubleTensor) -> Gradient {
        Gradient(isRequested: source.requiresGradient, shape: source.shape, value: start(source.shape))
    }
}

/// The tensors that a kernel case passes to the requirements as buffers.
///
/// A buffer does not keep its tensor alive, so the inputs keep every tensor until the case returns.
private final class Inputs {
    private var tensors: [DoubleTensor] = []
    private var labels: [Tensor<Int32, CPU>] = []

    func buffer(_ tensor: DoubleTensor) -> ShapedBuffer<Double, CPU> {
        tensors.append(tensor)
        return tensor.values
    }

    func buffer(_ tensor: Tensor<Int32, CPU>) -> ShapedBuffer<Int32, CPU> {
        labels.append(tensor)
        return tensor.values
    }

    func buffer(_ tensor: DoubleTensor?) -> ShapedBuffer<Double, CPU>? {
        tensor.map { buffer($0) }
    }
}

/// Returns a tensor of the given shape, whose elements a forward requirement writes.
private func output(_ shape: [Int], _ forward: (MutableShapedBuffer<Double, CPU>) -> Void) -> DoubleTensor {
    var result = DoubleTensor(uninitializedShape: shape)
    forward(result.mutableValues)
    return result
}

/// A call of fused requirements, with inputs that exercise the kernels of the CPU.
struct KernelCase: CustomTestStringConvertible, Sendable {
    let name: String
    private let body: @Sendable (Operations, Accumulation, Inputs) -> [DoubleTensor?]

    var testDescription: String {
        name
    }

    fileprivate init(_ name: String, _ body: @escaping @Sendable (Operations, Accumulation, Inputs) -> [DoubleTensor?]) {
        self.name = name
        self.body = body
    }

    /// Returns the results and the accumulated gradients of the case.
    fileprivate func run(_ ops: Operations, _ accumulation: Accumulation) -> [DoubleTensor?] {
        let inputs = Inputs()
        return withExtendedLifetime(inputs) {
            body(ops, accumulation, inputs)
        }
    }

    /// The results of a forward requirement and of its backward requirement for an input that requires a gradient.
    fileprivate static func unary(
        _ name: String,
        shape: [Int] = [7, 1000],
        forward: @escaping @Sendable (Operations, ShapedBuffer<Double, CPU>, MutableShapedBuffer<Double, CPU>) -> Void,
        backward: @escaping @Sendable (Operations, ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>, GradientBuffer<Double, CPU>?) -> Void,
    ) -> KernelCase {
        KernelCase(name) { ops, accumulation, inputs in
            let x = inputs.buffer(uniform(shape, seed: 1, requiresGradient: true))
            var gradient = accumulation.accumulator(shape)
            backward(ops, x, inputs.buffer(uniform(shape, seed: 2)), gradient.buffer())
            return [output(shape) { forward(ops, x, $0) }, gradient.value]
        }
    }

    /// The results of a softmax requirement and of its backward requirement, which reads the result.
    fileprivate static func softmax(_ name: String, shape: [Int] = [7, 1000], axis: Int, logarithmic: Bool) -> KernelCase {
        KernelCase(name) { ops, accumulation, inputs in
            let x = inputs.buffer(uniform(shape, seed: 1))
            let result = output(shape) { logarithmic ? ops.logSoftmax(input: x, axis: axis, result: $0) : ops.softmax(input: x, axis: axis, result: $0) }
            var gradient = accumulation.accumulator(shape)
            let (y, g) = (inputs.buffer(result), inputs.buffer(uniform(shape, seed: 2)))
            if logarithmic {
                ops.logSoftmaxBackward(output: y, outputGradient: g, axis: axis, inputGradient: gradient.buffer())
            } else {
                ops.softmaxBackward(output: y, outputGradient: g, axis: axis, inputGradient: gradient.buffer())
            }
            return [result, gradient.value]
        }
    }
}

extension KernelCase {
    static let activations: [KernelCase] = [
        KernelCase("tanhBackward") { ops, accumulation, inputs in
            var gradient = accumulation.accumulator([7, 1000])
            ops.tanhBackward(output: inputs.buffer(uniform([7, 1000], min: -1, max: 1, seed: 1)), outputGradient: inputs.buffer(uniform([7, 1000], seed: 2)), inputGradient: gradient.buffer())
            return [gradient.value]
        },
        .unary("rectified linear", forward: { _, x, y in CPU.Engine.relu(values: x, result: y) }, backward: { ops, x, g, gradient in ops.reluBackward(input: x, outputGradient: g, inputGradient: gradient) }),
        KernelCase("sigmoid") { ops, accumulation, inputs in
            let x = inputs.buffer(uniform([7, 1000], seed: 1))
            let result = output(x.shape) { ops.sigmoid(input: x, result: $0) }
            var gradient = accumulation.accumulator(x.shape)
            ops.sigmoidBackward(output: inputs.buffer(result), outputGradient: inputs.buffer(uniform([7, 1000], seed: 2)), inputGradient: gradient.buffer())
            return [result, gradient.value]
        },
        KernelCase("leakyRelu") { ops, accumulation, inputs in
            let (x, leakage) = (uniform([7, 1000], seed: 1, requiresGradient: true), DoubleTensor(0.2, requiresGradient: true))
            var (inputGradient, leakageGradient) = (accumulation.accumulator(for: x), accumulation.accumulator(for: leakage))
            let (xs, leakages) = (inputs.buffer(x), inputs.buffer(leakage))
            ops.leakyReluBackward(input: xs, leakage: leakages, outputGradient: inputs.buffer(uniform([7, 1000], seed: 2)), inputGradient: inputGradient.buffer(), leakageGradient: leakageGradient.buffer())
            return [output(x.shape) { ops.leakyRelu(input: xs, leakage: leakages, result: $0) }, inputGradient.value, leakageGradient.value]
        },
        .unary("gelu", forward: { ops, x, y in ops.gelu(input: x, result: y) }, backward: { ops, x, g, gradient in ops.geluBackward(input: x, outputGradient: g, inputGradient: gradient) }),
        KernelCase("swish with a scalar beta") { ops, accumulation, inputs in
            swish(ops, accumulation, inputs, x: uniform([7, 1000], seed: 1, requiresGradient: true), beta: DoubleTensor(1.3, requiresGradient: true))
        },
        KernelCase("swish with a beta per channel") { ops, accumulation, inputs in
            swish(ops, accumulation, inputs, x: uniform([7, 1000], seed: 1, requiresGradient: true), beta: uniform([1000], min: 0.5, max: 2, seed: 3, requiresGradient: true))
        },
        KernelCase("swish with a broadcast beta") { ops, accumulation, inputs in
            swish(ops, accumulation, inputs, x: uniform([7, 10, 3], seed: 1, requiresGradient: true), beta: uniform([10, 1], seed: 3, requiresGradient: true))
        },
        .unary("mish", forward: { ops, x, y in ops.mish(input: x, result: y) }, backward: { ops, x, g, gradient in ops.mishBackward(input: x, outputGradient: g, inputGradient: gradient) }),
        .unary("lisht", forward: { ops, x, y in ops.lisht(input: x, result: y) }, backward: { ops, x, g, gradient in ops.lishtBackward(input: x, outputGradient: g, inputGradient: gradient) }),
        KernelCase("elu") { ops, accumulation, inputs in
            let (x, alpha) = (uniform([7, 1000], seed: 1, requiresGradient: true), DoubleTensor(0.7, requiresGradient: true))
            var (inputGradient, alphaGradient) = (accumulation.accumulator(for: x), accumulation.accumulator(for: alpha))
            let (xs, alphas) = (inputs.buffer(x), inputs.buffer(alpha))
            ops.eluBackward(input: xs, alpha: alphas, outputGradient: inputs.buffer(uniform([7, 1000], seed: 2)), inputGradient: inputGradient.buffer(), alphaGradient: alphaGradient.buffer())
            return [output(x.shape) { ops.elu(input: xs, alpha: alphas, result: $0) }, inputGradient.value, alphaGradient.value]
        },
        .unary("softplus", forward: { ops, x, y in ops.softplus(input: x, result: y) }, backward: { ops, x, g, gradient in ops.softplusBackward(input: x, outputGradient: g, inputGradient: gradient) }),
        .unary("squareplus", forward: { ops, x, y in ops.squareplus(input: x, result: y) }, backward: { ops, x, g, gradient in ops.squareplusBackward(input: x, outputGradient: g, inputGradient: gradient) }),
    ]

    private static func swish(_ ops: Operations, _ accumulation: Accumulation, _ inputs: Inputs, x: DoubleTensor, beta: DoubleTensor) -> [DoubleTensor?] {
        var (inputGradient, betaGradient) = (accumulation.accumulator(for: x), accumulation.accumulator(for: beta))
        let (xs, betas) = (inputs.buffer(x), inputs.buffer(beta))
        ops.swishBackward(input: xs, beta: betas, outputGradient: inputs.buffer(uniform(x.shape, seed: 2)), inputGradient: inputGradient.buffer(), betaGradient: betaGradient.buffer())
        return [output(x.shape) { ops.swish(input: xs, beta: betas, result: $0) }, inputGradient.value, betaGradient.value]
    }

    static let softmax: [KernelCase] = [
        .softmax("softmax along long rows", axis: 1, logarithmic: false),
        .softmax("softmax along short rows", shape: [300, 10], axis: 1, logarithmic: false),
        .softmax("softmax along the first axis", shape: [4, 3, 5], axis: 0, logarithmic: false),
        .softmax("logSoftmax along long rows", axis: 1, logarithmic: true),
        .softmax("logSoftmax along short rows", shape: [300, 10], axis: 1, logarithmic: true),
    ]

    /// The sources of a normalization and the accumulators of their gradients.
    private struct Normalization {
        let x: ShapedBuffer<Double, CPU>
        let scale: ShapedBuffer<Double, CPU>
        let shift: ShapedBuffer<Double, CPU>
        let outputGradient: ShapedBuffer<Double, CPU>
        var inputGradient: Gradient
        var scaleGradient: Gradient
        var shiftGradient: Gradient

        init(_ inputs: Inputs, _ accumulation: Accumulation, x: DoubleTensor, scale: DoubleTensor, shift: DoubleTensor) {
            (self.x, self.scale, self.shift) = (inputs.buffer(x), inputs.buffer(scale), inputs.buffer(shift))
            outputGradient = inputs.buffer(uniform(x.shape, seed: 4))
            (inputGradient, scaleGradient, shiftGradient) = (accumulation.accumulator(for: x), accumulation.accumulator(for: scale), accumulation.accumulator(for: shift))
        }

        var gradients: [DoubleTensor?] {
            [inputGradient.value, scaleGradient.value, shiftGradient.value]
        }
    }

    static let normalization: [KernelCase] = [
        KernelCase("layerNormalization along the last axis") { ops, accumulation, inputs in
            var n = Normalization(inputs, accumulation, x: uniform([6, 5, 64], seed: 1, requiresGradient: true), scale: uniform([64], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), shift: uniform([64], seed: 3, requiresGradient: true))
            ops.layerNormalizationBackward(input: n.x, scale: n.scale, shift: n.shift, outputGradient: n.outputGradient, epsilon: 1e-5, inputGradient: n.inputGradient.buffer(), scaleGradient: n.scaleGradient.buffer(), shiftGradient: n.shiftGradient.buffer())
            return [output(n.x.shape) { ops.layerNormalization(input: n.x, scale: n.scale, shift: n.shift, epsilon: 1e-5, result: $0) }] + n.gradients
        },
        KernelCase("layerNormalization along 2 axes without input gradient") { ops, accumulation, inputs in
            var n = Normalization(inputs, accumulation, x: uniform([6, 5, 64], seed: 1), scale: uniform([5, 64], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), shift: uniform([5, 64], seed: 3, requiresGradient: true))
            ops.layerNormalizationBackward(input: n.x, scale: n.scale, shift: n.shift, outputGradient: n.outputGradient, epsilon: 1e-5, inputGradient: n.inputGradient.buffer(), scaleGradient: n.scaleGradient.buffer(), shiftGradient: n.shiftGradient.buffer())
            return [output(n.x.shape) { ops.layerNormalization(input: n.x, scale: n.scale, shift: n.shift, epsilon: 1e-5, result: $0) }] + n.gradients
        },
        KernelCase("batchNormalization") { ops, accumulation, inputs in
            batchNormalization(ops, inputs, Normalization(inputs, accumulation, x: uniform([9, 20], seed: 1, requiresGradient: true), scale: uniform([20], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), shift: uniform([20], seed: 3, requiresGradient: true)))
        },
        KernelCase("batchNormalization with a broadcast scale") { ops, accumulation, inputs in
            batchNormalization(ops, inputs, Normalization(inputs, accumulation, x: uniform([9, 4, 3, 3], seed: 1, requiresGradient: true), scale: uniform([4, 1, 1], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), shift: uniform([4, 1, 1], seed: 3, requiresGradient: true)))
        },
        KernelCase("batchNormalization with fixed statistics") { ops, accumulation, inputs in
            var n = Normalization(inputs, accumulation, x: uniform([9, 4, 3, 3], seed: 1, requiresGradient: true), scale: uniform([4, 1, 1], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), shift: uniform([4, 1, 1], seed: 3, requiresGradient: true))
            let (mean, variance) = (inputs.buffer(uniform([4, 3, 3], seed: 5)), inputs.buffer(uniform([4, 1, 1], min: 0.5, max: 2, seed: 6)))
            ops.batchNormalizationBackward(input: n.x, scale: n.scale, shift: n.shift, mean: mean, variance: variance, outputGradient: n.outputGradient, epsilon: 1e-5, inputGradient: n.inputGradient.buffer(), scaleGradient: n.scaleGradient.buffer(), shiftGradient: n.shiftGradient.buffer())
            return [output(n.x.shape) { ops.batchNormalization(input: n.x, scale: n.scale, shift: n.shift, mean: mean, variance: variance, epsilon: 1e-5, result: $0) }] + n.gradients
        },
    ]

    private static func batchNormalization(_ ops: Operations, _ inputs: Inputs, _ normalization: Normalization) -> [DoubleTensor?] {
        var n = normalization
        let columnShape = Array(n.x.shape.dropFirst())
        var (result, mean, variance) = (DoubleTensor(uninitializedShape: n.x.shape), DoubleTensor(uninitializedShape: columnShape), DoubleTensor(uninitializedShape: columnShape))
        ops.batchNormalization(input: n.x, scale: n.scale, shift: n.shift, epsilon: 1e-5, result: result.mutableValues, mean: mean.mutableValues, variance: variance.mutableValues)
        ops.batchNormalizationBackward(input: n.x, scale: n.scale, shift: n.shift, outputGradient: n.outputGradient, epsilon: 1e-5, inputGradient: n.inputGradient.buffer(), scaleGradient: n.scaleGradient.buffer(), shiftGradient: n.shiftGradient.buffer())
        return [result, mean, variance] + n.gradients
    }

    /// Computes a convolution and its gradients. `forward` and `backward` receive the input, the filters, and the bias.
    private static func convolution(
        _ inputs: Inputs,
        _ accumulation: Accumulation,
        x: DoubleTensor,
        filters: DoubleTensor,
        bias: DoubleTensor?,
        outputShape: [Int],
        forward: (ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>?, MutableShapedBuffer<Double, CPU>) -> Void,
        backward: (ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>?, ShapedBuffer<Double, CPU>, GradientBuffer<Double, CPU>?, GradientBuffer<Double, CPU>?, GradientBuffer<Double, CPU>?) -> Void,
    ) -> [DoubleTensor?] {
        let (xs, filterValues, biasValues) = (inputs.buffer(x), inputs.buffer(filters), inputs.buffer(bias))
        var (inputGradient, filterGradient) = (accumulation.accumulator(for: x), accumulation.accumulator(for: filters))
        var biasGradient = bias.map { accumulation.accumulator(for: $0) } ?? Gradient(isRequested: false, shape: [])
        backward(xs, filterValues, biasValues, inputs.buffer(uniform(outputShape, seed: 4)), inputGradient.buffer(), filterGradient.buffer(), biasGradient.buffer())
        return [output(outputShape) { forward(xs, filterValues, biasValues, $0) }, inputGradient.value, filterGradient.value, biasGradient.value]
    }

    /// Computes a pooling operation and its gradient.
    private static func pooling(
        _ inputs: Inputs,
        _ accumulation: Accumulation,
        shape: [Int],
        outputShape: [Int],
        forward: (ShapedBuffer<Double, CPU>, MutableShapedBuffer<Double, CPU>) -> Void,
        backward: (ShapedBuffer<Double, CPU>, ShapedBuffer<Double, CPU>, GradientBuffer<Double, CPU>?) -> Void,
    ) -> [DoubleTensor?] {
        let x = inputs.buffer(uniform(shape, seed: 1))
        var gradient = accumulation.accumulator(shape)
        backward(x, inputs.buffer(uniform(outputShape, seed: 2)), gradient.buffer())
        return [output(outputShape) { forward(x, $0) }, gradient.value]
    }

    static let convolution: [KernelCase] = [
        KernelCase("convolution2d with several images per chunk") { ops, accumulation, inputs in
            convolution(inputs, accumulation, x: uniform([5, 3, 9, 8], seed: 1, requiresGradient: true), filters: uniform([4, 3, 3, 2], seed: 2, requiresGradient: true), bias: uniform([4], seed: 3, requiresGradient: true), outputShape: [5, 4, 5, 5]) {
                ops.convolution2d(input: $0, filters: $1, bias: $2, padding: 1, stride: 2, result: $3)
            } backward: {
                ops.convolution2dBackward(input: $0, filters: $1, bias: $2, outputGradient: $3, padding: 1, stride: 2, inputGradient: $4, filterGradient: $5, biasGradient: $6)
            }
        },
        KernelCase("convolution2d with one image per chunk and without bias") { ops, accumulation, inputs in
            convolution(inputs, accumulation, x: uniform([2, 64, 64, 64], seed: 1, requiresGradient: true), filters: uniform([8, 64, 3, 3], seed: 2, requiresGradient: true), bias: nil, outputShape: [2, 8, 64, 64]) {
                ops.convolution2d(input: $0, filters: $1, bias: $2, padding: 1, stride: 1, result: $3)
            } backward: {
                ops.convolution2dBackward(input: $0, filters: $1, bias: $2, outputGradient: $3, padding: 1, stride: 1, inputGradient: $4, filterGradient: $5, biasGradient: $6)
            }
        },
        KernelCase("convolution2d with a bias for one image per chunk") { ops, accumulation, inputs in
            convolution(inputs, accumulation, x: uniform([1, 64, 64, 64], seed: 1), filters: uniform([8, 64, 3, 3], seed: 2, requiresGradient: true), bias: uniform([8], seed: 3, requiresGradient: true), outputShape: [1, 8, 64, 64]) {
                ops.convolution2d(input: $0, filters: $1, bias: $2, padding: 1, stride: 1, result: $3)
            } backward: {
                ops.convolution2dBackward(input: $0, filters: $1, bias: $2, outputGradient: $3, padding: 1, stride: 1, inputGradient: $4, filterGradient: $5, biasGradient: $6)
            }
        },
        KernelCase("transposedConvolution2d with several images per chunk") { ops, accumulation, inputs in
            convolution(inputs, accumulation, x: uniform([5, 3, 4, 4], seed: 1, requiresGradient: true), filters: uniform([2, 3, 3, 3], seed: 2, requiresGradient: true), bias: uniform([2], seed: 3, requiresGradient: true), outputShape: [5, 2, 7, 7]) {
                ops.transposedConvolution2d(input: $0, filters: $1, bias: $2, inset: 1, stride: 2, result: $3)
            } backward: {
                ops.transposedConvolution2dBackward(input: $0, filters: $1, bias: $2, outputGradient: $3, inset: 1, stride: 2, inputGradient: $4, filterGradient: $5, biasGradient: $6)
            }
        },
        KernelCase("transposedConvolution2d with one image per chunk") { ops, accumulation, inputs in
            convolution(inputs, accumulation, x: uniform([2, 8, 64, 64], seed: 1, requiresGradient: true), filters: uniform([16, 8, 4, 4], seed: 2, requiresGradient: true), bias: uniform([16], seed: 3, requiresGradient: true), outputShape: [2, 16, 128, 128]) {
                ops.transposedConvolution2d(input: $0, filters: $1, bias: $2, inset: 1, stride: 2, result: $3)
            } backward: {
                ops.transposedConvolution2dBackward(input: $0, filters: $1, bias: $2, outputGradient: $3, inset: 1, stride: 2, inputGradient: $4, filterGradient: $5, biasGradient: $6)
            }
        },
        KernelCase("maxPooling2d with padding and overlapping windows") { ops, accumulation, inputs in
            pooling(inputs, accumulation, shape: [3, 2, 7, 7], outputShape: [3, 2, 4, 4]) {
                ops.maxPooling2d(input: $0, windowSize: 3, padding: 1, stride: 2, result: $1)
            } backward: {
                ops.maxPooling2dBackward(input: $0, outputGradient: $1, windowSize: 3, padding: 1, stride: 2, inputGradient: $2)
            }
        },
        KernelCase("maxPooling2d") { ops, accumulation, inputs in
            pooling(inputs, accumulation, shape: [3, 2, 8, 8], outputShape: [3, 2, 4, 4]) {
                ops.maxPooling2d(input: $0, windowSize: 2, padding: 0, stride: 2, result: $1)
            } backward: {
                ops.maxPooling2dBackward(input: $0, outputGradient: $1, windowSize: 2, padding: 0, stride: 2, inputGradient: $2)
            }
        },
        KernelCase("maxPooling2d with an odd size") { ops, accumulation, inputs in
            pooling(inputs, accumulation, shape: [2, 3, 9, 7], outputShape: [2, 3, 4, 3]) {
                ops.maxPooling2d(input: $0, windowSize: 2, padding: 0, stride: 2, result: $1)
            } backward: {
                ops.maxPooling2dBackward(input: $0, outputGradient: $1, windowSize: 2, padding: 0, stride: 2, inputGradient: $2)
            }
        },
        KernelCase("averagePooling2d") { ops, accumulation, inputs in
            pooling(inputs, accumulation, shape: [3, 2, 7, 7], outputShape: [3, 2, 4, 4]) {
                ops.averagePooling2d(input: $0, windowSize: 3, padding: 1, stride: 2, result: $1)
            } backward: {
                ops.averagePooling2dBackward(input: $0, outputGradient: $1, windowSize: 3, padding: 1, stride: 2, inputGradient: $2)
            }
        },
        KernelCase("averagePooling2d without padding") { ops, accumulation, inputs in
            pooling(inputs, accumulation, shape: [3, 2, 8, 9], outputShape: [3, 2, 4, 4]) {
                ops.averagePooling2d(input: $0, windowSize: 2, padding: 0, stride: 2, result: $1)
            } backward: {
                ops.averagePooling2dBackward(input: $0, outputGradient: $1, windowSize: 2, padding: 0, stride: 2, inputGradient: $2)
            }
        },
    ]

    private static func attention(_ ops: Operations, _ accumulation: Accumulation, _ inputs: Inputs, queries: DoubleTensor, keys: DoubleTensor, values: DoubleTensor, mask: DoubleTensor?) -> [DoubleTensor?] {
        var (queryGradient, keyGradient, valueGradient) = (accumulation.accumulator(for: queries), accumulation.accumulator(for: keys), accumulation.accumulator(for: values))
        let outputShape = Array(queries.shape.dropLast()) + [values.shape[values.dim - 1]]
        let (q, k, v, m) = (inputs.buffer(queries), inputs.buffer(keys), inputs.buffer(values), inputs.buffer(mask))
        ops.scaledDotProductAttentionBackward(queries: q, keys: k, values: v, mask: m, outputGradient: inputs.buffer(uniform(outputShape, seed: 4)), temperature: 2, queryGradient: queryGradient.buffer(), keyGradient: keyGradient.buffer(), valueGradient: valueGradient.buffer())
        return [output(outputShape) { ops.scaledDotProductAttention(queries: q, keys: k, values: v, mask: m, temperature: 2, result: $0) }, queryGradient.value, keyGradient.value, valueGradient.value]
    }

    static let attention: [KernelCase] = [
        KernelCase("scaledDotProductAttention with a mask per batch") { ops, accumulation, inputs in
            let mask = DoubleTensor([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1], shape: [2, 1, 1, 6])
            return attention(
                ops, accumulation, inputs,
                queries: uniform([2, 3, 5, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2, requiresGradient: true), values: uniform([2, 3, 6, 3], seed: 3, requiresGradient: true), mask: mask,
            )
        },
        KernelCase("scaledDotProductAttention with a causal mask") { ops, accumulation, inputs in
            let mask = DoubleTensor((0 ..< 36).map { $0 % 6 > $0 / 6 ? 1 : 0 }, shape: [6, 6])
            return attention(
                ops, accumulation, inputs,
                queries: uniform([2, 3, 6, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2), values: uniform([2, 3, 6, 3], seed: 3, requiresGradient: true), mask: mask,
            )
        },
        KernelCase("scaledDotProductAttention with grouped heads") { ops, accumulation, inputs in
            let mask = DoubleTensor((0 ..< 24).map { $0 % 5 == 1 ? 1 : 0 }, shape: [1, 4, 1, 6])
            return attention(
                ops, accumulation, inputs,
                queries: uniform([2, 4, 5, 4], seed: 1, requiresGradient: true), keys: uniform([2, 2, 6, 4], seed: 2, requiresGradient: true), values: uniform([2, 1, 6, 3], seed: 3, requiresGradient: true), mask: mask,
            )
        },
        KernelCase("scaledDotProductAttention without mask") { ops, accumulation, inputs in
            attention(
                ops, accumulation, inputs,
                queries: uniform([2, 3, 5, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2, requiresGradient: true), values: uniform([2, 3, 6, 3], seed: 3), mask: nil,
            )
        },
    ] + [([true, true, true, true, true, true, true], 2), ([false, false, true, true, false, false, false], 2), ([true, false, false, false, false, true, false], 2), ([true, true, true, true, true, true, true], 1)].map { computes, keyHeads in
        KernelCase("multiHeadAttention with \(keyHeads) key heads computing \(computes)") { ops, accumulation, inputs in
            let shapes = [[2, 5, 8], [2, 6, 8], [2, 6, 8], [8, 8], [8, 4 * keyHeads], [8, 6 * keyHeads], [12, 7]]
            let sources = shapes.enumerated().map { index, shape in
                uniform(shape, min: -1, max: 1, seed: UInt64(index + 1), requiresGradient: computes[index])
            }
            let s = sources.map { inputs.buffer($0) }
            let mask = inputs.buffer(DoubleTensor((0 ..< 30).map { $0 % 6 > $0 / 6 ? 1 : 0 }, shape: [5, 6]))
            var accumulators = sources.map { accumulation.accumulator(for: $0) }
            ops.multiHeadAttentionBackward(
                queries: s[0], keys: s[1], values: s[2], mask: mask,
                queryWeights: s[3], keyWeights: s[4], valueWeights: s[5], outputWeights: s[6],
                outputGradient: inputs.buffer(uniform([2, 5, 7], seed: 9)), heads: 2, temperature: 2,
                gradients: MultiHeadAttentionGradients(inSourceOrder: accumulators.indices.map { accumulators[$0].buffer() }),
            )
            let result = output([2, 5, 7]) {
                ops.multiHeadAttention(
                    queries: s[0], keys: s[1], values: s[2], mask: mask,
                    queryWeights: s[3], keyWeights: s[4], valueWeights: s[5], outputWeights: s[6],
                    heads: 2, temperature: 2, result: $0,
                )
            }
            return [result] + accumulators.map(\.value)
        }
    }

    static let losses: [KernelCase] = [
        KernelCase("categoricalCrossEntropy with an ignored label") { ops, accumulation, inputs in
            let actual = uniform([6, 10], min: 0.05, max: 1, seed: 1, requiresGradient: true)
            let (e, a) = (inputs.buffer(Tensor<Int32, CPU>([3, -1, 0, 9, 3, 5])), inputs.buffer(actual))
            var gradient = accumulation.accumulator(for: actual)
            ops.categoricalCrossEntropyBackward(expected: e, actual: a, outputGradient: inputs.buffer(DoubleTensor(1.5)), ignoreIndex: -1, actualGradient: gradient.buffer())
            return [output([]) { ops.categoricalCrossEntropy(expected: e, actual: a, ignoreIndex: -1, result: $0) }, gradient.value]
        },
        KernelCase("categoricalNegativeLogLikelihood with an ignored label") { ops, accumulation, inputs in
            let actual = uniform([2, 2, 8], seed: 1, requiresGradient: true)
            let (e, a) = (inputs.buffer(Tensor<Int32, CPU>([[3, 7], [0, 4]])), inputs.buffer(actual))
            var gradient = accumulation.accumulator(for: actual)
            ops.categoricalNegativeLogLikelihoodBackward(expected: e, actual: a, outputGradient: inputs.buffer(DoubleTensor(0.7)), ignoreIndex: 4, actualGradient: gradient.buffer())
            return [output([]) { ops.categoricalNegativeLogLikelihood(expected: e, actual: a, ignoreIndex: 4, result: $0) }, gradient.value]
        },
        KernelCase("binaryCrossEntropy") { ops, accumulation, inputs in
            let (expected, actual) = (uniform([7, 1000], min: 0, max: 1, seed: 1, requiresGradient: true), uniform([7, 1000], min: 0.05, max: 0.95, seed: 2, requiresGradient: true))
            let (e, a) = (inputs.buffer(expected), inputs.buffer(actual))
            var (expectedGradient, actualGradient) = (accumulation.accumulator(for: expected), accumulation.accumulator(for: actual))
            ops.binaryCrossEntropyBackward(expected: e, actual: a, outputGradient: inputs.buffer(DoubleTensor(1.3)), expectedGradient: expectedGradient.buffer(), actualGradient: actualGradient.buffer())
            return [output([]) { ops.binaryCrossEntropy(expected: e, actual: a, result: $0) }, expectedGradient.value, actualGradient.value]
        },
        KernelCase("binaryCrossEntropy with predictions of another shape") { ops, accumulation, inputs in
            let (expected, actual) = (uniform([1000], min: 0, max: 1, seed: 1, requiresGradient: true), uniform([1000, 1], min: 0.05, max: 0.95, seed: 2, requiresGradient: true))
            let (e, a) = (inputs.buffer(expected), inputs.buffer(actual))
            var (expectedGradient, actualGradient) = (accumulation.accumulator(for: expected), accumulation.accumulator(for: actual))
            ops.binaryCrossEntropyBackward(expected: e, actual: a, outputGradient: inputs.buffer(DoubleTensor(1.3)), expectedGradient: expectedGradient.buffer(), actualGradient: actualGradient.buffer())
            return [output([]) { ops.binaryCrossEntropy(expected: e, actual: a, result: $0) }, expectedGradient.value, actualGradient.value]
        },
        KernelCase("meanSquaredError") { ops, accumulation, inputs in
            let (expected, actual) = (uniform([7, 1000], seed: 1), uniform([7, 1000], seed: 2, requiresGradient: true))
            let (e, a) = (inputs.buffer(expected), inputs.buffer(actual))
            var (expectedGradient, actualGradient) = (accumulation.accumulator(for: expected), accumulation.accumulator(for: actual))
            ops.meanSquaredErrorBackward(expected: e, actual: a, outputGradient: inputs.buffer(DoubleTensor(0.4)), expectedGradient: expectedGradient.buffer(), actualGradient: actualGradient.buffer())
            return [output([]) { ops.meanSquaredError(expected: e, actual: a, result: $0) }, expectedGradient.value, actualGradient.value]
        },
    ]

    static let layers: [KernelCase] = [true, false].map { hasBias in
        KernelCase(hasBias ? "linear" : "linear without bias") { ops, _, inputs in
            let (input, weights, bias) = (inputs.buffer(uniform([9, 13], seed: 1)), inputs.buffer(uniform([13, 6], seed: 2)), inputs.buffer(hasBias ? uniform([6], seed: 3) : nil))
            return [output([9, 6]) { ops.linear(input: input, weights: weights, bias: bias, result: $0) }]
        }
    }

    static let optimizers: [KernelCase] = [false, true].map { usesMaximum in
        KernelCase(usesMaximum ? "adamUpdate with AMSGrad" : "adamUpdate") { ops, _, inputs in
            var (firstMoment, secondMoment) = (uniform([7, 1000], seed: 3), uniform([7, 1000], min: 0, max: 1, seed: 4))
            var maximum: DoubleTensor? = usesMaximum ? uniform([7, 1000], min: 0, max: 1, seed: 5) : nil
            let parameter = output([7, 1000]) {
                ops.adamUpdate(
                    parameter: inputs.buffer(uniform([7, 1000], seed: 1)), gradient: inputs.buffer(uniform([7, 1000], seed: 2)),
                    firstMoment: firstMoment.mutableValues, secondMoment: secondMoment.mutableValues, secondMomentMax: maximum?.mutableValues,
                    learningRate: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, beta1Power: 0.9 * 0.9, beta2Power: 0.999 * 0.999,
                    result: $0,
                )
            }
            return [parameter, firstMoment, secondMoment, maximum]
        }
    }

    static let recurrent: [KernelCase] = [[true, true, true, true, true, true, true], [true, true, true, false, true, true, true], [false, false, false, true, false, false, false]].map { computes in
        KernelCase("gatedRecurrentUnitStep computing \(computes)") { ops, accumulation, inputs in
            let sources = [[5, 16], [5, 16], [5, 16], [5, 16], [16, 16], [16, 16], [16, 16]].enumerated().map { index, shape in
                uniform(shape, min: -1, max: 1, seed: UInt64(index + 1), requiresGradient: computes[index])
            }
            let s = sources.map { inputs.buffer($0) }
            var accumulators = sources.map { accumulation.accumulator(for: $0) }
            ops.gatedRecurrentUnitStepBackward(
                updateInput: s[0], resetInput: s[1], candidateInput: s[2], state: s[3],
                updateWeights: s[4], resetWeights: s[5], candidateWeights: s[6], outputGradient: inputs.buffer(uniform([5, 16], seed: 9)),
                gradients: GatedRecurrentUnitGradients(inSourceOrder: accumulators.indices.map { accumulators[$0].buffer() }),
            )
            let step = output([5, 16]) {
                ops.gatedRecurrentUnitStep(
                    updateInput: s[0], resetInput: s[1], candidateInput: s[2], state: s[3],
                    updateWeights: s[4], resetWeights: s[5], candidateWeights: s[6], result: $0,
                )
            }
            return [step] + accumulators.map(\.value)
        }
    }

    static let all = activations + softmax + normalization + convolution + attention + losses + layers + optimizers + recurrent
}

/// Compares the fused kernels of the CPU with the default implementations of the fused operations.
struct CPUFusedKernelTests {
    @Test(arguments: KernelCase.all, Accumulation.allCases)
    func kernelMatchesDefaultImplementation(_ kernelCase: KernelCase, _ accumulation: Accumulation) {
        let fused = kernelCase.run(CPUFusedOperations.self, accumulation)
        let composed = kernelCase.run(DefaultFusedOperations<CPU>.self, accumulation)
        #expect(fused.count == composed.count)
        for (index, (actual, expected)) in zip(fused, composed).enumerated() {
            guard let actual, let expected else {
                #expect(actual == nil && expected == nil, "result \(index): only one implementation computes it")
                continue
            }
            #expect(!actual.requiresGradient, "result \(index)")
            guard actual.shape == expected.shape else {
                Issue.record("result \(index): shape \(actual.shape) differs from \(expected.shape)")
                continue
            }
            let difference = zip(actual.elements, expected.elements).map { abs($0 - $1) }.max() ?? 0
            let magnitude = expected.elements.map(abs).max() ?? 0
            #expect(difference <= 1e-9 * (1 + magnitude), "result \(index): difference \(difference)")
        }
    }

    /// Every result with a prefilled accumulator is the result without it, or the result without it plus the start value.
    @Test(arguments: KernelCase.all)
    func kernelAddsToAccumulatedGradients(_ kernelCase: KernelCase) throws {
        let empty = kernelCase.run(CPUFusedOperations.self, .empty)
        let prefilled = kernelCase.run(CPUFusedOperations.self, .prefilled)
        #expect(empty.count == prefilled.count)
        for (index, (withoutStart, withStart)) in zip(empty, prefilled).enumerated() {
            guard let withStart else {
                #expect(withoutStart == nil, "result \(index): the prefilled run returns no value")
                continue
            }
            let start = try #require(Accumulation.prefilled.start(withStart.shape))
            guard let withoutStart else {
                #expect(withStart == start, "result \(index): the start value of a gradient that is not computed changed")
                continue
            }
            let magnitude = withStart.elements.map(abs).max() ?? 0
            let unchanged = zip(withStart.elements, withoutStart.elements).map { abs($0 - $1) }.max() ?? 0
            let added = zip(withStart.elements, (withoutStart + start).elements).map { abs($0 - $1) }.max() ?? 0
            #expect(min(unchanged, added) <= 1e-9 * (1 + magnitude), "result \(index): difference \(added)")
        }
    }

    @Test func dropoutKeepsElementsWithTheGivenProbability() {
        let input = uniform([100_000], min: 1, max: 2, seed: 1)
        let (output, mask) = dropout(input, rate: 0.3)
        #expect(mask.elements.allSatisfy { $0 == 0 || $0 == 1 })
        #expect(output == input * mask)
        let keptFraction = mask.elements.reduce(0, +) / 100_000
        #expect(abs(keptFraction - 0.7) < 0.01, "kept fraction \(keptFraction)")

        #expect(dropout(input, rate: 0).mask.elements.allSatisfy { $0 == 1 })
        #expect(dropout(input, rate: 1).mask.elements.allSatisfy { $0 == 0 })
    }

    /// The result and the mask of the dropout kernel of the CPU.
    private func dropout(_ input: DoubleTensor, rate: Float) -> (output: DoubleTensor, mask: DoubleTensor) {
        var (output, mask) = (DoubleTensor(uninitializedShape: input.shape), DoubleTensor(uninitializedShape: input.shape))
        withExtendedLifetime(input) {
            CPUFusedOperations.dropout(input: input.values, rate: rate, result: output.mutableValues, mask: mask.mutableValues)
        }
        return (output, mask)
    }
}

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

import DL4S
import Testing

private typealias DoubleTensor = Tensor<Double, CPU>

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

    /// Returns nil, or a tensor with the shape of `source` that the backward requirement adds the gradient to.
    fileprivate func start(for source: DoubleTensor) -> DoubleTensor? {
        start(source.shape)
    }
}

/// A call of fused requirements, with inputs that exercise the kernels of the CPU.
struct KernelCase: CustomTestStringConvertible, Sendable {
    let name: String
    fileprivate let run: @Sendable (Operations, Accumulation) -> [DoubleTensor?]

    var testDescription: String {
        name
    }

    fileprivate init(_ name: String, _ run: @escaping @Sendable (Operations, Accumulation) -> [DoubleTensor?]) {
        self.name = name
        self.run = run
    }

    /// The results of a forward requirement and of its backward requirement for an input that requires a gradient.
    fileprivate static func unary(
        _ name: String,
        shape: [Int] = [7, 1000],
        forward: @escaping @Sendable (Operations, DoubleTensor) -> DoubleTensor,
        backward: @escaping @Sendable (Operations, DoubleTensor, DoubleTensor, inout DoubleTensor?) -> Void,
    ) -> KernelCase {
        KernelCase(name) { ops, accumulation in
            let x = uniform(shape, seed: 1, requiresGradient: true)
            var gradient = accumulation.start(for: x)
            backward(ops, x, uniform(shape, seed: 2), &gradient)
            return [forward(ops, x), gradient]
        }
    }
}

extension KernelCase {
    static let activations: [KernelCase] = [
        KernelCase("tanhBackward") { ops, accumulation in
            var gradient = accumulation.start([7, 1000])
            ops.tanhBackward(output: uniform([7, 1000], min: -1, max: 1, seed: 1), outputGradient: uniform([7, 1000], seed: 2), accumulating: &gradient)
            return [gradient]
        },
        .unary("rectified linear", forward: { _, x in x.detached().rectifiedLinear() }, backward: { ops, x, g, gradient in ops.reluBackward(input: x, outputGradient: g, accumulating: &gradient) }),
        KernelCase("sigmoid") { ops, accumulation in
            let x = uniform([7, 1000], seed: 1)
            var gradient = accumulation.start(for: x)
            ops.sigmoidBackward(output: ops.sigmoid(input: x), outputGradient: uniform([7, 1000], seed: 2), accumulating: &gradient)
            return [ops.sigmoid(input: x), gradient]
        },
        KernelCase("leakyRelu") { ops, accumulation in
            let (x, leakage, outputGradient) = (uniform([7, 1000], seed: 1, requiresGradient: true), DoubleTensor(0.2, requiresGradient: true), uniform([7, 1000], seed: 2))
            var gradients = (input: accumulation.start(for: x), leakage: accumulation.start(for: leakage))
            ops.leakyReluBackward(input: x, leakage: leakage, outputGradient: outputGradient, accumulating: &gradients)
            return [ops.leakyRelu(input: x, leakage: leakage), gradients.input, gradients.leakage]
        },
        .unary("gelu", forward: { ops, x in ops.gelu(input: x) }, backward: { ops, x, g, gradient in ops.geluBackward(input: x, outputGradient: g, accumulating: &gradient) }),
        KernelCase("swish with a scalar beta") { ops, accumulation in
            swish(ops, accumulation, x: uniform([7, 1000], seed: 1, requiresGradient: true), beta: DoubleTensor(1.3, requiresGradient: true))
        },
        KernelCase("swish with a beta per channel") { ops, accumulation in
            swish(ops, accumulation, x: uniform([7, 1000], seed: 1, requiresGradient: true), beta: uniform([1000], min: 0.5, max: 2, seed: 3, requiresGradient: true))
        },
        KernelCase("swish with a broadcast beta") { ops, accumulation in
            swish(ops, accumulation, x: uniform([7, 10, 3], seed: 1, requiresGradient: true), beta: uniform([10, 1], seed: 3, requiresGradient: true))
        },
        .unary("mish", forward: { ops, x in ops.mish(input: x) }, backward: { ops, x, g, gradient in ops.mishBackward(input: x, outputGradient: g, accumulating: &gradient) }),
        .unary("lisht", forward: { ops, x in ops.lisht(input: x) }, backward: { ops, x, g, gradient in ops.lishtBackward(input: x, outputGradient: g, accumulating: &gradient) }),
        KernelCase("elu") { ops, accumulation in
            let (x, alpha, outputGradient) = (uniform([7, 1000], seed: 1, requiresGradient: true), DoubleTensor(0.7, requiresGradient: true), uniform([7, 1000], seed: 2))
            var gradients = (input: accumulation.start(for: x), alpha: accumulation.start(for: alpha))
            ops.eluBackward(input: x, alpha: alpha, outputGradient: outputGradient, accumulating: &gradients)
            return [ops.elu(input: x, alpha: alpha), gradients.input, gradients.alpha]
        },
        .unary("softplus", forward: { ops, x in ops.softplus(input: x) }, backward: { ops, x, g, gradient in ops.softplusBackward(input: x, outputGradient: g, accumulating: &gradient) }),
        .unary("squareplus", forward: { ops, x in ops.squareplus(input: x) }, backward: { ops, x, g, gradient in ops.squareplusBackward(input: x, outputGradient: g, accumulating: &gradient) }),
    ]

    private static func swish(_ ops: Operations, _ accumulation: Accumulation, x: DoubleTensor, beta: DoubleTensor) -> [DoubleTensor?] {
        var gradients = (input: accumulation.start(for: x), beta: accumulation.start(for: beta))
        ops.swishBackward(input: x, beta: beta, outputGradient: uniform(x.shape, seed: 2), accumulating: &gradients)
        return [ops.swish(input: x, beta: beta), gradients.input, gradients.beta]
    }

    static let softmax: [KernelCase] = [
        .unary("softmax along long rows", forward: { ops, x in ops.softmax(input: x, axis: 1) }, backward: { ops, x, g, gradient in ops.softmaxBackward(output: ops.softmax(input: x, axis: 1), outputGradient: g, axis: 1, accumulating: &gradient) }),
        .unary("softmax along short rows", shape: [300, 10], forward: { ops, x in ops.softmax(input: x, axis: 1) }, backward: { ops, x, g, gradient in ops.softmaxBackward(output: ops.softmax(input: x, axis: 1), outputGradient: g, axis: 1, accumulating: &gradient) }),
        .unary("softmax along the first axis", shape: [4, 3, 5], forward: { ops, x in ops.softmax(input: x, axis: 0) }, backward: { ops, x, g, gradient in ops.softmaxBackward(output: ops.softmax(input: x, axis: 0), outputGradient: g, axis: 0, accumulating: &gradient) }),
        .unary("logSoftmax along long rows", forward: { ops, x in ops.logSoftmax(input: x, axis: 1) }, backward: { ops, x, g, gradient in ops.logSoftmaxBackward(output: ops.logSoftmax(input: x, axis: 1), outputGradient: g, axis: 1, accumulating: &gradient) }),
        .unary("logSoftmax along short rows", shape: [300, 10], forward: { ops, x in ops.logSoftmax(input: x, axis: 1) }, backward: { ops, x, g, gradient in ops.logSoftmaxBackward(output: ops.logSoftmax(input: x, axis: 1), outputGradient: g, axis: 1, accumulating: &gradient) }),
    ]

    private typealias NormalizationGradients = (input: DoubleTensor?, scale: DoubleTensor?, shift: DoubleTensor?)

    private static func normalizationStart(_ accumulation: Accumulation, _ x: DoubleTensor, _ scale: DoubleTensor, _ shift: DoubleTensor) -> NormalizationGradients {
        (accumulation.start(for: x), accumulation.start(for: scale), accumulation.start(for: shift))
    }

    static let normalization: [KernelCase] = [
        KernelCase("layerNormalization along the last axis") { ops, accumulation in
            let (x, scale, shift) = (uniform([6, 5, 64], seed: 1, requiresGradient: true), uniform([64], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), uniform([64], seed: 3, requiresGradient: true))
            var gradients = normalizationStart(accumulation, x, scale, shift)
            ops.layerNormalizationBackward(input: x, scale: scale, shift: shift, outputGradient: uniform([6, 5, 64], seed: 4), epsilon: 1e-5, accumulating: &gradients)
            return [ops.layerNormalization(input: x, scale: scale, shift: shift, epsilon: 1e-5), gradients.input, gradients.scale, gradients.shift]
        },
        KernelCase("layerNormalization along 2 axes without input gradient") { ops, accumulation in
            let (x, scale, shift) = (uniform([6, 5, 64], seed: 1), uniform([5, 64], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), uniform([5, 64], seed: 3, requiresGradient: true))
            var gradients = normalizationStart(accumulation, x, scale, shift)
            ops.layerNormalizationBackward(input: x, scale: scale, shift: shift, outputGradient: uniform([6, 5, 64], seed: 4), epsilon: 1e-5, accumulating: &gradients)
            return [ops.layerNormalization(input: x, scale: scale, shift: shift, epsilon: 1e-5), gradients.input, gradients.scale, gradients.shift]
        },
        KernelCase("batchNormalization") { ops, accumulation in
            let (x, scale, shift) = (uniform([9, 20], seed: 1, requiresGradient: true), uniform([20], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), uniform([20], seed: 3, requiresGradient: true))
            let forward = ops.batchNormalization(input: x, scale: scale, shift: shift, epsilon: 1e-5)
            var gradients = normalizationStart(accumulation, x, scale, shift)
            ops.batchNormalizationBackward(input: x, scale: scale, shift: shift, outputGradient: uniform([9, 20], seed: 4), epsilon: 1e-5, accumulating: &gradients)
            return [forward.output, forward.mean, forward.variance, gradients.input, gradients.scale, gradients.shift]
        },
        KernelCase("batchNormalization with a broadcast scale") { ops, accumulation in
            let (x, scale, shift) = (uniform([9, 4, 3, 3], seed: 1, requiresGradient: true), uniform([4, 1, 1], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), uniform([4, 1, 1], seed: 3, requiresGradient: true))
            let forward = ops.batchNormalization(input: x, scale: scale, shift: shift, epsilon: 1e-5)
            var gradients = normalizationStart(accumulation, x, scale, shift)
            ops.batchNormalizationBackward(input: x, scale: scale, shift: shift, outputGradient: uniform([9, 4, 3, 3], seed: 4), epsilon: 1e-5, accumulating: &gradients)
            return [forward.output, forward.mean, forward.variance, gradients.input, gradients.scale, gradients.shift]
        },
        KernelCase("batchNormalization with fixed statistics") { ops, accumulation in
            let (x, scale, shift) = (uniform([9, 4, 3, 3], seed: 1, requiresGradient: true), uniform([4, 1, 1], min: 0.5, max: 1.5, seed: 2, requiresGradient: true), uniform([4, 1, 1], seed: 3, requiresGradient: true))
            let (mean, variance) = (uniform([4, 3, 3], seed: 5), uniform([4, 1, 1], min: 0.5, max: 2, seed: 6))
            var gradients = normalizationStart(accumulation, x, scale, shift)
            ops.batchNormalizationBackward(input: x, scale: scale, shift: shift, mean: mean, variance: variance, outputGradient: uniform([9, 4, 3, 3], seed: 4), epsilon: 1e-5, accumulating: &gradients)
            return [ops.batchNormalization(input: x, scale: scale, shift: shift, mean: mean, variance: variance, epsilon: 1e-5), gradients.input, gradients.scale, gradients.shift]
        },
    ]

    private typealias ConvolutionGradients = (input: DoubleTensor?, filters: DoubleTensor?, bias: DoubleTensor?)

    private static func convolutionStart(_ accumulation: Accumulation, _ x: DoubleTensor, _ filters: DoubleTensor, _ bias: DoubleTensor?) -> ConvolutionGradients {
        (accumulation.start(for: x), accumulation.start(for: filters), bias.flatMap { accumulation.start(for: $0) })
    }

    static let convolution: [KernelCase] = [
        KernelCase("convolution2d with several images per chunk") { ops, accumulation in
            let (x, filters, bias) = (uniform([5, 3, 9, 8], seed: 1, requiresGradient: true), uniform([4, 3, 3, 2], seed: 2, requiresGradient: true), uniform([4], seed: 3, requiresGradient: true))
            var gradients = convolutionStart(accumulation, x, filters, bias)
            ops.convolution2dBackward(input: x, filters: filters, bias: bias, outputGradient: uniform([5, 4, 5, 5], seed: 4), padding: 1, stride: 2, accumulating: &gradients)
            return [ops.convolution2d(input: x, filters: filters, bias: bias, padding: 1, stride: 2), gradients.input, gradients.filters, gradients.bias]
        },
        KernelCase("convolution2d with one image per chunk and without bias") { ops, accumulation in
            let (x, filters) = (uniform([2, 64, 64, 64], seed: 1, requiresGradient: true), uniform([8, 64, 3, 3], seed: 2, requiresGradient: true))
            var gradients = convolutionStart(accumulation, x, filters, nil)
            ops.convolution2dBackward(input: x, filters: filters, bias: nil, outputGradient: uniform([2, 8, 64, 64], seed: 4), padding: 1, stride: 1, accumulating: &gradients)
            return [ops.convolution2d(input: x, filters: filters, bias: nil, padding: 1, stride: 1), gradients.input, gradients.filters, gradients.bias]
        },
        KernelCase("convolution2d with a bias for one image per chunk") { ops, accumulation in
            let (x, filters, bias) = (uniform([1, 64, 64, 64], seed: 1), uniform([8, 64, 3, 3], seed: 2, requiresGradient: true), uniform([8], seed: 3, requiresGradient: true))
            var gradients = convolutionStart(accumulation, x, filters, bias)
            ops.convolution2dBackward(input: x, filters: filters, bias: bias, outputGradient: uniform([1, 8, 64, 64], seed: 4), padding: 1, stride: 1, accumulating: &gradients)
            return [ops.convolution2d(input: x, filters: filters, bias: bias, padding: 1, stride: 1), gradients.input, gradients.filters, gradients.bias]
        },
        KernelCase("transposedConvolution2d with several images per chunk") { ops, accumulation in
            let (x, filters, bias) = (uniform([5, 3, 4, 4], seed: 1, requiresGradient: true), uniform([2, 3, 3, 3], seed: 2, requiresGradient: true), uniform([2], seed: 3, requiresGradient: true))
            var gradients = convolutionStart(accumulation, x, filters, bias)
            ops.transposedConvolution2dBackward(input: x, filters: filters, bias: bias, outputGradient: uniform([5, 2, 7, 7], seed: 4), inset: 1, stride: 2, accumulating: &gradients)
            return [ops.transposedConvolution2d(input: x, filters: filters, bias: bias, inset: 1, stride: 2), gradients.input, gradients.filters, gradients.bias]
        },
        KernelCase("transposedConvolution2d with one image per chunk") { ops, accumulation in
            let (x, filters, bias) = (uniform([2, 8, 64, 64], seed: 1, requiresGradient: true), uniform([16, 8, 4, 4], seed: 2, requiresGradient: true), uniform([16], seed: 3, requiresGradient: true))
            var gradients = convolutionStart(accumulation, x, filters, bias)
            ops.transposedConvolution2dBackward(input: x, filters: filters, bias: bias, outputGradient: uniform([2, 16, 128, 128], seed: 4), inset: 1, stride: 2, accumulating: &gradients)
            return [ops.transposedConvolution2d(input: x, filters: filters, bias: bias, inset: 1, stride: 2), gradients.input, gradients.filters, gradients.bias]
        },
        KernelCase("maxPooling2d with padding and overlapping windows") { ops, accumulation in
            let x = uniform([3, 2, 7, 7], seed: 1)
            var gradient = accumulation.start(for: x)
            ops.maxPooling2dBackward(input: x, outputGradient: uniform([3, 2, 4, 4], seed: 2), windowSize: 3, padding: 1, stride: 2, accumulating: &gradient)
            return [ops.maxPooling2d(input: x, windowSize: 3, padding: 1, stride: 2), gradient]
        },
        KernelCase("maxPooling2d") { ops, accumulation in
            let x = uniform([3, 2, 8, 8], seed: 1)
            var gradient = accumulation.start(for: x)
            ops.maxPooling2dBackward(input: x, outputGradient: uniform([3, 2, 4, 4], seed: 2), windowSize: 2, padding: 0, stride: 2, accumulating: &gradient)
            return [ops.maxPooling2d(input: x, windowSize: 2, padding: 0, stride: 2), gradient]
        },
        KernelCase("averagePooling2d") { ops, accumulation in
            let x = uniform([3, 2, 7, 7], seed: 1)
            var gradient = accumulation.start(for: x)
            ops.averagePooling2dBackward(input: x, outputGradient: uniform([3, 2, 4, 4], seed: 2), windowSize: 3, padding: 1, stride: 2, accumulating: &gradient)
            return [ops.averagePooling2d(input: x, windowSize: 3, padding: 1, stride: 2), gradient]
        },
    ]

    private static func attention(_ ops: Operations, _ accumulation: Accumulation, queries: DoubleTensor, keys: DoubleTensor, values: DoubleTensor, mask: DoubleTensor?) -> [DoubleTensor?] {
        var gradients = (queries: accumulation.start(for: queries), keys: accumulation.start(for: keys), values: accumulation.start(for: values))
        let outputGradient = uniform(Array(queries.shape.dropLast()) + [values.shape[values.dim - 1]], seed: 4)
        ops.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: 2, accumulating: &gradients)
        return [ops.scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: 2), gradients.queries, gradients.keys, gradients.values]
    }

    static let attention: [KernelCase] = [
        KernelCase("scaledDotProductAttention with a mask per batch") { ops, accumulation in
            let mask = DoubleTensor([0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1], shape: [2, 1, 1, 6])
            return attention(
                ops, accumulation,
                queries: uniform([2, 3, 5, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2, requiresGradient: true), values: uniform([2, 3, 6, 3], seed: 3, requiresGradient: true), mask: mask,
            )
        },
        KernelCase("scaledDotProductAttention with a causal mask") { ops, accumulation in
            let mask = DoubleTensor((0 ..< 36).map { $0 % 6 > $0 / 6 ? 1 : 0 }, shape: [6, 6])
            return attention(
                ops, accumulation,
                queries: uniform([2, 3, 6, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2), values: uniform([2, 3, 6, 3], seed: 3, requiresGradient: true), mask: mask,
            )
        },
        KernelCase("scaledDotProductAttention without mask") { ops, accumulation in
            attention(
                ops, accumulation,
                queries: uniform([2, 3, 5, 4], seed: 1, requiresGradient: true), keys: uniform([2, 3, 6, 4], seed: 2, requiresGradient: true), values: uniform([2, 3, 6, 3], seed: 3), mask: nil,
            )
        },
    ]

    static let optimizers: [KernelCase] = [false, true].map { usesMaximum in
        KernelCase(usesMaximum ? "adamUpdate with AMSGrad" : "adamUpdate") { ops, _ in
            var (firstMoment, secondMoment) = (uniform([7, 1000], seed: 3), uniform([7, 1000], min: 0, max: 1, seed: 4))
            var maximum: DoubleTensor? = usesMaximum ? uniform([7, 1000], min: 0, max: 1, seed: 5) : nil
            let parameter = ops.adamUpdate(
                parameter: uniform([7, 1000], seed: 1), gradient: uniform([7, 1000], seed: 2),
                firstMoment: &firstMoment, secondMoment: &secondMoment, secondMomentMax: &maximum,
                learningRate: 0.01, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, beta1Power: 0.9 * 0.9, beta2Power: 0.999 * 0.999,
            )
            return [parameter, firstMoment, secondMoment, maximum]
        }
    }

    static let recurrent: [KernelCase] = [[true, true, true, true, true, true, true], [true, true, true, false, true, true, true], [false, false, false, true, false, false, false]].map { computes in
        KernelCase("gatedRecurrentUnitStep computing \(computes)") { ops, accumulation in
            let sources = [[5, 16], [5, 16], [5, 16], [5, 16], [16, 16], [16, 16], [16, 16]].enumerated().map { index, shape in
                uniform(shape, min: -1, max: 1, seed: UInt64(index + 1), requiresGradient: computes[index])
            }
            var gradients = GatedRecurrentUnitGradients(inSourceOrder: sources.map { accumulation.start(for: $0) })
            ops.gatedRecurrentUnitStepBackward(
                updateInput: sources[0], resetInput: sources[1], candidateInput: sources[2], state: sources[3],
                updateWeights: sources[4], resetWeights: sources[5], candidateWeights: sources[6], outputGradient: uniform([5, 16], seed: 9),
                accumulating: &gradients,
            )
            let step = ops.gatedRecurrentUnitStep(
                updateInput: sources[0], resetInput: sources[1], candidateInput: sources[2], state: sources[3],
                updateWeights: sources[4], resetWeights: sources[5], candidateWeights: sources[6],
            )
            return [step] + gradients.inSourceOrder
        }
    }

    static let all = activations + softmax + normalization + convolution + attention + optimizers + recurrent
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
        let (output, mask) = CPUFusedOperations.dropout(input: input, rate: 0.3)
        #expect(mask.elements.allSatisfy { $0 == 0 || $0 == 1 })
        #expect(output == input * mask)
        let keptFraction = mask.elements.reduce(0, +) / 100_000
        #expect(abs(keptFraction - 0.7) < 0.01, "kept fraction \(keptFraction)")

        #expect(CPUFusedOperations.dropout(input: input, rate: 0).mask.elements.allSatisfy { $0 == 1 })
        #expect(CPUFusedOperations.dropout(input: input, rate: 1).mask.elements.allSatisfy { $0 == 0 })
    }
}

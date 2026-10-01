//
//  GPUFusedOperationTests.swift
//  DL4STests
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
@testable import DL4S
import Foundation
import Testing

/// An activation with its parameter, applied with the public tensor operations.
enum GPUActivation: String, CaseIterable, CustomTestStringConvertible, Sendable {
    case tanh
    case relu
    case sigmoid
    case leakyRelu
    case gelu
    case swishScalar
    case swishPerChannel
    case swishBroadcast
    case mish
    case lisht
    case elu
    case softplus
    case squareplus

    var testDescription: String {
        rawValue
    }

    /// Shape of the parameter, or nil for activations without a parameter.
    var parameterShape: [Int]? {
        switch self {
        case .leakyRelu, .swishScalar, .elu: []
        case .swishPerChannel: [300]
        case .swishBroadcast: [7, 1]
        default: nil
        }
    }

    func apply<D: DeviceType>(_ x: Tensor<Float, D>, parameter: Tensor<Float, D>?) -> Tensor<Float, D> {
        switch self {
        case .tanh: x.tanh()
        case .relu: x.rectifiedLinear()
        case .sigmoid: x.sigmoid()
        case .leakyRelu: x.leakyRectifiedLinear(leakage: parameter!)
        case .gelu: x.gaussianErrorLinear()
        case .swishScalar, .swishPerChannel, .swishBroadcast: x.swishActivated(beta: parameter!)
        case .mish: x.mishActivated()
        case .lisht: x.lishtActivated()
        case .elu: x.exponentialLinearActivated(alpha: parameter!)
        case .softplus: x.softplus()
        case .squareplus: x.squareplus()
        }
    }
}

/// Shapes and the mask of a case of scaled dot product attention.
struct GPUAttentionCase: CustomTestStringConvertible, Sendable {
    enum Mask: String, Sendable {
        case none
        /// A causal mask of the shape [queryCount, keyCount]
        case causal
        /// A causal mask in which query 5 attends to no key
        case causalWithMaskedRow
        /// A mask of the keys of every batch, shape [batchSize, 1, 1, keyCount]
        case keys
        /// A mask of every score, shape [batchSize, heads, queryCount, keyCount]
        case scores
    }

    let batchSize: Int
    let heads: Int
    let keyHeads: Int
    /// Batch size of the keys and the values: the batch size of the queries, or 1
    let keyBatchSize: Int
    let queryCount: Int
    let keyCount: Int
    let keyDim: Int
    let valueDim: Int
    let mask: Mask
    /// Whether the queries, the keys, and the values require a gradient
    let gradients: [Bool]

    init(batchSize: Int, heads: Int, keyHeads: Int, keyBatchSize: Int? = nil, queryCount: Int, keyCount: Int, keyDim: Int, valueDim: Int, mask: Mask, gradients: [Bool] = [true, true, true]) {
        (self.batchSize, self.heads, self.keyHeads, self.keyBatchSize) = (batchSize, heads, keyHeads, keyBatchSize ?? batchSize)
        (self.queryCount, self.keyCount, self.keyDim, self.valueDim, self.mask, self.gradients) = (queryCount, keyCount, keyDim, valueDim, mask, gradients)
    }

    var testDescription: String {
        let gradientNames = zip(["q", "k", "v"], gradients).filter(\.1).map(\.0).joined()
        return "\(batchSize)x\(heads)/\(keyBatchSize)x\(keyHeads)x\(queryCount)x\(keyCount), sizes \(keyDim)/\(valueDim), mask \(mask.rawValue), gradients \(gradientNames)"
    }

    func maskTensor() -> Tensor<Float, CPU>? {
        switch mask {
        case .none:
            return nil
        case .causal:
            return Tensor((0 ..< queryCount * keyCount).map { $0 % keyCount > $0 / keyCount ? 1 : 0 }, shape: [queryCount, keyCount])
        case .causalWithMaskedRow:
            return Tensor((0 ..< queryCount * keyCount).map { $0 % keyCount > $0 / keyCount || $0 / keyCount == 5 ? 1 : 0 }, shape: [queryCount, keyCount])
        case .keys:
            return Tensor((0 ..< batchSize * keyCount).map { $0 % 7 == 3 ? 1 : 0 }, shape: [batchSize, 1, 1, keyCount])
        case .scores:
            let count = batchSize * heads * queryCount * keyCount
            return Tensor((0 ..< count).map { ($0 * 7919) % 10 < 3 ? 1 : 0 }, shape: [batchSize, heads, queryCount, keyCount])
        }
    }
}

extension GPUTests {
    @Suite(.serialized)
    struct GPUFusedOperationTests {
        /// Applies `forward` twice to the same input, so that the backward pass adds a gradient to an accumulated gradient,
        /// and returns the result and the gradients of the sources.
        private static func resultAndGradients<D: DeviceType>(_ sources: [Tensor<Float, D>], _ forward: ([Tensor<Float, D>]) -> Tensor<Float, D>) -> [Tensor<Float, D>] {
            let result = forward(sources)
            let weights = Tensor<Float, D>(GPUTest.random(result.shape, seed: 99, min: 0.5, max: 1.5))
            let loss = (result * weights).reduceSum() + (forward(sources) * 0.5).reduceSum()
            return [result] + loss.gradients(of: sources.filter(\.requiresGradient))
        }

        @Test(arguments: GPUActivation.allCases, [[5, 7, 300], [3, 7]])
        func activationsMatchCPU(activation: GPUActivation, shape: [Int]) {
            let x = GPUTest.random(shape, seed: 1, min: -3, max: 3, requiresGradient: true)
            guard let parameterShape = activation.parameterShape else {
                GPUTest.compare("\(activation) \(shape)") { gpu in
                    GPUTest.run(on: gpu, [x], cpu: { Self.resultAndGradients($0) { activation.apply($0[0], parameter: nil) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { activation.apply($0[0], parameter: nil) }) })
                }
                return
            }
            guard parameterShape.isEmpty || Array(shape.suffix(parameterShape.count)) == parameterShape || parameterShape == [7, 1] && shape.count == 3 else {
                return
            }
            // With a parameter that requires a gradient, the default implementation computes the gradients. Without, the kernel does.
            for parameterRequiresGradient in [false, true] {
                let parameter = GPUTest.random(parameterShape, seed: 2, min: 0.2, max: 1.5, requiresGradient: parameterRequiresGradient)
                GPUTest.compare("\(activation) \(shape) parameter gradient \(parameterRequiresGradient)") { gpu in
                    GPUTest.run(on: gpu, [x, parameter], cpu: { Self.resultAndGradients($0) { activation.apply($0[0], parameter: $0[1]) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { activation.apply($0[0], parameter: $0[1]) }) })
                }
            }
        }

        @Test(arguments: [([300, 1000], 1), ([700, 10], 1), ([4, 3, 5], 0), ([2, 3, 4], 2), ([37, 3000], 1)])
        func softmaxMatchesCPU(shape: [Int], axis: Int) {
            let x = GPUTest.random(shape, seed: 3, min: -4, max: 4, requiresGradient: true)
            GPUTest.compare("softmax \(shape) \(axis)") { gpu in
                GPUTest.run(on: gpu, [x], cpu: { Self.resultAndGradients($0) { $0[0].softmax(axis: axis) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { $0[0].softmax(axis: axis) }) })
            }
            GPUTest.compare("logSoftmax \(shape) \(axis)") { gpu in
                GPUTest.run(on: gpu, [x], cpu: { Self.resultAndGradients($0) { $0[0].logSoftmax(axis: axis) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { $0[0].logSoftmax(axis: axis) }) })
            }
        }

        @Test(arguments: [([64, 512], [512]), ([6, 3, 100], [3, 100]), ([5, 20], [20]), ([3, 2000], [2000])])
        func layerNormalizationMatchesCPU(shape: [Int], scaleShape: [Int]) {
            let x = GPUTest.random(shape, seed: 4, min: -2, max: 3, requiresGradient: true)
            let scale = GPUTest.random(scaleShape, seed: 5, min: 0.5, max: 1.5, requiresGradient: true)
            let shift = GPUTest.random(scaleShape, seed: 6, requiresGradient: true)
            GPUTest.compare("layer normalization \(shape)", tolerance: 2e-3) { gpu in
                GPUTest.run(on: gpu, [x, scale, shift], cpu: { Self.resultAndGradients($0) { $0[0].layerNormalized(scale: $0[1], shift: $0[2]) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { $0[0].layerNormalized(scale: $0[1], shift: $0[2]) }) })
            }
        }

        // The inputs are mostly negative, so that windows at the borders take their maximum from the zeros of the padding.
        @Test(arguments: [([2, 3, 7, 7], 3, 1, 2), ([2, 3, 8, 8], 2, 0, 2), ([2, 3, 9, 7], 2, 0, 2), ([1, 2, 5, 6], 3, 0, 1), ([2, 2, 6, 6], 3, 1, 3), ([4, 16, 33, 40], 3, 1, 2)])
        func poolingMatchesCPU(shape: [Int], windowSize: Int, padding: Int, stride: Int) {
            let x = GPUTest.random(shape, seed: 17, min: -3, max: 1, requiresGradient: true)
            GPUTest.compare("max pooling \(shape) window \(windowSize) padding \(padding) stride \(stride)") { gpu in
                GPUTest.run(on: gpu, [x], cpu: { Self.resultAndGradients($0) { $0[0].maxPooled2d(windowSize: windowSize, padding: padding, stride: stride) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { $0[0].maxPooled2d(windowSize: windowSize, padding: padding, stride: stride) }) })
            }
            GPUTest.compare("average pooling \(shape) window \(windowSize) padding \(padding) stride \(stride)") { gpu in
                GPUTest.run(on: gpu, [x], cpu: { Self.resultAndGradients($0) { $0[0].averagePooled2d(windowSize: windowSize, padding: padding, stride: stride) } }, gpu: { GPUTest.host(Self.resultAndGradients($0) { $0[0].averagePooled2d(windowSize: windowSize, padding: padding, stride: stride) }) })
            }
        }

        @Test(arguments: [([32, 8, 4, 4], [8, 1, 1]), ([32, 8, 4, 4], [8, 4, 4]), ([256, 40], [40]), ([5, 3], [3])])
        func batchNormalizationMatchesCPU(shape: [Int], scaleShape: [Int]) {
            let x = GPUTest.random(shape, seed: 12, min: -2, max: 3, requiresGradient: true)
            let scale = GPUTest.random(scaleShape, seed: 13, min: 0.5, max: 1.5, requiresGradient: true)
            let shift = GPUTest.random(scaleShape, seed: 14, requiresGradient: true)
            let columnShape = Array(shape.dropFirst())
            let mean = GPUTest.random(columnShape, seed: 15)
            let variance = GPUTest.random(columnShape, seed: 16, min: 0.5, max: 2)
            func body<D: DeviceType>(_ values: [Tensor<Float, D>]) -> [Tensor<Float, D>] {
                let statistics = values[0].batchNormalized(scale: values[1], shift: values[2])
                let batch = Self.resultAndGradients(Array(values[0 ..< 3])) { $0[0].batchNormalized(scale: $0[1], shift: $0[2]).output }
                let fixed = Self.resultAndGradients(Array(values[0 ..< 3])) { $0[0].batchNormalized(scale: $0[1], shift: $0[2], mean: values[3], variance: values[4]) }
                return [statistics.mean, statistics.variance] + batch + fixed
            }
            GPUTest.compare("batch normalization \(shape) \(scaleShape)", tolerance: 2e-3) { gpu in
                GPUTest.run(on: gpu, [x, scale, shift, mean, variance], cpu: { body($0) }, gpu: { GPUTest.host(body($0)) })
            }
        }

        @Test(arguments: [
            GPUAttentionCase(batchSize: 2, heads: 4, keyHeads: 4, queryCount: 77, keyCount: 100, keyDim: 64, valueDim: 64, mask: .none),
            GPUAttentionCase(batchSize: 2, heads: 4, keyHeads: 2, queryCount: 64, keyCount: 64, keyDim: 64, valueDim: 64, mask: .causal),
            GPUAttentionCase(batchSize: 3, heads: 6, keyHeads: 3, queryCount: 200, keyCount: 300, keyDim: 64, valueDim: 64, mask: .causal),
            GPUAttentionCase(batchSize: 1, heads: 8, keyHeads: 1, queryCount: 130, keyCount: 33, keyDim: 32, valueDim: 32, mask: .keys),
            GPUAttentionCase(batchSize: 2, heads: 2, keyHeads: 2, queryCount: 100, keyCount: 150, keyDim: 32, valueDim: 32, mask: .causalWithMaskedRow),
            GPUAttentionCase(batchSize: 1, heads: 4, keyHeads: 2, queryCount: 70, keyCount: 90, keyDim: 128, valueDim: 128, mask: .causal),
            // Few key heads and many queries split the queries of a block of keys between threadgroups.
            GPUAttentionCase(batchSize: 1, heads: 8, keyHeads: 2, queryCount: 520, keyCount: 100, keyDim: 64, valueDim: 64, mask: .keys),
            // Keys and values that broadcast along the batch serve the queries of every batch.
            GPUAttentionCase(batchSize: 3, heads: 4, keyHeads: 2, keyBatchSize: 1, queryCount: 100, keyCount: 80, keyDim: 64, valueDim: 64, mask: .causal),
            GPUAttentionCase(batchSize: 2, heads: 4, keyHeads: 2, queryCount: 70, keyCount: 90, keyDim: 64, valueDim: 64, mask: .keys, gradients: [false, true, true]),
            GPUAttentionCase(batchSize: 2, heads: 4, keyHeads: 2, queryCount: 70, keyCount: 90, keyDim: 64, valueDim: 64, mask: .none, gradients: [true, false, false]),
            GPUAttentionCase(batchSize: 2, heads: 4, keyHeads: 4, queryCount: 70, keyCount: 90, keyDim: 32, valueDim: 32, mask: .causal, gradients: [false, false, true]),
            GPUAttentionCase(batchSize: 2, heads: 2, keyHeads: 2, queryCount: 40, keyCount: 70, keyDim: 128, valueDim: 128, mask: .none, gradients: [true, false, true]),
            GPUAttentionCase(batchSize: 2, heads: 2, keyHeads: 2, queryCount: 40, keyCount: 70, keyDim: 128, valueDim: 128, mask: .scores),
            GPUAttentionCase(batchSize: 2, heads: 2, keyHeads: 1, queryCount: 9, keyCount: 17, keyDim: 16, valueDim: 24, mask: .keys),
            GPUAttentionCase(batchSize: 1, heads: 2, keyHeads: 2, queryCount: 5, keyCount: 3, keyDim: 12, valueDim: 5, mask: .none),
        ])
        func attentionMatchesCPU(_ attention: GPUAttentionCase) {
            Self.compareAttention(attention)
        }

        // With the head size 128, the backward kernels run for more than 2^26 scores.
        // The operations run only on the GPU: with the host limit of the mixed mode, the host would compute the 2^26 scores again.
        @Test(.releaseBuild)
        func largeAttentionMatchesCPU() {
            Self.compareAttention(GPUAttentionCase(batchSize: 1, heads: 16, keyHeads: 4, queryCount: 2048, keyCount: 2049, keyDim: 128, valueDim: 128, mask: .keys), modes: [.gpu])
        }

        private static func compareAttention(_ attention: GPUAttentionCase, modes: [GPUPlacementMode] = GPUPlacementMode.allCases) {
            let (batchSize, keyBatchSize, heads, keyHeads) = (attention.batchSize, attention.keyBatchSize, attention.heads, attention.keyHeads)
            let queries = GPUTest.random([batchSize, heads, attention.queryCount, attention.keyDim], seed: 30, min: -2, max: 2, requiresGradient: attention.gradients[0])
            let keys = GPUTest.random([keyBatchSize, keyHeads, attention.keyCount, attention.keyDim], seed: 31, min: -2, max: 2, requiresGradient: attention.gradients[1])
            let values = GPUTest.random([keyBatchSize, keyHeads, attention.keyCount, attention.valueDim], seed: 32, requiresGradient: attention.gradients[2])
            let mask = attention.maskTensor()
            let temperature = Float(attention.keyDim).squareRoot()
            func body<D: DeviceType>(_ sources: [Tensor<Float, D>]) -> [Tensor<Float, D>] {
                let mask = mask.map { Tensor<Float, D>($0) }
                return resultAndGradients(sources) { scaledDotProductAttention(queries: $0[0], keys: $0[1], values: $0[2], mask: mask, temperature: temperature) }
            }
            GPUTest.compare("attention \(attention.testDescription)", modes: modes) { gpu in
                GPUTest.run(on: gpu, [queries, keys, values], cpu: { body($0) }, gpu: { GPUTest.host(body($0)) })
            }
        }

        @Test(arguments: [4, 2, 1])
        func multiHeadAttentionMatchesCPU(keyHeads: Int) {
            Self.compareMultiHeadAttention(keyHeads: keyHeads, gradients: Array(repeating: true, count: 7))
        }

        // The gradients of the sources in the order of the operation: queries, keys, values, and the four weights.
        @Test(arguments: [[false, false, false, true, true, true, true], [true, false, false, false, false, false, true], [false, true, true, false, false, false, false]])
        func multiHeadAttentionGradientSubsetsMatchCPU(gradients: [Bool]) {
            Self.compareMultiHeadAttention(keyHeads: 2, gradients: gradients)
        }

        private static func compareMultiHeadAttention(keyHeads: Int, gradients: [Bool]) {
            let (heads, hidden, keyDim) = (4, 64, 32)
            let inputs = [[2, 50, hidden], [2, 70, hidden], [2, 70, hidden]].enumerated().map { index, shape in
                GPUTest.random(shape, seed: UInt64(40 + index), requiresGradient: gradients[index])
            }
            let weights = [[hidden, heads * keyDim], [hidden, keyHeads * keyDim], [hidden, keyHeads * keyDim], [heads * keyDim, hidden]].enumerated().map { index, shape in
                GPUTest.random(shape, seed: UInt64(43 + index), min: -0.2, max: 0.2, requiresGradient: gradients[3 + index])
            }
            let mask = Tensor<Float, CPU>((0 ..< 2 * 70).map { $0 % 9 == 4 ? 1 : 0 }, shape: [2, 1, 1, 70])
            func body<D: DeviceType>(_ sources: [Tensor<Float, D>]) -> [Tensor<Float, D>] {
                let mask = Tensor<Float, D>(mask)
                return Self.resultAndGradients(sources) {
                    multiHeadAttention(
                        queries: $0[0], keys: $0[1], values: $0[2], mask: mask,
                        queryWeights: $0[3], keyWeights: $0[4], valueWeights: $0[5], outputWeights: $0[6], heads: heads, temperature: Float(keyDim).squareRoot(),
                    )
                }
            }
            GPUTest.compare("multi-head attention with \(keyHeads) key heads, gradients \(gradients)") { gpu in
                GPUTest.run(on: gpu, inputs + weights, cpu: { body($0) }, gpu: { GPUTest.host(body($0)) })
            }
        }

        @Test func dropoutKeepsTheExpectedFraction() {
            GPUTest.withHostExecutionLimit(0) {
                checkDropout()
            }
        }

        private func checkDropout() {
            let x = Tensor<Float, GPU>(GPUTest.random([256, 1024], seed: 7, min: 1, max: 2), requiresGradient: true)
            let (output, mask) = dropout(x, rate: 0.3)
            let (values, maskValues, outputValues) = (x.elements, mask.elements, output.elements)
            let kept = Float(maskValues.reduce(0, +)) / Float(maskValues.count)
            #expect(Swift.abs(kept - 0.7) < 0.01, "kept fraction \(kept)")
            #expect(maskValues.allSatisfy { $0 == 0 || $0 == 1 })
            #expect(zip(zip(values, maskValues), outputValues).allSatisfy { $0.0 * $0.1 == $1 })
            var gradient = Tensor<Float, GPU>(uninitializedShape: x.shape)
            withExtendedLifetime((mask, x)) {
                GPU.FusedOperations.dropoutBackward(mask: mask.values, outputGradient: x.values, inputGradient: GradientBuffer(values: gradient.mutableValues, adds: false))
            }
            #expect(gradient.elements == outputValues)
            // Two calls use different random numbers.
            #expect(dropout(x, rate: 0.3).mask.elements != maskValues)
        }

        @Test func dropoutWithoutDroppedElementsKeepsTheInput() {
            GPUTest.withHostExecutionLimit(0) {
                let x = Tensor<Float, GPU>(GPUTest.random([64, 100], seed: 8))
                let (output, mask) = dropout(x, rate: 0)
                #expect(output.elements == x.elements)
                #expect(mask.elements.allSatisfy { $0 == 1 })
                // Double has no GPU kernels, so it uses the default implementation.
                let doubles = Tensor<Double, GPU>(Tensor<Double, CPU>(x.elements.map { Double($0) }, shape: x.shape))
                #expect(doubles.droppedOut(rate: 0).elements == doubles.elements)
            }
        }

        /// The result and the mask of the dropout kernel of the GPU.
        private func dropout(_ input: Tensor<Float, GPU>, rate: Float) -> (output: Tensor<Float, GPU>, mask: Tensor<Float, GPU>) {
            var (output, mask) = (Tensor<Float, GPU>(uninitializedShape: input.shape), Tensor<Float, GPU>(uninitializedShape: input.shape))
            withExtendedLifetime(input) {
                GPU.FusedOperations.dropout(input: input.values, rate: rate, result: output.mutableValues, mask: mask.mutableValues)
            }
            return (output, mask)
        }

        @Test(arguments: [false, true])
        func adamStepsMatchCPU(useAMSGrad: Bool) {
            let parameter = GPUTest.random([300, 70], seed: 8, requiresGradient: true)
            let gradients = (0 ..< 4).map { GPUTest.random([300, 70], seed: 20 + UInt64($0)) }
            GPUTest.compare("adam amsgrad \(useAMSGrad)") { gpu in
                if gpu {
                    var optimizer = Adam<Float, GPU>(learningRate: 0.01, useAMSGrad: useAMSGrad)
                    var parameters = [Tensor<Float, GPU>(parameter, requiresGradient: true)]
                    for gradient in gradients {
                        optimizer.update(&parameters, along: [Tensor<Float, GPU>(gradient)])
                    }
                    return GPUTest.host(parameters)
                }
                var optimizer = Adam<Float, CPU>(learningRate: 0.01, useAMSGrad: useAMSGrad)
                var parameters = [parameter]
                for gradient in gradients {
                    optimizer.update(&parameters, along: [gradient])
                }
                return parameters
            }
        }

        @Test(arguments: [(28, 16, 28, 64), (5, 1, 30, 200)])
        func gatedRecurrentUnitsMatchCPU(steps: Int, batch: Int, inputSize: Int, hiddenSize: Int) {
            var generator = WyHash(seed: 9)
            let cpuLayer = GRU<Float, CPU>(inputSize: inputSize, hiddenSize: hiddenSize, direction: .forward, using: &generator)
            let input = GPUTest.random([steps, batch, inputSize], seed: 10, requiresGradient: true)
            func body<D: DeviceType>(_ layer: GRU<Float, D>, _ input: Tensor<Float, D>) -> [Tensor<Float, D>] {
                let (states, final) = layer(input)
                let loss = (states * states).reduceSum() + final().reduceSum()
                return [states] + loss.gradients(of: [input] + layer.parameters)
            }
            GPUTest.compare("gru \(steps)x\(batch)x\(inputSize) h\(hiddenSize)", tolerance: 2e-3) { gpu in
                if gpu {
                    var layer = GRU<Float, GPU>(inputSize: inputSize, hiddenSize: hiddenSize, direction: .forward)
                    let weights = cpuLayer.parameters
                    layer.update { parameters in
                        parameters = weights.map { Tensor<Float, GPU>($0, requiresGradient: true) }
                    }
                    return GPUTest.host(body(layer, Tensor<Float, GPU>(input, requiresGradient: true)))
                }
                return body(cpuLayer, input)
            }
        }

        @Test func trainingStepsDoNotWaitForTheGPU() throws {
            try GPUTest.withHostExecutionLimit(GPU.defaultHostExecutionLimit) {
                try checkTrainingSteps()
            }
        }

        private func checkTrainingSteps() throws {
            var model = Sequential {
                Dense<Float, GPU>(inputSize: 100, outputSize: 256)
                Relu<Float, GPU>()
                Dense<Float, GPU>(inputSize: 256, outputSize: 10)
                LogSoftmax<Float, GPU>()
            }
            var optimizer = Adam<Float, GPU>(learningRate: 0.001)
            let input = Tensor<Float, GPU>(GPUTest.random([64, 100], seed: 11))
            let labels = Tensor<Int32, GPU>((0 ..< 64).map { Int32($0 % 10) })
            var losses: [Tensor<Float, GPU>] = []
            let before = GPUContext.current.statistics
            for _ in 0 ..< 5 {
                let loss = categoricalNegativeLogLikelihood(expected: labels, actual: model(input))
                model.update { parameters in
                    optimizer.update(&parameters, along: loss.gradients(of: parameters))
                }
                losses.append(loss)
            }
            #expect(GPUContext.current.statistics.waits == before.waits, "The training steps waited for the GPU.")
            #expect(try #require(losses.last?.item) < losses.first!.item)
        }
    }
}
#endif

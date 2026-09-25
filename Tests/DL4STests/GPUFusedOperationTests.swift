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

        @Test func dropoutKeepsTheExpectedFraction() {
            GPU.hostExecutionLimit = 0
            defer { GPU.hostExecutionLimit = 4096 }
            let x = Tensor<Float, GPU>(GPUTest.random([256, 1024], seed: 7, min: 1, max: 2), requiresGradient: true)
            let (output, mask) = GPU.FusedOperations.dropout(input: x, rate: 0.3)
            let (values, maskValues, outputValues) = (x.elements, mask.elements, output.elements)
            let kept = Float(maskValues.reduce(0, +)) / Float(maskValues.count)
            #expect(Swift.abs(kept - 0.7) < 0.01, "kept fraction \(kept)")
            #expect(maskValues.allSatisfy { $0 == 0 || $0 == 1 })
            #expect(zip(zip(values, maskValues), outputValues).allSatisfy { $0.0 * $0.1 == $1 })
            var gradient: Tensor<Float, GPU>?
            GPU.FusedOperations.dropoutBackward(mask: mask, outputGradient: x, accumulating: &gradient)
            #expect(gradient?.elements == outputValues)
            // Two calls use different random numbers.
            #expect(GPU.FusedOperations.dropout(input: x, rate: 0.3).mask.elements != maskValues)
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
            GPU.hostExecutionLimit = 4096
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

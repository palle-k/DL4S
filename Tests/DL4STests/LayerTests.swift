//
//  LayerTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.09.26.
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
import Foundation
import Testing

/// A layer with every kind of stored property that the `@Layer` macro handles.
@Layer
struct ProbeLayer {
    typealias Element = Float
    typealias Device = CPU

    var weights: Tensor<Float, CPU>
    @Frozen var mask: Tensor<Float, CPU>
    var optionalBias: Tensor<Float, CPU>?
    var scales: [Tensor<Float, CPU>]
    var inner: Dense<Float, CPU>
    var optionalInner: Dense<Float, CPU>?
    var blocks: [Dense<Float, CPU>]
    var block: Sequential<Dense<Float, CPU>, Relu<Float, CPU>>
    var rate: Float = 0.5
    let constant: Tensor<Float, CPU>

    init(withOptionals: Bool) {
        weights = Tensor(repeating: 1, shape: [2, 2], requiresGradient: true)
        mask = Tensor(repeating: 1, shape: [2])
        optionalBias = withOptionals ? Tensor(repeating: 0, shape: [2], requiresGradient: true) : nil
        scales = [Tensor(repeating: 1, shape: [2], requiresGradient: true), Tensor(repeating: 2, shape: [2], requiresGradient: true)]
        inner = Dense(inputSize: 2, outputSize: 2)
        optionalInner = withOptionals ? Dense(inputSize: 2, outputSize: 2) : nil
        blocks = [Dense(inputSize: 2, outputSize: 2), Dense(inputSize: 2, outputSize: 2)]
        block = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 2)
            Relu<Float, CPU>()
        }
        constant = Tensor(repeating: 3, shape: [2])
    }

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        var x = inputs.matrixMultiplied(with: weights) * mask
        if let optionalBias {
            x += optionalBias
        }
        x = x * scales[0] * scales[1]
        x = inner(x)
        if let optionalInner {
            x = optionalInner(x)
        }
        for layer in blocks {
            x = layer(x)
        }
        return block(x) + constant
    }
}

/// A layer whose backbone is saved but not trained.
@Layer
struct FrozenBackbone {
    typealias Element = Float
    typealias Device = CPU

    @Frozen var backbone: Dense<Float, CPU>
    var head: Dense<Float, CPU>

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        head(backbone(inputs))
    }
}

/// A reference-typed layer.
@Layer
final class ClassLayer {
    typealias Element = Float
    typealias Device = CPU

    var inner: Dense<Float, CPU>
    var scale: Tensor<Float, CPU>

    init() {
        inner = Dense(inputSize: 2, outputSize: 2)
        scale = Tensor(repeating: 2, shape: [2], requiresGradient: true)
    }

    func callAsFunction(_ inputs: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        inner(inputs) * scale
    }
}

struct LayerTests {
    private typealias TensorLayer = any LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU>

    /// Collects the path and role of every tensor that a layer reports.
    private func roles<Layer: LayerType>(of layer: Layer) -> [(path: String, role: TensorRole)] where Layer.Parameter == Float, Layer.Device == CPU {
        var roles: [(path: String, role: TensorRole)] = []
        var copy = layer
        var visitor = TensorVisitor<Float, CPU>(tensors: { _, role, path in
            roles.append((path.description, role))
        })
        copy.visitTensors(&visitor)
        return roles
    }

    @Test func testMacroReportsStoredPropertiesByType() {
        let layer = ProbeLayer(withOptionals: true)

        #expect(layer.weightPaths.map(\.description) == [
            "weights",
            "optionalBias",
            "scales.0", "scales.1",
            "inner.weights", "inner.bias",
            "optionalInner.weights", "optionalInner.bias",
            "blocks.0.weights", "blocks.0.bias",
            "blocks.1.weights", "blocks.1.bias",
            "block.0.weights", "block.0.bias",
        ])
        #expect(layer.parameters.count == layer.weightPaths.count)

        let reported = roles(of: layer)
        #expect(reported.count == layer.weightPaths.count + 1)
        #expect(reported.first { $0.path == "mask" }?.role == .frozen)
        #expect(reported.allSatisfy { $0.path == "mask" || $0.role == .weight })
        #expect(!reported.contains { $0.path == "constant" || $0.path == "rate" })
    }

    @Test func testMacroSkipsNilOptionals() {
        let layer = ProbeLayer(withOptionals: false)

        #expect(layer.weightPaths.map(\.description) == [
            "weights",
            "scales.0", "scales.1",
            "inner.weights", "inner.bias",
            "blocks.0.weights", "blocks.0.bias",
            "blocks.1.weights", "blocks.1.bias",
            "block.0.weights", "block.0.bias",
        ])
    }

    @Test func testUpdateWritesTensorsBackInOrder() {
        var layer = ProbeLayer(withOptionals: true)
        let before = layer.parameters

        layer.update { parameters in
            #expect(parameters.count == before.count)
            for index in parameters.indices {
                parameters[index] += Tensor(Float(index + 1))
            }
        }

        let after = layer.parameters
        #expect(after.count == before.count)
        for index in before.indices {
            #expect(after[index] == before[index] + Tensor(Float(index + 1)), "tensor \(index)")
            #expect(after[index].requiresGradient)
        }
        #expect(layer.weights == before[0] + 1)
        #expect(layer.optionalBias == before[1] + 2)
        #expect(layer.inner.weights == before[4] + 5)
        #expect(layer.block.first.bias == before[13] + 14)
        #expect(layer.mask == Tensor(repeating: 1, shape: [2]))
        #expect(layer.constant == Tensor(repeating: 3, shape: [2]))
    }

    @Test func testUpdateThrowsThroughTheClosure() {
        struct Stop: Error {}
        var layer = Dense<Float, CPU>(inputSize: 2, outputSize: 2)
        let before = layer.weights

        #expect(throws: Stop.self) {
            try layer.update { parameters in
                parameters[0] += 1
                throw Stop()
            }
        }
        #expect(layer.weights == before)
    }

    @Test func testFreezeAndUnfreeze() {
        var layer = ProbeLayer(withOptionals: true)
        let count = layer.parameters.count

        layer.freeze()
        #expect(layer.parameters.isEmpty)
        #expect(!layer.weights.requiresGradient)
        #expect(!layer.mask.requiresGradient)

        layer.unfreeze()
        #expect(layer.parameters.count == count)
        #expect(layer.weights.requiresGradient)
        #expect(!layer.mask.requiresGradient, "frozen tensors stay frozen")

        layer.inner.freeze()
        #expect(layer.parameters.count == count - 2)
        #expect(layer.weightPaths.map(\.description).contains("inner.weights") == false)
    }

    @Test func testFrozenSublayerReportsAllTensorsAsFrozen() {
        let layer = FrozenBackbone(backbone: Dense(inputSize: 2, outputSize: 2), head: Dense(inputSize: 2, outputSize: 2))
        #expect(layer.weightPaths.map(\.description) == ["head.weights", "head.bias"])
        let reported = roles(of: layer)
        #expect(reported.filter { $0.role == .frozen }.map(\.path) == ["backbone.weights", "backbone.bias"])
    }

    @Test func testClassLayersAreTraversedAndUpdatedInPlace() {
        let layer = ClassLayer()
        #expect(layer.weightPaths.map(\.description) == ["inner.weights", "inner.bias", "scale"])

        var reference = layer
        reference.update { parameters in
            for index in parameters.indices {
                parameters[index] = Tensor(repeating: 0, shape: parameters[index].shape)
            }
        }

        #expect(layer.scale == Tensor(repeating: 0, shape: [2]), "the update reaches the shared instance")
        #expect(layer(Tensor(repeating: 1, shape: [1, 2])) == Tensor(repeating: 0, shape: [1, 2]))
    }

    @Test func testSwishDecidesRoleAtRunTime() {
        #expect(Swish<Float, CPU>(trainableWithChannels: 3).weightPaths.map(\.description) == ["beta"])
        #expect(Swish<Float, CPU>(fixedWithBeta: 2).parameters.isEmpty)
        #expect(roles(of: Swish<Float, CPU>(fixedWithBeta: 2)).map(\.role) == [.frozen])
    }

    @Test func testSequentialNumbersPositionsAndNestsBlocks() {
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 2)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 2)
            Sequential {
                Dense<Float, CPU>(inputSize: 2, outputSize: 2)
                Relu<Float, CPU>()
            }
            Dense<Float, CPU>(inputSize: 2, outputSize: 2)
        }

        #expect(model.weightPaths.map(\.description) == [
            "0.weights", "0.bias",
            "2.weights", "2.bias",
            "3.0.weights", "3.0.bias",
            "4.weights", "4.bias",
        ])
        #expect(model(Tensor(repeating: 1, shape: [1, 2])).shape == [1, 2])
    }

    @Test func testExistentialLayerRunsInferenceAndUpdatesIndependently() {
        let dense = Dense<Float, CPU>(inputSize: 4, outputSize: 3)
        var layer: TensorLayer = dense
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [2, 4])
        let expected = dense(input)

        #expect(layer(input) == expected)
        #expect(layer.parameters.count == dense.parameters.count)

        layer.update { parameters in
            for index in parameters.indices {
                parameters[index] = Tensor(repeating: 0, shape: parameters[index].shape)
            }
        }

        #expect(layer(input) == Tensor(repeating: 0, shape: [2, 3]))
        #expect(dense(input) == expected)
    }

    /// Builds a small classifier. The opaque return type hides the layer types of the block and keeps the tensor types.
    private func makeClassifier() -> some LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU> {
        Sequential {
            Dense<Float, CPU>(inputSize: 4, outputSize: 8)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 8, outputSize: 3)
            Softmax<Float, CPU>()
        }
    }

    @Test func testOpaqueLayerReturnTypeTrains() {
        var model = makeClassifier()
        var optimizer = SGD<Float, CPU>(learningRate: 0.1)
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let labels = Tensor<Int32, CPU>([0, 1, 2, 0, 1])
        let initialLoss = categoricalCrossEntropy(expected: labels, actual: model(input))

        for _ in 0 ..< 20 {
            let loss = categoricalCrossEntropy(expected: labels, actual: model(input))
            model.update { parameters in
                optimizer.update(&parameters, along: loss.gradients(of: parameters))
            }
        }

        let finalLoss = categoricalCrossEntropy(expected: labels, actual: model(input))
        #expect(finalLoss.item < initialLoss.item)
        #expect(model(input).shape == [5, 3])
    }

    /// The type of a block lists its layers in order.
    @Test func testSequentialTypeListsTheLayers() {
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 2)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 1)
            Sigmoid<Float, CPU>()
        }
        let typed: Sequential<Dense<Float, CPU>, Tanh<Float, CPU>, Dense<Float, CPU>, Sigmoid<Float, CPU>> = model
        #expect(typed.first.parameters.count == 2)
        #expect(typed.last.parameters.isEmpty)
        #expect(typed.parameters.count == 4)
    }

    /// The old fixed-arity overloads stopped at 16 layers.
    @Test func testSequentialBuilderAcceptsTwentyLayers() {
        var generator = WyHash(seed: 7)
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 4, using: &generator)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 2, using: &generator)
            Softmax<Float, CPU>()
        }
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4], using: &generator)
        let output = model(input)

        #expect(output.shape == [3, 2])
        #expect(model.parameters.count == 20)
        #expect(model.weightPaths.count == 20)
        #expect(model.weightPaths.last?.description == "18.bias")
    }

    /// A block with mixed value types between the layers.
    @Test func testSequentialPassesTuplesBetweenLayers() {
        let model = Sequential {
            GRU<Float, CPU>(inputSize: 3, hiddenSize: 4)
            Lambda<GRU<Float, CPU>.Outputs, Tensor<Float, CPU>, Float, CPU> { outputs in
                outputs.0
            }
            Dense<Float, CPU>(inputSize: 4, outputSize: 2)
        }
        let output = model(Tensor(repeating: 0.5, shape: [5, 2, 3]))
        #expect(output.shape == [2, 2])
        #expect(model.weightPaths.map(\.description).contains("2.weights"))
    }

    @Test func testSequentialCodableRoundTrip() throws {
        var generator = WyHash(seed: 3)
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 2, using: &generator)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 1, using: &generator)
        }
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 3], using: &generator)

        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(type(of: model), from: data)

        #expect(decoded(input) == model(input))
    }

    @Test func testModifyLayersReachesNestedLayersOfAType() {
        var model = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 2)
            Dropout<Float, CPU>(rate: 0.5)
            Sequential {
                Dense<Float, CPU>(inputSize: 2, outputSize: 2)
                Dropout<Float, CPU>(rate: 0.5)
            }
        }
        #expect(model.layers(of: Dropout<Float, CPU>.self).count == 2)
        #expect(model.layers(of: Dropout<Float, CPU>.self).allSatisfy { $0.isActive })

        model.modifyLayers(of: Dropout<Float, CPU>.self) { dropout in
            dropout.isActive = false
        }

        #expect(model.layers(of: Dropout<Float, CPU>.self).allSatisfy { !$0.isActive })
        #expect(!model.last.last.isActive)

        var alexNet = AlexNet<Float, CPU>(inputChannels: 1, classes: 2)
        #expect(alexNet.isDropoutActive)
        alexNet.isDropoutActive = false
        #expect(!alexNet.isDropoutActive)
    }

    @Test func testTensorPathParsesAndPrints() {
        let path = TensorPath("encoder.blocks.3.Wq")
        #expect(path.segments == [.name("encoder"), .name("blocks"), .index(3), .name("Wq")])
        #expect(path.description == "encoder.blocks.3.Wq")
        #expect(path.appending(.index(0)).description == "encoder.blocks.3.Wq.0")
        #expect(TensorPath("0.weights") == TensorPath([.index(0), .name("weights")]))
    }
}

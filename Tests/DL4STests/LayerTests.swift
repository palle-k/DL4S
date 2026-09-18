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
import Synchronization
import Testing

struct LayerTests {
    private typealias TensorLayer = any LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU>

    /// Returns a copy of the layer with all parameters set to zero.
    ///
    /// The generic parameter opens the existential, so the writable key paths of the concrete layer type can be used.
    private func zeroingParameters<Layer: LayerType>(of layer: Layer) -> Layer where Layer.Parameter == Float, Layer.Device == CPU {
        var copy = layer
        for path in layer.parameterPaths {
            copy[keyPath: path] = Tensor(repeating: 0, shape: layer[keyPath: path].shape)
        }
        return copy
    }

    @Test func testExistentialLayerRunsInferenceAndCopiesIndependently() {
        let dense = Dense<Float, CPU>(inputSize: 4, outputSize: 3)
        let layer: TensorLayer = dense
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [2, 4])
        let expected = dense(input)

        #expect(layer(input) == expected)
        #expect(layer.parameters.count == dense.parameters.count)

        let zeroed: TensorLayer = zeroingParameters(of: layer)

        #expect(zeroed(input) == Tensor(repeating: 0, shape: [2, 3]))
        #expect(layer(input) == expected)
        #expect(dense(input) == expected)
    }

    /// Builds a small classifier. The opaque return type hides the nesting of the block and keeps the tensor types.
    private func makeClassifier() -> some LayerType<Tensor<Float, CPU>, Tensor<Float, CPU>, Float, CPU> {
        Sequential {
            Dense<Float, CPU>(inputSize: 4, outputSize: 8)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 8, outputSize: 3)
            Softmax<Float, CPU>()
        }
    }

    @Test func testOpaqueLayerReturnTypeTrains() {
        var optimizer = SGD(model: makeClassifier(), learningRate: 0.1)
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let labels = Tensor<Int32, CPU>([0, 1, 2, 0, 1])
        let initialLoss = categoricalCrossEntropy(expected: labels, actual: optimizer.model(input))

        for _ in 0 ..< 20 {
            let loss = categoricalCrossEntropy(expected: labels, actual: optimizer.model(input))
            optimizer.update(along: loss.gradients(of: optimizer.model.parameters))
        }

        let finalLoss = categoricalCrossEntropy(expected: labels, actual: optimizer.model(input))
        #expect(finalLoss.item < initialLoss.item)
        #expect(optimizer.model(input).shape == [5, 3])
    }

    /// The builder nests from the left, so the type of a block follows one rule for any number of layers.
    @Test func testSequentialBuilderNestsFromTheLeft() {
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 2)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 1)
            Sigmoid<Float, CPU>()
        }
        let typed: Sequential<Sequential<Sequential<Dense<Float, CPU>, Tanh<Float, CPU>>, Dense<Float, CPU>>, Sigmoid<Float, CPU>> = model
        #expect(typed.first.first.first.parameters.count == 2)
        #expect(typed.first.second.parameters.count == 2)
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
        #expect(model.parameterPaths.count == 20)
    }

    /// The `Codable` form of a block follows the nesting of the builder, so a block must decode as the type it was encoded from.
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

    /// Parameter key paths of a composite layer are appended key paths. They must cross a `@Sendable` boundary
    /// and must still write to the right parameter.
    @Test func testAppendedParameterPathsAreSendable() {
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 2)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 2, outputSize: 1)
        }
        let paths = model.parameterPaths
        #expect(paths.count == 4)

        let zeroed: @Sendable () -> [Tensor<Float, CPU>] = {
            var copy = model
            for path in paths {
                copy[keyPath: path] = Tensor(repeating: 0, shape: copy[keyPath: path].shape)
            }
            return copy.parameters
        }

        let group = DispatchGroup()
        group.enter()
        let result = Mutex<[Tensor<Float, CPU>]>([])
        Thread {
            result.withLock { $0 = zeroed() }
            group.leave()
        }.start()
        group.wait()

        let parameters = result.withLock { $0 }
        #expect(parameters.map(\.shape) == model.parameters.map(\.shape))
        #expect(parameters.allSatisfy { $0.elements.allSatisfy { $0 == 0 } })
        #expect(!model.parameters[0].elements.allSatisfy { $0 == 0 })
    }
}

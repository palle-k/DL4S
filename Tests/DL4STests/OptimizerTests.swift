//
//  OptimizerTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 21.09.26.
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

struct OptimizerTests {
    private typealias Model = Sequential<Dense<Float, CPU>, Tanh<Float, CPU>, Dense<Float, CPU>>

    private func makeModel(seed: UInt64 = 1) -> Model {
        var generator = WyHash(seed: seed)
        return Sequential {
            Dense<Float, CPU>(inputSize: 3, outputSize: 4, using: &generator)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 4, outputSize: 1, using: &generator)
        }
    }

    private let inputs = Tensor<Float, CPU>([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    private let targets = Tensor<Float, CPU>([[1], [2], [3], [0]])

    @discardableResult
    private func step<Optim: Optimizer>(_ model: inout Model, with optimizer: inout Optim) -> Float where Optim.Element == Float, Optim.Device == CPU {
        let loss = meanSquaredError(expected: targets, actual: model(inputs))
        model.update { parameters in
            optimizer.update(&parameters, along: loss.gradients(of: parameters))
        }
        return loss.item
    }

    private func expectLossDecreases<Optim: Optimizer>(with optimizer: Optim, steps: Int = 50) where Optim.Element == Float, Optim.Device == CPU {
        var model = makeModel()
        var optimizer = optimizer
        let first = step(&model, with: &optimizer)
        var last = first
        for _ in 1 ..< steps {
            last = step(&model, with: &optimizer)
        }
        #expect(last < first, "\(Optim.self): loss went from \(first) to \(last)")
    }

    @Test func testEveryOptimizerReducesTheLoss() {
        expectLossDecreases(with: SGD<Float, CPU>(learningRate: 0.05))
        expectLossDecreases(with: Momentum<Float, CPU>(learningRate: 0.02))
        expectLossDecreases(with: Adam<Float, CPU>(learningRate: 0.02))
        expectLossDecreases(with: Adam<Float, CPU>(learningRate: 0.02, useAMSGrad: true))
        expectLossDecreases(with: Adagrad<Float, CPU>(learningRate: 0.1))
        expectLossDecreases(with: Adadelta<Float, CPU>(learningRate: 0.05))
        expectLossDecreases(with: RMSProp<Float, CPU>(learningRate: 0.01))
    }

    @Test func testUpdatedWeightsAreDetachedAndTrainable() {
        var model = makeModel()
        var optimizer = Adam<Float, CPU>(learningRate: 0.01)
        step(&model, with: &optimizer)

        for parameter in model.parameters {
            #expect(parameter.requiresGradient)
        }
        // A second step works only when the weights carry no graph from the first step.
        step(&model, with: &optimizer)
        #expect(model.parameters.count == 4)
    }

    @Test func testOptimizerStateRoundTripsThroughCodable() throws {
        var model = makeModel()
        var optimizer = Adam<Float, CPU>(learningRate: 0.01, useAMSGrad: true)
        for _ in 0 ..< 3 {
            step(&model, with: &optimizer)
        }

        let data = try JSONEncoder().encode(optimizer)
        var decoded = try JSONDecoder().decode(Adam<Float, CPU>.self, from: data)
        var copy = model

        step(&model, with: &optimizer)
        step(&copy, with: &decoded)

        #expect(model.parameters == copy.parameters)

        var momentum = Momentum<Float, CPU>(learningRate: 0.01)
        step(&model, with: &momentum)
        let momentumData = try JSONEncoder().encode(momentum)
        #expect(try JSONDecoder().decode(Momentum<Float, CPU>.self, from: momentumData).learningRate == momentum.learningRate)
    }

    @Test func testResetAcceptsANewSetOfWeights() {
        var model = makeModel()
        var optimizer = Adam<Float, CPU>(learningRate: 0.01)
        step(&model, with: &optimizer)

        model.first.freeze()
        optimizer.reset()
        #expect(model.parameters.count == 2)

        let before = model.first.weights
        step(&model, with: &optimizer)
        #expect(model.first.weights == before)
        #expect(model.parameters.count == 2)
    }
}

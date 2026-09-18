//
//  TransformerMNISTTests.swift
//  DL4S
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
import Testing

struct RowTransformerClassifier: LayerType, Codable {
    typealias Inputs = Tensor<Float, CPU>
    typealias Outputs = Tensor<Float, CPU>

    var project: Dense<Float, CPU>
    var positionalEncoding: PositionalEncoding<Float, CPU>
    var encoder: TransformerEncoder<Float, CPU>
    var classify: Dense<Float, CPU>

    var parameters: [Tensor<Float, CPU>] {
        Array([project.parameters, encoder.parameters, classify.parameters].joined())
    }

    var parameterPaths: [WritableKeyPath<Self, Tensor<Float, CPU>> & Sendable] {
        Array([
            parameterPaths(of: \.project),
            parameterPaths(of: \.encoder),
            parameterPaths(of: \.classify),
        ].joined())
    }

    init<Generator: RandomNumberGenerator>(hiddenDim: Int, layers: Int, heads: Int, using generator: inout Generator) {
        project = Dense(inputSize: 28, outputSize: hiddenDim, using: &generator)
        positionalEncoding = PositionalEncoding(hiddenSize: hiddenDim)
        encoder = TransformerEncoder(layerCount: layers, heads: heads, keyDim: hiddenDim / heads, valueDim: hiddenDim / heads, modelDim: hiddenDim, forwardDim: hiddenDim * 2, dropout: 0, using: &generator)
        classify = Dense(inputSize: hiddenDim, outputSize: 10, using: &generator)
    }

    func callAsFunction(_ images: Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        let batchSize = images.shape[0]
        let hiddenDim = positionalEncoding.hiddenSize
        let rows = project(images.view(as: [batchSize * 28, 28])).view(as: [batchSize, 28, hiddenDim])
        let encoded = encoder((input: rows + positionalEncoding(28), sequenceLengths: Array(repeating: 28, count: batchSize)))
        return classify(encoded.reduceMean(along: [1])).logSoftmax()
    }
}

@Suite(.serialized)
struct TransformerMNISTTests {
    @Test(.longRunning)
    func testRowTransformerLearnsMNIST() {
        let data = MNIST.full
        var generator = WyHash(seed: 42)
        let model = RowTransformerClassifier(hiddenDim: 64, layers: 2, heads: 4, using: &generator)
        var optimizer = Adam(model: model, learningRate: 0.001)
        let batchSize = 64
        let steps = 600
        var bar = ProgressBar<Float>(totalUnitCount: steps, formatUserInfo: { "loss: \($0)" }, label: "training")

        for _ in 1 ... steps {
            let (input, target) = MNIST.minibatch(from: data.trainingImages, labels: data.trainingLabels, count: batchSize, using: &generator)
            let prediction = optimizer.model(input.view(as: [batchSize, 28, 28]))
            let loss = categoricalNegativeLogLikelihood(expected: target, actual: prediction)
            optimizer.update(along: loss.gradients(of: optimizer.model.parameters))
            bar.next(userInfo: loss.item)
        }
        bar.complete()

        let prediction = optimizer.model(data.testImages.view(as: [-1, 28, 28]))
        let accuracy = MNIST.accuracy(of: prediction, labels: data.testLabels)
        #expect(accuracy > 0.9, "test accuracy \(accuracy)")
        print("Accuracy: \(accuracy)")
    }
}

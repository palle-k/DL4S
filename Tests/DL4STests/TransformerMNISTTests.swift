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

import XCTest
import DL4S


/// Classifies an MNIST image as a sequence of 28 rows with a Transformer encoder.
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

    var parameterPaths: [WritableKeyPath<Self, Tensor<Float, CPU>>] {
        Array([
            project.parameterPaths.map((\Self.project).appending(path:)),
            encoder.parameterPaths.map((\Self.encoder).appending(path:)),
            classify.parameterPaths.map((\Self.classify).appending(path:))
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

class TransformerMNISTTests: XCTestCase {
    func testRowTransformerLearnsMNIST() throws {
        try skipUnlessLongTestsEnabled()

        let ((images, labels), (testImages, testLabels)) = MNISTTests.loadMNIST(type: Float.self, device: CPU.self)
        var generator = WyHash(seed: 42)
        let model = RowTransformerClassifier(hiddenDim: 32, layers: 2, heads: 4, using: &generator)
        var optimizer = Adam(model: model, learningRate: 0.001)
        let batchSize = 64
        let steps = 600
        var bar = ProgressBar<Float>(totalUnitCount: steps, formatUserInfo: {"loss: \($0)"}, label: "training")

        for _ in 1 ... steps {
            let (input, target) = Random.minibatch(from: images, labels: labels, count: batchSize)
            let prediction = optimizer.model(input.view(as: [batchSize, 28, 28]))
            let loss = categoricalNegativeLogLikelihood(expected: target, actual: prediction)
            optimizer.update(along: loss.gradients(of: optimizer.model.parameters))
            bar.next(userInfo: loss.item)
        }
        bar.complete()

        let evaluationCount = 2000
        let prediction = optimizer.model(testImages[0 ..< evaluationCount].view(as: [evaluationCount, 28, 28])).elements
        let expected = testLabels[0 ..< evaluationCount].elements
        var correct = 0
        for i in 0 ..< evaluationCount {
            let scores = prediction[(i * 10) ..< (i * 10 + 10)]
            let predictedClass = scores.indices.max { scores[$0] < scores[$1] }! - i * 10
            if Int32(predictedClass) == expected[i] {
                correct += 1
            }
        }
        let accuracy = Float(correct) / Float(evaluationCount)
        print("test accuracy: \(accuracy)")
        XCTAssertGreaterThan(accuracy, 0.9)
    }
}

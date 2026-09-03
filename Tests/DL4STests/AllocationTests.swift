//
//  AllocationTests.swift
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


class AllocationTests: XCTestCase {
    /// Micro-benchmark: a loop over small-tensor operations.
    func testSmallTensorOperationThroughput() throws {
        try skipUnlessLongTestsEnabled()
        
        let a = Tensor<Float, CPU>((0 ..< 64).map(Float.init), shape: [8, 8])
        let b = Tensor<Float, CPU>(repeating: 0.5, shape: [8, 8])
        
        measure {
            for _ in 0 ..< 100_000 {
                _ = (a * b + a).reduceSum(along: [1])[0].item
            }
        }
    }
    
    /// Macro-benchmark: MNIST training steps on a dense network.
    func testMNISTDenseTrainingThroughput() throws {
        try skipUnlessLongTestsEnabled()
        
        let ((images, labels), _) = MNISTTests.loadMNIST(type: Float.self, device: CPU.self)
        let batchSize = 256
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 28 * 28, outputSize: 500)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 500, outputSize: 300)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 300, outputSize: 10)
            LogSoftmax<Float, CPU>()
        }
        var optimizer = Adam(model: model, learningRate: 0.001)
        
        measure {
            for _ in 0 ..< 20 {
                let (input, target) = Random.minibatch(from: images, labels: labels, count: batchSize)
                let predicted = optimizer.model(input.view(as: [batchSize, 28 * 28]))
                let loss = categoricalNegativeLogLikelihood(expected: target, actual: predicted)
                optimizer.update(along: loss.gradients(of: optimizer.model.parameters))
            }
        }
    }
    
    /// Macro-benchmark: MNIST training steps on a CNN.
    func testMNISTConvTrainingThroughput() throws {
        try skipUnlessLongTestsEnabled()
        
        let ((images, labels), _) = MNISTTests.loadMNIST(type: Float.self, device: CPU.self)
        let batchSize = 256
        let model = Sequential {
            Convolution2D<Float, CPU>(inputChannels: 1, outputChannels: 6, kernelSize: (5, 5), padding: 0)
            LayerNorm<Float, CPU>(inputSize: [6, 24, 24])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            Convolution2D<Float, CPU>(inputChannels: 6, outputChannels: 16, kernelSize: (5, 5), padding: 0)
            LayerNorm<Float, CPU>(inputSize: [16, 8, 8])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            Flatten<Float, CPU>()
            Dense<Float, CPU>(inputSize: 16 * 4 * 4, outputSize: 120)
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 120, outputSize: 10)
            LogSoftmax<Float, CPU>()
        }
        var optimizer = Adam(model: model, learningRate: 0.001)
        
        measure {
            for _ in 0 ..< 5 {
                let (input, target) = Random.minibatch(from: images, labels: labels, count: batchSize)
                let predicted = optimizer.model(input.view(as: [batchSize, 1, 28, 28]))
                let loss = categoricalNegativeLogLikelihood(expected: target, actual: predicted)
                optimizer.update(along: loss.gradients(of: optimizer.model.parameters))
            }
        }
    }
}

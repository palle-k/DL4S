//
//  RandomTests.swift
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

import XCTest
import DL4S

/// Tests for seeded and unseeded random initialization.
final class RandomTests: XCTestCase {
    func testSameSeedProducesIdenticalTensors() {
        var first = WyHash(seed: 42)
        var second = WyHash(seed: 42)
        
        XCTAssertEqual(
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [64, 32], min: -1, max: 1, using: &first).elements,
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [64, 32], min: -1, max: 1, using: &second).elements
        )
        XCTAssertEqual(
            Tensor<Double, CPU>(normalDistributedWithShape: [33], mean: 1, stdev: 2, using: &first).elements,
            Tensor<Double, CPU>(normalDistributedWithShape: [33], mean: 1, stdev: 2, using: &second).elements
        )
        XCTAssertEqual(
            Tensor<Float, CPU>(xavierNormalWithShape: [16, 8], using: &first).elements,
            Tensor<Float, CPU>(xavierNormalWithShape: [16, 8], using: &second).elements
        )
        XCTAssertEqual(
            Tensor<Float, CPU>(bernoulliDistributedWithShape: [256], probability: 0.3, using: &first).elements,
            Tensor<Float, CPU>(bernoulliDistributedWithShape: [256], probability: 0.3, using: &second).elements
        )
    }
    
    func testSameSeedProducesIdenticalLayers() {
        var first = WyHash(seed: 7)
        var second = WyHash(seed: 7)
        
        let dense1 = Dense<Float, CPU>(inputSize: 16, outputSize: 8, using: &first)
        let dense2 = Dense<Float, CPU>(inputSize: 16, outputSize: 8, using: &second)
        XCTAssertEqual(dense1.weights.elements, dense2.weights.elements)
        
        let conv1 = Convolution2D<Float, CPU>(inputChannels: 3, outputChannels: 4, kernelSize: (3, 3), using: &first)
        let conv2 = Convolution2D<Float, CPU>(inputChannels: 3, outputChannels: 4, kernelSize: (3, 3), using: &second)
        XCTAssertEqual(conv1.filters.elements, conv2.filters.elements)
        
        let lstm1 = LSTM<Float, CPU>(inputSize: 5, hiddenSize: 6, using: &first)
        let lstm2 = LSTM<Float, CPU>(inputSize: 5, hiddenSize: 6, using: &second)
        XCTAssertEqual(lstm1.parameters.map(\.elements), lstm2.parameters.map(\.elements))
        
        let embedding1 = Embedding<Float, CPU>(inputFeatures: 10, outputSize: 4, using: &first)
        let embedding2 = Embedding<Float, CPU>(inputFeatures: 10, outputSize: 4, using: &second)
        XCTAssertEqual(embedding1.embeddingMatrix.elements, embedding2.embeddingMatrix.elements)
    }
    
    func testSameSeedProducesIdenticalModels() {
        var first = WyHash(seed: 11)
        var second = WyHash(seed: 11)
        
        let block1 = ResidualBlock<Float, CPU>(inputShape: [4, 8, 8], outPlanes: 8, downsample: 2, using: &first)
        let block2 = ResidualBlock<Float, CPU>(inputShape: [4, 8, 8], outPlanes: 8, downsample: 2, using: &second)
        XCTAssertEqual(block1.parameters.map(\.elements), block2.parameters.map(\.elements))
        
        let transformer1 = Transformer<Float, CPU>(encoderLayers: 2, decoderLayers: 2, vocabSize: 32, hiddenDim: 16, heads: 2, keyDim: 8, valueDim: 8, forwardDim: 32, using: &first)
        let transformer2 = Transformer<Float, CPU>(encoderLayers: 2, decoderLayers: 2, vocabSize: 32, hiddenDim: 16, heads: 2, keyDim: 8, valueDim: 8, forwardDim: 32, using: &second)
        XCTAssertEqual(transformer1.parameters.map(\.elements), transformer2.parameters.map(\.elements))
    }
    
    func testUnseededModelsDiffer() {
        let transformer1 = Transformer<Float, CPU>(encoderLayers: 1, decoderLayers: 1, vocabSize: 32, hiddenDim: 16, heads: 2, keyDim: 8, valueDim: 8, forwardDim: 32)
        let transformer2 = Transformer<Float, CPU>(encoderLayers: 1, decoderLayers: 1, vocabSize: 32, hiddenDim: 16, heads: 2, keyDim: 8, valueDim: 8, forwardDim: 32)
        XCTAssertNotEqual(transformer1.parameters.map(\.elements), transformer2.parameters.map(\.elements))
    }
    
    func testOneGeneratorAdvancesBetweenCalls() {
        var generator = WyHash(seed: 1)
        let first = Tensor<Float, CPU>(uniformlyDistributedWithShape: [128], using: &generator).elements
        let second = Tensor<Float, CPU>(uniformlyDistributedWithShape: [128], using: &generator).elements
        XCTAssertNotEqual(first, second)
    }
    
    func testDifferentSeedsProduceDifferentTensors() {
        var first = WyHash(seed: 1)
        var second = WyHash(seed: 2)
        XCTAssertNotEqual(
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [128], using: &first).elements,
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [128], using: &second).elements
        )
    }
    
    func testUnseededCallsProduceDifferentValues() {
        let first = Tensor<Float, CPU>(uniformlyDistributedWithShape: [128])
        let second = Tensor<Float, CPU>(uniformlyDistributedWithShape: [128])
        XCTAssertNotEqual(first.elements, second.elements)
        
        let dense1 = Dense<Float, CPU>(inputSize: 16, outputSize: 8)
        let dense2 = Dense<Float, CPU>(inputSize: 16, outputSize: 8)
        XCTAssertNotEqual(dense1.weights.elements, dense2.weights.elements)
    }
    
    func testNormalFillWritesEveryElementForOddCounts() {
        var generator = WyHash(seed: 3)
        let odd = Tensor<Float, CPU>(normalDistributedWithShape: [7], mean: 100, stdev: 0.001, using: &generator)
        for value in odd.elements {
            XCTAssertEqual(value, 100, accuracy: 0.1)
        }
    }
    
    func testBoundedDrawsAreNotConstant() {
        var generator = WyHash(seed: 5)
        let draws = (0 ..< 64).map { _ in generator.next(upperBound: UInt32(100)) }
        XCTAssertGreaterThan(Set(draws).count, 1)
        XCTAssertTrue(draws.allSatisfy { $0 < 100 })
    }
}

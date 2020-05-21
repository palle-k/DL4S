//
//  TransformerTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 20.05.20.
//  Copyright (c) 2020 - Palle Klewitz
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

class TransformerTests: XCTestCase {
    func testTransformerConvergence() {
        let transformer = Transformer<Float, CPU>(encoderLayers: 2, decoderLayers: 2, vocabSize: 4, hiddenDim: 16, heads: 4, keyDim: 8, valueDim: 8, forwardDim: 32, dropout: 0)
        let samples: [[Int32]] = [
            [1, 2, 3, 0],
            [2, 0, 3, 1],
            [3, 2, 1, 0],
            [2, 1, 3, 3]
        ]
        let outputs: [[Int32]] = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        
        var optim = Adam(model: transformer, learningRate: 0.001)
        
        var bar = ProgressBar<Float>(totalUnitCount: 1000, formatUserInfo: {"loss: \($0)"}, label: "training")
        
        var lastLoss = Float.infinity
        
        for _ in 1 ... 1000 {
            let indices = (0 ..< 4).shuffled()
            let input = indices.map {samples[$0]}
            let expected = indices.map {outputs[$0]}
            let decoderInput = expected.map {
                [0] + $0.dropLast()
            }
            
            let prediction = optim.model((encoderInput: Tensor(input), decoderInput: Tensor(decoderInput), encoderInputLengths: [4, 4, 4, 4], decoderInputLengths: [4, 4, 4, 4]))
            let loss = categoricalNegativeLogLikelihood(expected: Tensor(expected), actual: prediction)
            let grads = loss.gradients(of: optim.model.parameters)
            optim.update(along: grads)
            bar.next(userInfo: loss.item)
            
            lastLoss = loss.item
            
            if loss.item < 0.01 {
                break
            }
        }
        bar.complete()
        
        XCTAssertLessThanOrEqual(lastLoss, 0.1)
    }
}

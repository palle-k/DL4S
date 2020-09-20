//
//  VectorXORTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 27.02.19.
//  Copyright (c) 2019 - Palle Klewitz
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

class VectorXORTest: XCTestCase {
    func testXNN() {
        var xor_src = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        var xor_dst = Tensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0]
        ])
        
        #if DEBUG
        xor_src.tag = "xor_src"
        xor_dst.tag = "xor_dst"
        #endif
        
        let net = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 6)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 6, outputSize: 1)
            Sigmoid<Float, CPU>()
        }
        var optim = Adam(model: net, learningRate: 0.05)
        
        for epoch in 1 ... 100 {
            let pred = optim.model(xor_src)
            let loss = binaryCrossEntropy(expected: xor_dst, actual: pred)
            let grads = loss.gradients(of: optim.model.parameters)
            
            optim.update(along: grads)
            
            if epoch.isMultiple(of: 10) {
                print("[\(epoch)/\(100)] loss: \(loss.item)")
            }
        }
        
        let predictions = optim.model(xor_src).view(as: -1)
        
        var correctCount = 0
        for i in 0 ..< 4 {
            if round(predictions[i].item) == xor_dst[i, 0].item {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / 4
        print("Accuracy: \(accuracy)")
        
        XCTAssertEqual(accuracy, 1)
    }
}

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
@testable import DL4S

class VectorXORTest: XCTestCase {
    func testXOR() {
        
        let net = Sequential<Float, CPU>(
            Dense(inputFeatures: 2, outputFeatures: 6).asAny(),
            Sigmoid().asAny(),
            Dense(inputFeatures: 6, outputFeatures: 1).asAny(),
            // Logging().asAny(),
            Sigmoid().asAny()
        )
        
        let inputs = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        let expectedOutputs = Tensor<Float, CPU>([0, 1, 1, 0])
        
        let optimizer = MomentumOptimizer(parameters: net.trainableParameters, learningRate: 0.05)
        // let optimizer = Adam(parameters: net.parameters, learningRate: 0.05)
        
        let epochs = 1000
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            let predictions = net.forward(inputs)
            let loss = binaryCrossEntropy(expected: expectedOutputs, actual: predictions)
            
            loss.backwards()
            optimizer.step()
            
            let lossValue = loss.item
            
            if epoch % 100 == 0 {
                print("[\(epoch)/\(epochs)] loss: \(lossValue / 4)")
            }
        }
        
//        let correctCount = dataset.lazy.filter { input, output -> Bool in
//            round(net.forward(input).view().item) == output.item
//        }.count
        
        let predictions = net.forward(inputs).view(as: -1)
        
        var correctCount = 0
        for i in 0 ..< 4 {
            if round(predictions[i].item) == expectedOutputs[i].item {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / 4
        print("Accuracy: \(accuracy)")
        
        XCTAssertEqual(accuracy, 1)
    }
}

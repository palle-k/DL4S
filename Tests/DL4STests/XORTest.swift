//
//  XORTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
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

/// Tests, whether the system can solve the XOR problem
class XORTest: XCTestCase {
    func testXOR() {
        let dataset: [([Variable], Variable)] = [
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 1),
            ([1, 1,], 0)
        ]
        
        let net = ScalarSequential(
            ScalarDense(inputs: 2, outputs: 6, weightScale: 0.1),
            ScalarSigmoid(),
            ScalarDense(inputs: 6, outputs: 1, weightScale: 0.1),
            ScalarSigmoid()
        )
        
        let learningRate: Float = 0.1
        
        for epoch in 1 ... 10000 {
            var loss = Variable(value: 0)
            
            for (input, expected) in dataset {
                let pred = net.forward(input)[0]
                let l = binaryCrossEntropy(expected: expected, actual: pred)
                // print(l.value)
                //print(pred.value)
                loss = loss + l
            }
            
            loss.zeroGradient()
            loss.backwards()
            
            
            
            for param in net.allParameters {
                param.value -= param.gradient * learningRate
            }
            
            if epoch % 100 == 0 {
                print("[\(epoch)]: loss \(loss.value / Float(dataset.count))")
            }
        }
        
        let correctCount = dataset.filter { input, expected in
            round(net.forward(input)[0].value) == expected.value
        }.count
        
        let accuracy = Double(correctCount) / Double(dataset.count)
        print("Accuracy: \(accuracy)")
        XCTAssertEqual(accuracy, 1)
    }
}

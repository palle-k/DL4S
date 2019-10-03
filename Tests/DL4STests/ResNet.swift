//
//  ResNet.swift
//  DL4STests
//
//  Created by Palle Klewitz on 21.04.19.
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

class ResNetTests: XCTestCase {
    func testResNet() {
        let resnet = ResNet<Float, CPU>(inputShape: [3, 32, 32], classCount: 32)
        let optim = Adam(parameters: resnet.trainableParameters, learningRate: 0.001)
        optim.zeroGradient()
        
        let t = Tensor<Float, CPU>(repeating: 0, shape: 64, 3, 32, 32)
        t.tag = "input"
        Random.fill(t, a: 0, b: 1)
        
//        let expected = Tensor<Int32, CPU>((0 ..< 64).map {_ in Int32.random(in: 0 ..< 32)})
        
//        let epochs = 100
        
//        for i in 1 ... epochs {
//            optim.zeroGradient()
//
//            let result = resnet(t)
//
//            let loss = categoricalCrossEntropy(expected: expected, actual: result)
//            loss.backwards()
//            optim.step()
//
//            print("[\(i)/\(epochs)]: \(loss)")
//        }
//
        measure {
            for _ in 0 ..< 5 {
                _ = resnet(t)
            }
        }
    }
}

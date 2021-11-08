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


class ModelTests: XCTestCase {
    func testResNet() {
        let resnet = ResNet18<Float, CPU>(inputShape: [3, 64, 64], classes: 256)
        var optim = Adam(model: resnet, learningRate: 0.001)
        
        let t = Tensor<Float, CPU>(uniformlyDistributedWithShape: 32, 3, 64, 64, min: 0, max: 1)
        let expected = Tensor<Int32, CPU>(uniformlyDistributedWithShape: 32, min: 0, max: 255)
        
        let epochs = 5
        
        for i in 1 ... epochs {
            let result = optim.model(t)
            
            let loss = categoricalNegativeLogLikelihood(expected: expected, actual: result)
            let grads = loss.gradients(of: optim.model.parameters)
            optim.update(along: grads)
            
            print("[\(i)/\(epochs)] \(loss)")
        }
        
        XCTAssertLessThan(
            categoricalNegativeLogLikelihood(expected: expected, actual: optim.model(t)).item,
            0.01
        )
    }
    
    func testAlexNet() {
        var alexNet = AlexNet<Float, CPU>(inputChannels: 3, classes: 256)
        alexNet.isDropoutActive = false
        var optim = Adam(model: alexNet, learningRate: 0.001)
        
        let t = Tensor<Float, CPU>(uniformlyDistributedWithShape: 32, 3, 192, 192, min: 0, max: 1)
        let expected = Tensor<Int32, CPU>(uniformlyDistributedWithShape: 32, min: 0, max: 255)
        
        let epochs = 5
        
        for i in 1 ... epochs {
            let result = optim.model(t)
            
            let loss = categoricalNegativeLogLikelihood(expected: expected, actual: result)
            let grads = loss.gradients(of: optim.model.parameters)
            optim.update(along: grads)
            
            print("[\(i)/\(epochs)] \(loss)")
        }
        
        XCTAssertLessThan(
            categoricalNegativeLogLikelihood(expected: expected, actual: optim.model(t)).item,
            0.1
        )
    }
    
    func testVGG() {
        let vgg = VGG11<Float, CPU>(inputChannels: 3, classes: 256)
        var optim = Adam(model: vgg, learningRate: 0.001)
        
        let t = Tensor<Float, CPU>(uniformlyDistributedWithShape: 16, 3, 192, 192, min: 0, max: 1)
        let expected = Tensor<Int32, CPU>(uniformlyDistributedWithShape: 16, min: 0, max: 255)
        
        let epochs = 5
        
        for i in 1 ... epochs {
            let result = optim.model(t)
            
            let loss = categoricalNegativeLogLikelihood(expected: expected, actual: result)
            let grads = loss.gradients(of: optim.model.parameters)
            optim.update(along: grads)
            
            print("[\(i)/\(epochs)] \(loss)")
        }
        
        XCTAssertLessThan(
            categoricalNegativeLogLikelihood(expected: expected, actual: optim.model(t)).item,
            0.1
        )
    }
}

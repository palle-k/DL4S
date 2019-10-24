//
//  GPUTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 05.07.19.
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

class GPUTests: XCTestCase {
    func testGPU1() {
        let t = Tensor<Float, GPU>([[1,2,3,4]])
        let u = Tensor<Float, GPU>([[10], [100], [1000], [10000]])
        
        let sum = u + t
        let diff = u - t
        let prod = u * t
        let quot = u / t
        
        print(
            sum,
            diff,
            prod,
            quot,
            separator: "\n"
        )
    }
    
    func testGPU3() {
        let a = Tensor<Float, GPU>([[1, 2, 3, 4]], requiresGradient: true)
        let b = Tensor<Float, GPU>([10, 20, 30, 40], shape: 4, 1, requiresGradient: true)
        
        let sum = a + b
        print(sum)
        
        let grads = sum.gradients(of: [a, b])
        print(grads[0])
        print(grads[1])
    }
    
    func testGPU4() {
        let a = Tensor<Float, GPU>([[1, 2, 3, 4]], requiresGradient: true)
        let b = Tensor<Float, GPU>([10, 20, 30, 40], shape: 4, 1, requiresGradient: true)
        
        let prod = a * b
        print(prod)
        
        let grads = prod.gradients(of: [a, b])
        print(grads[0])
        print(grads[1])
    }
    
    func testGPU5() {
        let a = Tensor<Float, GPU>([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [0, 1, 1, 0]
        ], requiresGradient: true)
        
        let result = a.reduceMax(along: 1)
        print(result)
        let grads = result.gradients(of: [a])
        print(grads[0])
    }
    
    func testGPUXOR() {
        let xorInputs = Tensor<Float, GPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        let xorLabels = Tensor<Float, GPU>([0, 1, 1, 0])
        
        let model = Sequential {
            Dense<Float, GPU>(inputSize: 2, outputSize: 6)
            Tanh<Float, GPU>()
            Dense<Float, GPU>(inputSize: 6, outputSize: 1)
            Sigmoid<Float, GPU>()
        }
        var optimizer = Adam(model: model, learningRate: 0.01)
        
        let epochs = 1000
        for epoch in 1 ... epochs {
            let p = optimizer.model(xorInputs)
            
            let d = (p.flattened() - xorLabels)
            let l = (d * d).reduceSum()
            let grads = l.gradients(of: optimizer.model.parameters)
            optimizer.update(along: grads)
            
            if epoch.isMultiple(of: 100) {
                print("[\(epoch)/\(epochs)] loss: \(l)")
            }
        }
    }
    
    func testMMul() {
        let a = Tensor<Float, GPU>([[1,2,3]])
        let b = Tensor<Float, GPU>([[4],[5],[6]])
        
        let result = matMul(a, b)
        print(result)
    }
    
    func testRandom() {
        let a = Tensor<Float, GPU>(normalDistributedWithShape: [4, 4], mean: 0, stdev: 1)
        print(a)
    }
    
    func testTranspose() {
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        print(a.transposed())
    }
    
    func testReduce() {
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        print(a.reduceSum(along: 0, 1))
    }
    
    func testSoftmax() {
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        print(softmax(a))
    }
}

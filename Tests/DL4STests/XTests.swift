//
//  XTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.10.19.
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

class XTests: XCTestCase {
    func testSecondDerivative() {
        let t = XTensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        
        let result = t * t * t
        print(result)
        print()
        
        let grad = result.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(grad)
        print()
        
        let secondGrad = grad.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(secondGrad)
        print()
        
        let thirdGrad = secondGrad.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(thirdGrad)
    }
    
    func testGradient() {
        let t1 = Tensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        let r1 = 1 / t1
        
        
        let t2 = XTensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        let r2 = 1 / t2
        
        r1.backwards()
        print(r1)
        print(t1.gradientDescription!)
        
        let grad = r2.gradients(of: [t2])[0]
        print(r2)
        print(grad)
    }
    
    func testMatMul() {
        var lhs = XTensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
        ], requiresGradient: true)
        
        var rhs = XTensor<Float, CPU>([
            [1, 1],
            [2, 2],
            [3, 3]
        ], requiresGradient: true)
        
        #if DEBUG
        lhs.tag = "lhs"
        rhs.tag = "rhs"
        #endif
        
        var result = lhs.matrixMultiplied(with: rhs)
        #if DEBUG
        result.tag = "result"
        #endif
        
        var grads = result.gradients(of: [lhs, rhs], retainBackwardsGraph: true)
        
        #if DEBUG
        grads[0].tag = "∇lhs"
        grads[1].tag = "∇rhs"
        #endif
        
        print(grads.map {$0.reduceSum()}.reduce(0, +).graph())
        
        let lhsGradGrads = grads[0].reduceSum().gradients(of: [lhs, rhs])
        let rhsGradGrads = grads[1].reduceSum().gradients(of: [lhs, rhs])
        
        print(grads.map {$0.detached()})
        print(lhsGradGrads, rhsGradGrads)
    }
    
    func testXNN() {
        var xor_src = XTensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        var xor_dst = XTensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0]
        ])
        
        #if DEBUG
        xor_src.tag = "xor_src"
        xor_dst.tag = "xor_dst"
        #endif
        
        let net = XSequential {
            XDense<Float, CPU>(inputSize: 2, outputSize: 6)
            XTanh<Float, CPU>()
            XDense<Float, CPU>(inputSize: 6, outputSize: 1)
            XSigmoid<Float, CPU>()
        }
        var optim = XAdam(model: net, learningRate: 0.05)
        
        for epoch in 1 ... 10000 {
            let pred = optim.model(xor_src)
            let loss = binaryCrossEntropy(expected: xor_dst, actual: pred)
            let grads = loss.gradients(of: optim.model.parameters, retainBackwardsGraph: false)
            
            optim.update(along: grads)
            
            if epoch.isMultiple(of: 1000) {
                print("[\(epoch)/\(10000)] loss: \(loss.item)")
            }
        }
    }
    
    func testXRNN() {
        let model = XLSTM<Float, CPU>(inputSize: 32, hiddenSize: 32)
        var input = XTensor<Float, CPU>(uniformlyDistributedWithShape: [1, 4, 32], requiresGradient: true)
        input.requiresGradient = true
        #if DEBUG
        input.tag = "input"
        #endif
        
        let result = model(input).0.hiddenState
        let inputGrad = result.gradients(of: [input], retainBackwardsGraph: true)[0]
        
        print(inputGrad.graph())
    }
    
    func testGrads() {
        // let a = XTensor<Float, CPU>([[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]], requiresGradient: true)
        let b = Tensor<Float, CPU>([[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]], requiresGradient: true)
        
        // let ra = a[0, 2]
        let rb = b[0, 2]
        
        print(rb)
        
        // let ga = ra.gradients(of: [a])[0]
        rb.backwards()
        
        // print(ga)
        print(b.gradientDescription!)
        
        // print(ra.graph())
        print(rb.graph)
    }
}

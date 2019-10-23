//
//  GradientTests.swift
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

class GradientTests: XCTestCase {
    func testSecondDerivative() {
        let t = Tensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        
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
        let t2 = Tensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        let r2 = 1 / t2
        
        let grad = r2.gradients(of: [t2])[0]
        print(r2)
        print(grad)
    }
    
    func testMatMul() {
        var lhs = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
        ], requiresGradient: true)
        
        var rhs = Tensor<Float, CPU>([
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
    
    func testXRNN() {
        let model = LSTM<Float, CPU>(inputSize: 32, hiddenSize: 32)
        var input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [1, 4, 32], requiresGradient: true)
        input.requiresGradient = true
        #if DEBUG
        input.tag = "input"
        #endif
        
        let result = model(input).0.hiddenState
        let inputGrad = result.gradients(of: [input], retainBackwardsGraph: true)[0]
        
        print(inputGrad.graph())
    }
}

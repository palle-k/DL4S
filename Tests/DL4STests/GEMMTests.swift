//
//  GEMMTests.swift
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

class GEMMTests: XCTestCase {
    func testGEMM1() {
        let lhs = Tensor<Float, CPU>([0, 1, 2, 3, 4, 5]).view(as: -1, 1)
        let rhs = Tensor<Float, CPU>([1, 1, 1, 1, 1, 1]).view(as: 1, -1)
        
        let result = Tensor<Float, CPU>(repeating: 5, shape: 1, 1)
        
        Float.matMulAddInPlace(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values.pointer, lhsShape: (1, 6), rhsShape: (6, 1), resultShape: (1, 1), transposeFirst: false, transposeSecond: false)
        print(result)
    }
    
    func testGEMM2() {
        let lhs = Tensor<Float, CPU>([1, 2, 3, 4, 5]).view(as: 1, -1)
        let rhs = Tensor<Float, CPU>([1, 1, 1, 1, 1]).view(as: -1, 1)
        
        let result = Tensor<Float, CPU>(repeating: 1, shape: 5, 5)
        
        Float.matMulAddInPlace(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values.pointer, lhsShape: (5, 1), rhsShape: (1, 5), resultShape: (5, 5), transposeFirst: false, transposeSecond: false)
        print(result)
    }
    
    func testGEMM3() {
        let lhs = Tensor<Float, CPU>([1, 2, 3, 4, 5]).view(as: 1, -1)
        let rhs = Tensor<Float, CPU>([1, 1, 1, 1, 1]).view(as: -1, 1)
        
        let result = Tensor<Float, CPU>(repeating: 1, shape: 5, 5)
        
        Float.matMulAddInPlace(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values.pointer, lhsShape: (1, 5), rhsShape: (5, 1), resultShape: (5, 5), transposeFirst: true, transposeSecond: true)
        print(result)
    }
    
    func testGEMM4() {
        let lhs = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        let rhs = Tensor<Float, CPU>([
            [2, 2],
            [3, 3]
        ])
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        Float.matMulAddInPlace(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values.pointer, lhsShape: (2, 3), rhsShape: (2, 2), resultShape: (3, 2), transposeFirst: true, transposeSecond: false)
        
        let ref = lhs.T.mmul(rhs)
        
        print(result)
        print(ref)
    }
    
    func testGEMM5() {
        let lhs = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
            ])
        
        let rhs = Tensor<Float, CPU>([
            [2, 2],
            [3, 3]
            ])
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        Float.matMulAddInPlace(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values.pointer, lhsShape: (2, 3), rhsShape: (2, 2), resultShape: (3, 2), transposeFirst: true, transposeSecond: true)
        
        let ref = lhs.T.mmul(rhs.T)
        
        print(result)
        print(ref)
    }
}

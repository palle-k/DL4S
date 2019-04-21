//
//  GEMMTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 21.04.19.
//

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

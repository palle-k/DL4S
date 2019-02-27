//
//  VecTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 26.02.19.
//

import XCTest
@testable import DL4S

class VecTests: XCTestCase {
    func testVectorWriteItem() {
        let vector: Vector<Float> = Vector([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
        vector[0, 0].item = 2
    }
    
    func testVectorReadSlice() {
        let v: Vector<Float> = Vector([0,1,2,3,4,5], shape:3,2)
        print(v)
        print(v[nil, 0 ..< 2])
        print(v[nil, 0 ..< 1])
    }
    
    func testVectorWrite() {
        let v: Vector<Float> = Vector([0,1,2,3,4,5], shape:3,2)
        v[0,0] = 10
        v[2,1] = 20
        print(v)
    }
    
    func testVecOps() {
        let v: Vector<Float> = Vector([0,1,2,3,2,1], shape:3,2)
        
        let result = log(exp(v * v))
        print(result)
        result.backwards()
        
        debugPrint(v)
    }
}

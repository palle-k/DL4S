//
//  File.swift
//  
//
//  Created by Palle Klewitz on 21.05.20.
//

import Foundation
import XCTest
import DL4S
import AF

fileprivate typealias GPU = ArrayFire

class ArrayFireTests: XCTestCase {
    func testAF1() {
        GPU.setOpenCL()
        GPU.printInfo()
        
        let a = Tensor<Float, GPU>([0, 1, 2, 3, 4])
        let b = Tensor<Float, GPU>([[1], [-1], [0.5], [-0.5]])
        print(sigmoid(a * b))
    }
    
    func testAFIndex() {
        let a = Tensor<Float, GPU>((0 ..< 64).map(Float.init)).view(as: [8, 2, 4])
        print(a[nil, nil, 3])
    }
    
    func testAFMatMul() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        let b = Tensor<Float, GPU>([[1, 2], [3, 4], [5, 6]])
        print(a.matrixMultiplied(with: b))
    }
    
    func testAFStack() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        let b = Tensor<Float, GPU>([[7, 8, 9], [10, 11, 12]])
        print(Tensor(stacking: [a, b], along: 0))
    }
    
    func testAFReduce() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]])
        print(a.reduceSum(along: 1))
    }
}

//
//  EngineV2Tests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 16.03.19.
//

import XCTest
@testable import DL4S

class EngineV2Tests: XCTestCase {
    func testBroadcast() {
        let lhs = Tensor<Float, CPU>([[1,2],[3,4],[5,6]])
        let rhs = Tensor<Float, CPU>([1,2])
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        CPUEngine.broadcastMul(lhs: rhs.shapedValues, rhs: lhs.shapedValues, result: result.shapedValues)
        
        print(result)
    }
    
    func testBroadcast2() {
        let x = Tensor<Float, CPU>([1, 0.5, 0]).view(as: -1, 1)
        let result = 1 - x
        print(result)
    }
    
    func testBroadcast3() {
        let lhs = Tensor<Float, CPU>([[1,2],[3,4],[5,6]])
        let rhs = Tensor<Float, CPU>([1,2,3]).view(as: -1, 1)
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        CPUEngine.broadcastAdd(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        print(result)
    }
    
    func testBroadcast4() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let b = Tensor<Float, CPU>([1,3,3,7])
        
        let result = a.unsqueeze(at: 2) + b.view(as: -1, 1, 1)
        
        print(result.squeeze())
    }
    
    func testBroadcast5() {
        let a = Tensor<Float, CPU>(repeating: 0, shape: 16, 16)
        let b = Tensor<Float, CPU>(repeating: 0, shape: 16, 1)
        Random.fill(b, a: 0, b: 1)
        
        let result = a + b
        
        print(result)
    }
    
    func testReduceSum1() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.shapedValues
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.shapedValues
        
        CPU.Engine.reduceSum(values: v, result: r, axis: 0)
        
        print(result)
    }
    
    func testReduceSum2() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.shapedValues
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.shapedValues
        
        CPU.Engine.reduceSum(values: v, result: r, axis: 1)
        
        print(result)
    }
    
    func testReduceSum3() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.shapedValues
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: [])
        let r = result.shapedValues
        
        CPU.Engine.reduceSum(values: v, result: r, axes: [0, 1])
        
        print(result)
    }
}

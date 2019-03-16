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
        let lhs = Tensor<Float, CPU>([1,2,3,4])
        let rhs = Tensor<Float, CPU>([1,2,3]).view(as: -1, 1)
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 4)
        
        CPUEngine.broadcastMul(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        print(result)
    }
}

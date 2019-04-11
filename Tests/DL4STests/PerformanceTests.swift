//
//  PerformanceTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 10.04.19.
//

import XCTest
@testable import DL4S

class PerformanceTests: XCTestCase {
    func testHeaviside() {
        let src = Tensor<Float, CPU>(repeating: 0, shape: 32, 200)
        let dst = Tensor<Float, CPU>(repeating: 0, shape: 32, 200)
        
        Random.fill(src, a: -10, b: 10)
        
        let srcPtr = src.values.immutable
        let dstPtr = dst.values.pointer
        
        self.measure {
            for _ in 0 ..< 100000 {
                Float.heaviside(values: srcPtr, result: dstPtr, count: 32 * 200)
            }
        }
        
        print(sum(src) + sum(dst)) // retain src and dst
    }
    
}

//
//  GPUTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 05.07.19.
//

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
        sum.backwards()
        print(a.gradientDescription!)
        print(b.gradientDescription!)
    }
    
    func testGPU4() {
        let a = Tensor<Float, GPU>([[1, 2, 3, 4]], requiresGradient: true)
        let b = Tensor<Float, GPU>([10, 20, 30, 40], shape: 4, 1, requiresGradient: true)
        
        let prod = a * b
        print(prod)
        prod.backwards()
        print(a.gradientDescription!)
        print(b.gradientDescription!)
    }
    
    func testGPU5() {
        let a = Tensor<Float, GPU>([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [0, 1, 1, 0]
        ], requiresGradient: true)
        
        let result = max(a, axis: 0)
        print(result)
        result.backwards()
        print(a.gradientDescription!)
    }
}

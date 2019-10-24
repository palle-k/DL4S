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
        
        let grads = sum.gradients(of: [a, b])
        print(grads[0])
        print(grads[1])
    }
    
    func testGPU4() {
        let a = Tensor<Float, GPU>([[1, 2, 3, 4]], requiresGradient: true)
        let b = Tensor<Float, GPU>([10, 20, 30, 40], shape: 4, 1, requiresGradient: true)
        
        let prod = a * b
        print(prod)
        
        let grads = prod.gradients(of: [a, b])
        print(grads[0])
        print(grads[1])
    }
    
    func testGPU5() {
        let a = Tensor<Float, GPU>([
            [1, 2, 3, 4],
            [-1, -2, -3, -4],
            [0, 1, 1, 0]
        ], requiresGradient: true)
        
        let result = a.reduceMax(along: 1)
        print(result)
        let grads = result.gradients(of: [a])
        print(grads[0])
    }
    
    func testGPUXOR() {
        let xorInputs = Tensor<Float, GPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        let xorLabels = Tensor<Float, GPU>([0, 1, 1, 0])
        
        let model = Sequential {
            Dense<Float, GPU>(inputSize: 2, outputSize: 6)
            Tanh<Float, GPU>()
            Dense<Float, GPU>(inputSize: 6, outputSize: 1)
            Sigmoid<Float, GPU>()
        }
        var optimizer = SGD(model: model, learningRate: 0.01)
        
        let epochs = 100
        for epoch in 1 ... epochs {
            let p = optimizer.model(xorInputs)
            let d = (p.flattened() - xorLabels)
            let l = d * d / 4
            let grads = l.gradients(of: optimizer.model.parameters)
            optimizer.update(along: grads)
            
            if epoch.isMultiple(of: 1) {
                print("[\(epoch)/\(epochs)] loss: \(l)")
            }
        }
    }
}

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
    func testAFBinop() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, GPU>([0, 1, 2, 3])
        let b = Tensor<Float, GPU>([2, 1, 3, 2])
        print(a * b)
    }
    
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
    
    func testGather() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, GPU>([[1,2,3], [4,5,6], [7,8,9]])
        let ctx = Tensor<Int32, GPU>([0, 1, 2])
        print(a.gather(using: ctx, alongAxis: 0))
    }
    
    func testScatter() {
        GPU.setOpenCL()
        
        let ctx = Tensor<Int32, GPU>([2, 2, 1])
        let b = Tensor<Float, GPU>([3, 1, 4])
        print(b.scatter(using: ctx, alongAxis: 1, withSize: 3))
    }
    
    func testXOR() {
        GPU.setOpenCL()
        
        var xor_src_cpu = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        var xor_dst_cpu = Tensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0]
        ])
        
        #if DEBUG
        xor_src_cpu.tag = "xor_src"
        xor_dst_cpu.tag = "xor_dst"
        #endif
        
        let xor_src_gpu = xor_src_cpu.copied(to: GPU.self)
        let xor_dst_gpu = xor_dst_cpu.copied(to: GPU.self)
        
        let net_cpu = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 6)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 6, outputSize: 1)
            Sigmoid<Float, CPU>()
        }
        var net_gpu = Sequential {
            Dense<Float, GPU>(inputSize: 2, outputSize: 6)
            Tanh<Float, GPU>()
            Dense<Float, GPU>(inputSize: 6, outputSize: 1)
            Sigmoid<Float, GPU>()
        }
        let paths = net_gpu.parameterPaths
        for (p, t) in zip(paths, net_cpu.parameters) {
            net_gpu[keyPath: p] = t.copied(to: GPU.self)
            net_gpu[keyPath: p].requiresGradient = true
        }
        
        var optim_cpu = Adam(model: net_cpu, learningRate: 0.05)
        var optim_gpu = Adam(model: net_gpu, learningRate: 0.05)
        
        for epoch in 1 ... 100 {
            let pred_cpu = optim_cpu.model(xor_src_cpu)
            let loss_cpu = binaryCrossEntropy(expected: xor_dst_cpu, actual: pred_cpu)
            let grads_cpu = loss_cpu.gradients(of: optim_cpu.model.parameters)
            
            let pred_gpu = optim_gpu.model(xor_src_gpu)
            let loss_gpu = binaryCrossEntropy(expected: xor_dst_gpu, actual: pred_gpu)
            let grads_gpu = loss_gpu.gradients(of: optim_gpu.model.parameters)
            
            optim_cpu.update(along: grads_cpu)
            optim_gpu.update(along: grads_gpu)
            
            if epoch.isMultiple(of: 10) {
                print("[\(epoch)/\(100)] loss CPU:\(loss_cpu.item), GPU: \(loss_gpu.item)")
            }
        }
        
        let predictions = optim_cpu.model(xor_src_cpu).view(as: -1)
        
        var correctCount = 0
        for i in 0 ..< 4 {
            if round(predictions[i].item) == xor_dst_cpu[i, 0].item {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / 4
        print("Accuracy: \(accuracy)")
        
        XCTAssertEqual(accuracy, 1)
    }
    
    func testCrossEntropy() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [10], min: 0.01, max: 0.99, requiresGradient: true)
        var b = a.copied(to: GPU.self)
        b.requiresGradient = true
        
        let l = Tensor<Float, CPU>(bernoulliDistributedWithShape: [10], probability: 0.3)
        let m = l.copied(to: GPU.self)
        
        let ce_a = binaryCrossEntropy(expected: l, actual: a)
        let ce_b = binaryCrossEntropy(expected: m, actual: b)
        
        print(ce_a)
        print(ce_b)
        let grad_a = ce_a.gradients(of: [a])[0]
        let grad_b = ce_b.gradients(of: [b])[0]
        print(grad_a)
        print(grad_b)
    }
    
    func testReduceGrads() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [10, 10], min: 0.01, max: 0.99, requiresGradient: true)
        var b = a.copied(to: GPU.self)
        b.requiresGradient = true
        
        print(a.reduceSum(along: 1))
        print(b.reduceSum(along: 1))
        
        print(a.reduceSum(along: 1).gradients(of: [a])[0])
        print(b.reduceSum(along: 1).gradients(of: [b])[0])
    }
    
    func testCategoricalCrossEntropy() {
        GPU.setOpenCL()
        
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [10, 5], min: 0.01, max: 0.99, requiresGradient: true).softmax()
        let b = { () -> Tensor<Float, GPU> in
            var b = a.copied(to: GPU.self)
            b.requiresGradient = true
            return b
        }()
        
        let l = Tensor<Int32, CPU>(uniformlyDistributedWithShape: 10, min: 0, max: 4, requiresGradient: false)
        let m = l.copied(to: GPU.self)
        
        let x = categoricalCrossEntropy(expected: l, actual: a)
        let y = categoricalCrossEntropy(expected: m, actual: b)
        print(x)
        print(y)
        
        let ga = x.gradients(of: [a])[0]
        let gb = y.gradients(of: [b])[0]
        print()
        print(ga)
        print()
        print(gb)
    }
}

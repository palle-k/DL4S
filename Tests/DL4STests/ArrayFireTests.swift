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
import AppKit

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
    
    func testMNISTLoad() {
        let ((images, labels), _) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: CPU.self)
        let ((images_af, labels_af), _) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: GPU.self)
        
        print(images.flattened().reduceMean())
        print(images_af.flattened().reduceMean())
        let delta = images - images_af.copied(to: CPU.self)
        print((delta * delta).reduceSum())
        
        print(labels.reduceSum())
        print(labels_af.reduceSum())
        print(labels_af.copied(to: CPU.self) == labels)
    }
    
    func testStack() {
        let a = Tensor<Float, GPU>([[
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]])
        let b = Tensor<Float, GPU>([[
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]])
        print(a)
        print(b)
        
        print(Tensor(stacking: [a, b], along: 0))
    }
    
    func testStack2() {
        let a = Tensor<Int32, GPU>([2])
        let b = Tensor<Int32, GPU>([3])
        let c = Tensor<Int32, GPU>([7])
        let d = Tensor<Int32, GPU>([-1])
        print(a, b, c, d)
        print(Tensor(stacking: [a, b, c, d], along: 0))
    }
    
    func testStack3() {
        GPU.printInfo()
        let a = Tensor<Int32, GPU>(uniformlyDistributedWithShape: [60000], min: 0, max: 9)
        let b = Tensor<Float, GPU>(uniformlyDistributedWithShape: [256])
        let stacked = Random.minibatch(from: a, count: 256)
        print(stacked)
        print(b.scatter(using: stacked, alongAxis: 1, withSize: 10))
    }
    
    func testStack4() {
        let maxLen = 8
        let hiddenSize = 16
        
        let inputRange = Tensor<Float, GPU>((0 ..< maxLen).map(Float.init))
        
        let hiddenRange = Tensor<Float, GPU>((0 ..< hiddenSize / 2).map(Float.init))
        let frequencies = Tensor(10000).raised(toPowerOf: hiddenRange / Tensor(Float(hiddenSize / 2)))
        let samplePoints = inputRange.unsqueezed(at: 1) / frequencies.unsqueezed(at: 0) // [seqlen, hiddenSize / 2]
        
        let samples = Tensor(
            stacking: [
                sin(samplePoints).unsqueezed(at: 2),
                cos(samplePoints).unsqueezed(at: 2)
            ],
            along: 2
        ).view(as: -1, hiddenSize) // [seqlen, hiddenSize]
        
        print(samples)
    }
    
    func testArgmax() {
        let a = Tensor<Float, GPU>([5, 6, 7, 8, 3])
        print(a.argmax())
    }
    
    func testPermute1() {
        let a = Tensor<Int32, GPU>([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ])
        
        print(a.permuted(to: 1, 0))
        print(a.copied(to: CPU.self).permuted(to: 1, 0))
    }
    
    func testPermute() {
        let a = Tensor<Int32, GPU>([
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8]
            ],
            [
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            [
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ])
        
        print(a)
        print()
        print(a.copied(to: CPU.self).permuted(to: 1, 0, 2))
    }
    
    func testMultiSum() {
        let a = Tensor<Float, GPU>(uniformlyDistributedWithShape: 4, 5, 6)
        print(a.reduceSum(along: 0, 1))
        print()
        print(a.copied(to: CPU.self).reduceSum(along: 0, 1))
    }
    
    func testBand() {
        let x = Tensor<Float, GPU>(uniformlyDistributedWithShape: 5, 7)
        print(x.bandMatrix(belowDiagonal: 2, aboveDiagonal: -1))
    }
    
    
    func testBroadcastMatmul() {
        let a = Tensor<Float, GPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)
        let b = Tensor<Float, GPU>([[1, 2], [3, 4], [5, 6]], requiresGradient: true)
        let result = a.broadcastMatrixMultiplied(with: b)
        
        let loss = result.raised(toPowerOf: 2).reduceSum()
        let grads = loss.gradients(of: [a, b])
        print(loss)
        print(grads[0])
        print(grads[1])
    }
}

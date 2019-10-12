//
//  XTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.10.19.
//

import XCTest
@testable import DL4S

class XTests: XCTestCase {
    func testSecondDerivative() {
        let t = XTensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        
        let result = t * t * t
        print(result)
        print()
        
        let grad = result.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(grad)
        print()
        
        let secondGrad = grad.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(secondGrad)
        print()
        
        let thirdGrad = secondGrad.gradients(of: [t], retainBackwardsGraph: true)[0]
        print(thirdGrad)
    }
    
    func testGradient() {
        let t1 = Tensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        let r1 = 1 / t1
        
        
        let t2 = XTensor<Float, CPU>([1,2,3,4], requiresGradient: true)
        let r2 = 1 / t2
        
        r1.backwards()
        print(r1)
        print(t1.gradientDescription!)
        
        let grad = r2.gradients(of: [t2])[0]
        print(r2)
        print(grad)
    }
    
    func testMatMul() {
        var lhs = XTensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
        ], requiresGradient: true)
        
        var rhs = XTensor<Float, CPU>([
            [1, 1],
            [2, 2],
            [3, 3]
        ], requiresGradient: true)
        
        #if DEBUG
        lhs.tag = "lhs"
        rhs.tag = "rhs"
        #endif
        
        var result = lhs.matMul(rhs)
        #if DEBUG
        result.tag = "result"
        #endif
        
        var grads = result.gradients(of: [lhs, rhs], retainBackwardsGraph: true)
        
        #if DEBUG
        grads[0].tag = "∇lhs"
        grads[1].tag = "∇rhs"
        #endif
        
        print(grads.map {$0.reduceSum()}.reduce(0, +).graph())
        
        let lhsGradGrads = grads[0].reduceSum().gradients(of: [lhs, rhs])
        let rhsGradGrads = grads[1].reduceSum().gradients(of: [lhs, rhs])
        
        print(grads.map {$0.detached()})
        print(lhsGradGrads, rhsGradGrads)
    }
    
    func testXNN() {
        var xor_src = XTensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        var xor_dst = XTensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0]
        ])
        
        #if DEBUG
        xor_src.tag = "xor_src"
        xor_dst.tag = "xor_dst"
        #endif
        
        let net = XUnion {
            XDense<Float, CPU>(inputSize: 2, outputSize: 6)
            XTanh<Float, CPU>()
            XDense<Float, CPU>(inputSize: 6, outputSize: 1)
            XSigmoid<Float, CPU>()
        }
        var optim = XMomentum(model: net, learningRate: 0.05)
        
        for epoch in 1 ... 10000 {
            let pred = optim.model(xor_src)
            let loss = binaryCrossEntropy(expected: xor_dst, actual: pred)
            let grads = loss.gradients(of: optim.model.parameters, retainBackwardsGraph: true)
            
            var gradPenalty: XTensor<Float, CPU> = 0

            for g in grads {
                let gSq = g * g
                let penalty = (-gSq).reduceMean()
                gradPenalty += penalty
            }
            
            let penaltyLoss = 3 * gradPenalty
            let penaltyGrads = penaltyLoss.gradients(of: optim.model.parameters)
            
            let fullGrads = zip(grads, penaltyGrads).map(+)
            optim.update(along: fullGrads)
            
            if epoch.isMultiple(of: 100) {
                print("[\(epoch)/\(10000)] loss: \(loss), penalty: \(penaltyLoss)")
            }
        }
    }
}

//
//  NNTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
//

import XCTest
@testable import DL4S


class NNTests: XCTestCase {
    func testOptim() {
        let xorDataset: [([Variable], Variable)] = [
            ([0, 0], 0),
            ([1, 1], 0),
            ([0, 1], 1),
            ([1, 0], 1)
        ]
        
        let numHidden = 8
        
        let W1 = [[Variable]].init(repeating: 0, rows: numHidden, columns: 2)
        let W2 = [[Variable]].init(repeating: 0, rows: 1, columns: numHidden)
        let b1 = [Variable].init(repeating: Float(0), count: numHidden)
        let b2 = [Variable].init(repeating: Float(0), count: 1)
        
        W1.fillRandomly(-0.01 ... 0.01)
        W2.fillRandomly(-0.01 ... 0.01)
        b1.fillRandomly(-0.01 ... 0.01)
        b2.fillRandomly(-0.01 ... 0.01)
        
        func forward(_ x: [Variable]) -> Variable {
            return sigmoid(b2 + W2 * sigmoid(b1 + W1 * x))[0]
        }
        
        func loss(expected: Variable, actual: Variable) -> Variable {
            return -(expected * log(actual) + (1 - expected) * log(1 - actual))
        }
        
        let learningRate: Float = 0.1
        
        for epoch in 0 ..< 3000 {
            var l: Variable = 0
            
            for (input, out) in xorDataset {
                let pred = forward(input)
                l = l + loss(expected: out, actual: pred)
            }
            
            l.zeroGradient()
            l.backwards()

            for param in Array(W1.joined()) + Array(W2.joined()) {
                param.value -= param.gradient * learningRate
            }
            
            if epoch % 100 == 0 {
                print("Loss after \(epoch): \(l.value / Float(xorDataset.count))")
            }
        }
        
        let correctCount = xorDataset
            .map {(forward($0), $1)}
            .filter {round($0.value) == $1.value}
            .count
        
        print("Acc: \(Double(correctCount) / Double(xorDataset.count))")
    }
}

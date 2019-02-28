//
//  XORTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
//

import XCTest
@testable import DL4S

/// Tests, whether the system can solve the XOR problem
class XORTest: XCTestCase {
    func testXOR() {
        let dataset: [([Variable], Variable)] = [
            ([0, 0], 0),
            ([1, 0], 1),
            ([0, 1], 1),
            ([1, 1,], 0)
        ]
        
        let net = ScalarSequential(
            ScalarDense(inputs: 2, outputs: 6, weightScale: 0.1),
            ScalarSigmoid(),
            ScalarDense(inputs: 6, outputs: 1, weightScale: 0.1),
            ScalarSigmoid()
        )
        
        let learningRate: Float = 0.1
        
        for epoch in 1 ... 1000 {
            var loss = Variable(value: 0)
            
            for (input, expected) in dataset {
                let pred = net.forward(input)[0]
                let l = binaryCrossEntropy(expected: expected, actual: pred)
                // print(l.value)
                loss = loss + l
            }
            
            loss.zeroGradient()
            loss.backwards()
            
            for param in net.allParameters {
                param.value -= param.gradient * learningRate
            }
            
            if epoch % 100 == 0 {
                print("[\(epoch)]: loss \(loss.value / Float(dataset.count))")
            }
        }
        
        let correctCount = dataset.filter { input, expected in
            round(net.forward(input)[0].value) == expected.value
        }.count
        
        let accuracy = Double(correctCount) / Double(dataset.count)
        print("Accuracy: \(accuracy)")
        XCTAssertEqual(accuracy, 1)
    }
}

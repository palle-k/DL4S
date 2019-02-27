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
            ([0, 1], 0),
            ([1, 1,], 0)
        ]
        
        let net = Sequential(
            Dense(inputs: 2, outputs: 4, weightScale: 0.1),
            Sigmoid(),
            Dense(inputs: 4, outputs: 1, weightScale: 0.1),
            Sigmoid()
        )
        
        let learningRate: Float = 1.0
        
        for epoch in 1 ... 30 {
            var loss = Variable(value: 0)
            
            for (input, expected) in dataset {
                let pred = net.forward(input)[0]
                loss = loss + binaryCrossEntropy(expected: expected, actual: pred)
            }
            
            loss.zeroGradient()
            loss.backwards()
            
            for param in net.allParameters {
                param.value -= param.gradient * learningRate
            }
            
            if epoch % 10 == 0 {
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

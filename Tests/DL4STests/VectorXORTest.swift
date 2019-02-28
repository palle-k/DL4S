//
//  VectorXORTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 27.02.19.
//

import XCTest
@testable import DL4S

class VectorXORTest: XCTestCase {
    func testXOR() {
        
        let net = Sequential<Float>(
            Dense(inputFeatures: 2, outputFeatures: 6).asAny(),
            Sigmoid().asAny(),
            Dense(inputFeatures: 6, outputFeatures: 1).asAny(),
            Sigmoid().asAny()
        )
        
        let inputs = Vector<Float>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        let expectedOutputs = Vector<Float>([0, 1, 1, 0])
        
        let optimizer = SGDOptimizer(learningRate: 0.3, parameters: net.parameters)
        
        for epoch in 1 ... 1000 {
            let predictions = net.forward(inputs)
            let loss = binaryCrossEntropy(expected: expectedOutputs, actual: predictions)
            
            loss.zeroGradient()
            loss.backwards()
            
            optimizer.step()
            
            let lossValue = loss.item
            
            if epoch % 10 == 0 {
                print("AVG loss: \(lossValue / 4)")
            }
        }
        
//        let correctCount = dataset.lazy.filter { input, output -> Bool in
//            round(net.forward(input).view().item) == output.item
//        }.count
        
        let predictions = net.forward(inputs).view(as: -1)
        
        var correctCount = 0
        for i in 0 ..< 4 {
            if round(predictions[i].item) == expectedOutputs[i].item {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / 4
        print("Accuracy: \(accuracy)")
        
        XCTAssertEqual(accuracy, 1)
    }
}

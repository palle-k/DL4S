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
        
        let net = Sequential<Float, CPU>(
            Dense(inputFeatures: 2, outputFeatures: 6).asAny(),
            Sigmoid().asAny(),
            Dense(inputFeatures: 6, outputFeatures: 1).asAny(),
            Sigmoid().asAny()
        )
        
        let inputs = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        let expectedOutputs = Tensor<Float, CPU>([0, 1, 1, 0])
        
        let optimizer = MomentumOptimizer(parameters: net.parameters, learningRate: 0.05)
        // let optimizer = Adam(parameters: net.parameters, learningRate: 0.05)
        
        let epochs = 1000
        
        for epoch in 1 ... epochs {
            let predictions = net.forward(inputs)
            
            //print(predictions)
            
            let loss = binaryCrossEntropy(expected: expectedOutputs, actual: predictions)
            
            loss.zeroGradient()
            loss.backwards()
            
            optimizer.step()
            
            let lossValue = loss.item
            
            if epoch % 10 == 0 {
                print("[\(epoch)/\(epochs)] loss: \(lossValue / 4)")
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

//
//  XMNIST.swift
//  DL4STests
//
//  Created by Palle Klewitz on 15.10.19.
//

import XCTest
@testable import DL4S

class XMNIST: XCTestCase {

    func testExample() {
        let model = XUnion {
            XDense<Float, CPU>(inputSize: 28 * 28, outputSize: 500)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 500, outputSize: 200)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 200, outputSize: 50)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 50, outputSize: 10)
            XSoftmax<Float, CPU>()
        }
        var optimizer = XAdam(model: model, learningRate: 0.0001)
        
        let ((images, labels), _) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let epochs = 10_000
        let batchSize = 64
        
        for epoch in 1 ... epochs {
            let (realT, labelT) = Random.minibatch(from: images, labels: labels, count: batchSize)
            let input = XTensor(realT.view(as: [batchSize, 28 * 28]))
            let expected = XTensor(labelT)
            
            let predicted = optimizer.model(input)
            let loss = categoricalCrossEntropy(expected: expected, actual: predicted)
            
            let gradients = loss.gradients(of: optimizer.model.parameters)
            optimizer.update(along: gradients)
            
            if epoch.isMultiple(of: 100) {
                print("[\(epoch)/\(epochs)] loss: \(loss)")
            }
        }
    }

}

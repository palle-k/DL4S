//
//  XMNIST.swift
//  DL4STests
//
//  Created by Palle Klewitz on 15.10.19.
//  Copyright (c) 2019 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import XCTest
@testable import DL4S

struct SecondOrderOptimizer<Model: XLayer> {
    var model: Model
    private var keyPaths = Model.parameters
    
    init(model: Model) {
        self.model = model
    }
    
    mutating func update(along gradients: [XTensor<Model.Parameter, Model.Device>], secondOrder: [XTensor<Model.Parameter, Model.Device>]) {
        for i in keyPaths.indices {
            let path = keyPaths[i]
            let grad = gradients[i].detached()
            let hessDiag = secondOrder[i].detached()
            //let hessMag = sqrt(hessDiag * hessDiag)
            let update = hessDiag
            
            self.model[keyPath: path] -= update * grad
            self.model[keyPath: path].discardContext()
        }
    }
}

class XMNIST: XCTestCase {

    func testExample() {
        var model = XUnion {
            XDense<Float, CPU>(inputSize: 28 * 28, outputSize: 500)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 500, outputSize: 200)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 200, outputSize: 50)
            XRelu<Float, CPU>()
            XDense<Float, CPU>(inputSize: 50, outputSize: 10)
            XSoftmax<Float, CPU>()
        }
        model.tag = "Classifier"
        var optimizer = XAdam(model: model, learningRate: 0.001)
        
        let ((images, labels), _) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let epochs = 10_000
        let batchSize = 64
        let lambda: XTensor<Float, CPU> = 1
        
        for epoch in 1 ... epochs {
            let (realT, labelT) = Random.minibatch(from: images, labels: labels, count: batchSize)
            var input = XTensor(realT.view(as: [batchSize, 28 * 28]))
            input.requiresGradient = true
            let expected = XTensor(labelT)
            
            let predicted = optimizer.model(input)
            
            let targetLoss = categoricalCrossEntropy(expected: expected, actual: predicted)
            
            let inputGradient = targetLoss.gradients(of: [input], retainBackwardsGraph: true)[0]

            let partialPenaltyTerm = (inputGradient * inputGradient).reduceSum(along: [1])
            let gradientPenaltyLoss = lambda * partialPenaltyTerm.reduceMean()
            
            let loss = targetLoss + gradientPenaltyLoss
            let gradients = loss.gradients(of: optimizer.model.parameters)
            
            optimizer.update(along: gradients)
            
            if epoch.isMultiple(of: 100) {
                print("[\(epoch)/\(epochs)] loss: \(targetLoss), gp: \(gradientPenaltyLoss * 100) / 100")
            }
        }
    }
}

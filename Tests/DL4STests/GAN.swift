//
//  GAN.swift
//  DL4STests
//
//  Created by Palle Klewitz on 07.03.19.
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

import Foundation
@testable import DL4S
import XCTest


class GANTests: XCTestCase {
    func testGAN() {
        print("Loading images...")
        let ((images, labels_cat), _) = XMNIST.loadMNIST(from: "/Users/Palle/Developer/DL4S/", type: Float.self, device: CPU.self)

        let labels = labels_cat.oneHotEncoded(dim: 10, type: Float.self)
        
        print("Creating networks...")
        let latentSize = 20
        
        let gen = XSequential {
            XConcat<Float, CPU>()
            
            XDense<Float, CPU>(inputSize: latentSize + 10, outputSize: 200)
            XBatchNorm<Float, CPU>(inputSize: [200])
            XLeakyRelu<Float, CPU>(leakage: 0.2)
            XDropout<Float, CPU>(rate: 0.5)
            
            XDense<Float, CPU>(inputSize: 200, outputSize: 800)
            XBatchNorm<Float, CPU>(inputSize: [800])
            XLeakyRelu<Float, CPU>(leakage: 0.2)
            XDropout<Float, CPU>(rate: 0.5)
            
            XDense<Float, CPU>(inputSize: 800, outputSize: 28 * 28)
            XSigmoid<Float, CPU>()
        }
        
        let disc = XSequential {
            XConcat<Float, CPU>()
            
            XDense<Float, CPU>(inputSize: 28 * 28 + 10, outputSize: 400)
            XRelu<Float, CPU>()
            
            XDense<Float, CPU>(inputSize: 400, outputSize: 100)
            XRelu<Float, CPU>()
            
            XDense<Float, CPU>(inputSize: 100, outputSize: 1)
            XSigmoid<Float, CPU>()
        }
        
        var optimGen = XAdam(model: gen, learningRate: 0.0003)
        var optimDis = XAdam(model: disc, learningRate: 0.0003)
        
        let batchSize = 32
        let epochs = 50_000
        let regularization: Float = 0.0003
        
        print("Training...")
        
        for epoch in 1 ... epochs {
            let (real, realLabels) = Random.minibatch(from: images, labels: labels, count: batchSize)
            
            let genInputs = XTensor<Float, CPU>(normalDistributedWithShape: batchSize, latentSize)
            let genLabels = XTensor<Int32, CPU>(uniformlyDistributedWithShape: batchSize, min: 0, max: 9)
            
            let realResult = optimDis.model([real.view(as: [batchSize, -1]), realLabels])
            let gl = genLabels.oneHotEncoded(dim: 10, type: Float.self)
            
            let fakeGenerated = optimGen.model([genInputs, gl])
            let fakeResult = optimDis.model([fakeGenerated, gl])
            
            let dRegLoss = optimDis.model.parameters.map {($0 * $0).reduceMean() * XTensor(regularization)}.reduce(0, +)

            let discriminatorLoss = -mean(log(realResult)) - mean(log(1 - fakeResult)) + dRegLoss
            
            let discriminatorGrads = discriminatorLoss.gradients(of: optimDis.model.parameters)
            optimDis.update(along: discriminatorGrads)
            
            var generatorLoss = XTensor<Float, CPU>(0)

            for _ in 0 ..< 4 {
                let genInputs = XTensor<Float, CPU>(normalDistributedWithShape: batchSize, latentSize)
                let genLabels = XTensor<Int32, CPU>(uniformlyDistributedWithShape: batchSize, min: 0, max: 9)
                let gl = genLabels.oneHotEncoded(dim: 10, type: Float.self)
                
                let generated = optimGen.model([genInputs, gl])
                let genResult = optimDis.model([generated, gl])
                
                let gRegLoss = optimGen.model.parameters.map {($0 * $0).reduceMean() * XTensor(regularization)}.reduce(0, +)
                generatorLoss = -0.5 * mean(log(genResult)) + gRegLoss // heuristic non-saturating loss

                let generatorGradients = generatorLoss.gradients(of: optimGen.model.parameters)
                optimGen.update(along: generatorGradients)
            }
            
            if epoch % 100 == 0 {
                print(" [\(epoch)/\(epochs)] loss d: \(discriminatorLoss.item), g: \(generatorLoss.item)")
            }

            if epoch % 1000 == 0 {
                let genInputs = XTensor<Float, CPU>(normalDistributedWithShape: batchSize, latentSize)
                let genLabels = XTensor<Int32, CPU>(uniformlyDistributedWithShape: batchSize, min: 0, max: 9)
                let gl = genLabels.oneHotEncoded(dim: 10, type: Float.self)
                
                let genResult = optimGen.model([genInputs, gl])
                    .view(as: [-1, 1, 28, 28])
                
                for i in 0 ..< batchSize {
                    let slice = genResult[i].T.unsqueezed(at: 0)
                    guard let image = NSImage(slice), let imgData = image.tiffRepresentation else {
                        continue
                    }
                    guard let rep = NSBitmapImageRep.init(data: imgData) else {
                        continue
                    }
                    let png = rep.representation(using: .png, properties: [:])
                    try? png?.write(to: URL(fileURLWithPath: "/Users/Palle/Desktop/gan/gen_\(epoch)_\(i).png"))
                }
            }
        }
        
    }
}

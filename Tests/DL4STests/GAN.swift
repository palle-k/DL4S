//
//  GAN.swift
//  DL4STests
//
//  Created by Palle Klewitz on 07.03.19.
////  Copyright (c) 2019 - Palle Klewitz
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
import DL4S
import XCTest


class GANTests: XCTestCase {
    func testGAN() {
        print("Loading images...")
        let ((images, _), _) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        print("Creating networks...")
        
        let d1 = Dropout<Float, CPU>(rate: 0.5)
        let d2 = Dropout<Float, CPU>(rate: 0.5)
        
        let generator = Sequential<Float, CPU>(
            Dense(inputFeatures: 20, outputFeatures: 200).asAny(),
            Tanh().asAny(),
            d1.asAny(),
            Dense(inputFeatures: 200, outputFeatures: 800).asAny(),
            Tanh().asAny(),
            d2.asAny(),
            Dense(inputFeatures: 800, outputFeatures: 28 * 28).asAny(),
            Sigmoid().asAny(),
            Reshape(shape: 28, 28).asAny()
        )
        
        let discriminator = Sequential<Float, CPU>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 400).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 400, outputFeatures: 100).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 100, outputFeatures: 1).asAny(),
            Sigmoid().asAny()
        )
        
        let network = Sequential(generator.asAny(), discriminator.asAny())
        
        let optimGen = Adam(parameters: generator.trainableParameters, learningRate: 0.0003)
        let optimDis = Adam(parameters: discriminator.trainableParameters, learningRate: 0.0003)
        
        let batchSize = 32
        let epochs = 10_000
        let regularization: Float = 0.001
        
        let genInputs = Tensor<Float, CPU>(repeating: 0, shape: batchSize, 20)
        
        print("Training...")
        
        for epoch in 1 ... epochs {
            optimDis.zeroGradient()

            let real = Random.minibatch(from: images, count: batchSize)
            Random.fillNormal(genInputs)

            let realResult = discriminator.forward(real)
            let fakeResult = network.forward(genInputs)

            let dRegLoss = optimDis.parameters.map {l2loss($0, loss: regularization)}.reduce(0, +)
            let discriminatorLoss = -mean(log(realResult)) - mean(log(1 - fakeResult)) + dRegLoss

            discriminatorLoss.backwards()
            optimDis.step()

            var generatorLoss = Tensor<Float, CPU>(0)

            for _ in 0 ..< 4 {
                optimGen.zeroGradient()
                Random.fillNormal(genInputs)

                let genResult = network.forward(genInputs)
                
                let gRegLoss = optimDis.parameters.map {l2loss($0, loss: regularization)}.reduce(0, +)
                generatorLoss = -0.5 * mean(log(genResult)) + gRegLoss // heuristic non-saturating loss

                generatorLoss.backwards()
                optimGen.step()
            }

            if epoch % 100 == 0 {
                print(" [\(epoch)/\(epochs)] loss d: \(discriminatorLoss.item), g: \(generatorLoss.item)")
            }

            if epoch % 1000 == 0 {
                d1.isActive = false
                d2.isActive = false

                Random.fillNormal(genInputs)
                let genResult = generator.forward(genInputs)

                d1.isActive = true
                d2.isActive = true

                for i in 0 ..< batchSize {
                    let slice = genResult[i].T.unsqueeze(at: 0)
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

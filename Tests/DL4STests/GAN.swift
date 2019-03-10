//
//  GAN.swift
//  DL4STests
//
//  Created by Palle Klewitz on 07.03.19.
//

import Foundation
import DL4S
import XCTest


class GANTests: XCTestCase {
    func testGAN() {
        print("Loading images...")
        let ((images, _), _) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        print("Creating networks...")
        
        let d1 = Dropout<Float>(rate: 0.5)
        let d2 = Dropout<Float>(rate: 0.5)
        
        let generator = Sequential<Float>(
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
        
        let discriminator = Sequential<Float>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 400).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 400, outputFeatures: 100).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 100, outputFeatures: 1).asAny(),
            Sigmoid().asAny()
        )
        
        let network = Sequential(generator.asAny(), discriminator.asAny())
        
        let optimGen = Adam(parameters: generator.parameters, learningRate: 0.001)
        let optimDis = Adam(parameters: discriminator.parameters, learningRate: 0.001)
        
        let batchSize = 16
        let epochs = 10_000
        
        let genInputs = Tensor<Float>(repeating: 0, shape: batchSize, 20)
        
        print("Training...")
        
        for epoch in 1 ... epochs {
            optimDis.zeroGradient()

            let real = Random.minibatch(from: images, count: batchSize)
            Random.fillNormal(genInputs)

            let realResult = discriminator.forward(real)
            let fakeResult = network.forward(genInputs)

            let discriminatorLoss = -sum(log(realResult)) - sum(log(1 - fakeResult))

            discriminatorLoss.backwards()
            optimDis.step()

            var generatorLoss = Tensor<Float>(0)

            for _ in 0 ..< 4 {
                optimGen.zeroGradient()
                Random.fillNormal(genInputs)

                let genResult = network.forward(genInputs)
                generatorLoss = -0.5 * sum(log(genResult)) // heuristic non-saturating loss
//                generatorLoss = sum(log(1 - genResult)) // "-" is left out as the goal is to ascend the stochastic gradient

                generatorLoss.backwards()
                optimGen.step()
            }

            if epoch % 100 == 0 {
                print(" [\(epoch)/\(epochs)] loss d: \(discriminatorLoss.item), g: \(generatorLoss.item)")
            }

            if epoch % 1000 == 0 {
                d1.isTrainingMode = false
                d2.isTrainingMode = false

                Random.fillNormal(genInputs)
                let genResult = generator.forward(genInputs)

                d1.isTrainingMode = true
                d2.isTrainingMode = true

                for i in 0 ..< batchSize {
                    let slice = genResult[i].unsqueeze(at: 0)
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

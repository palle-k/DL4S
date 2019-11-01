//
//  WGANGPTests.swift
//  DL4S
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

import Foundation
import XCTest
@testable import DL4S

struct Critic<Element: RandomizableType, Device: DeviceType>: LayerType {
    var parameters: [Tensor<Element, Device>] {
        convolutions.parameters + classifier.parameters
    }
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            convolutions.parameterPaths.map((\Self.convolutions).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:))
        ].joined())
    }
    
    var convolutions = Sequential {
        Convolution2D<Element, Device>(inputChannels: 1, outputChannels: 6, kernelSize: (3, 3))  // 28x28
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 14x14
        
        Convolution2D<Element, Device>(inputChannels: 6, outputChannels: 12, kernelSize: (3, 3), padding: 2)  // 16x16
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 8x8
        
        Convolution2D<Element, Device>(inputChannels: 12, outputChannels: 16, kernelSize: (3, 3))  // 8x8
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 4x4
        Flatten<Element, Device>() // 256
    }
    
    var classifier = Sequential {
        Concat<Element, Device>()
        
        Dense<Element, Device>(inputSize: 256 + 10, outputSize: 128)
        Relu<Element, Device>()
        
        Dense<Element, Device>(inputSize: 128, outputSize: 1)
    }
    
    init() {
        convolutions.tag = "Convolutions"
        classifier.tag = "Classifier"
    }
    
    func callAsFunction(_ inputs: (Tensor<Element, Device>, Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Critic") {
            let conv = convolutions(inputs.0)
            return classifier([conv, inputs.1])
        }
    }
}

class WGANGPTests: XCTestCase {
    func testWGANGP() {
        var generator = Sequential {
            Dense<Float, CPU>(inputSize: 50 + 10, outputSize: 512)
            BatchNorm<Float, CPU>(inputSize: [512])
            LeakyRelu<Float, CPU>(leakage: 0.2)
            
            Reshape<Float, CPU>(outputShape: [8, 8, 8])
            
            TransposedConvolution2D<Float, CPU>(inputChannels: 8, outputChannels: 6, kernelSize: (3, 3), inset: 1, stride: 2)
            BatchNorm<Float, CPU>(inputSize: [6, 15, 15])
            LeakyRelu<Float, CPU>(leakage: 0.2)
            
            TransposedConvolution2D<Float, CPU>(inputChannels: 6, outputChannels: 3, kernelSize: (3, 3), inset: 1, stride: 2)
            BatchNorm<Float, CPU>(inputSize: [3, 29, 29])
            LeakyRelu<Float, CPU>(leakage: 0.2)
            
            Convolution2D<Float, CPU>(inputChannels: 3, outputChannels: 1, kernelSize: (2, 2), padding: 0, stride: 1)
            Sigmoid<Float, CPU>()
        }
        generator.tag = "Generator"
        
        let critic = Critic<Float, CPU>()
        
        print("Loading images...")
        let ((_images, labels_cat), _) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: CPU.self)
        
        let images = _images.view(as: [-1, 1, 28, 28])
        let labels = labels_cat.oneHotEncoded(dim: 10, type: Float.self)
        
        print("Creating networks...")
        
        var optimGen = Adam(model: generator, learningRate: 0.0003, beta1: 0.0, beta2: 0.9)
        var optimCrit = Adam(model: critic, learningRate: 0.0003, beta1: 0.0, beta2: 0.9)
        
        let batchSize = 32
        let epochs = 20_000
        let n_critic = 5
        let n_gen = 1
        let lambda = Tensor<Float, CPU>(10)
        
        print("Training...")
        
        for epoch in 1 ... epochs {
            var lastCriticDiscriminationLoss: Tensor<Float, CPU> = 0
            var lastGradientPenaltyLoss: Tensor<Float, CPU> = 0
            
            for _ in 0 ..< n_critic {
                let (real, realLabels) = Random.minibatch(from: images, labels: labels, count: batchSize)
                
                let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
                let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
                    .oneHotEncoded(dim: 10, type: Float.self)
                
                let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
                
                let fakeGenerated = optimGen.model(genInputs)
                let eps = Tensor<Float, CPU>(Float.random(in: 0 ... 1))
                let mixed = real * eps + fakeGenerated * (1 - eps)
                
                let fakeDiscriminated = optimCrit.model((mixed, genLabelInput))
                let realDiscriminated = optimCrit.model((real, realLabels))
                
                let criticDiscriminationLoss = OperationGroup.capture(named: "CriticDiscriminationLoss") {
                    fakeDiscriminated.reduceMean() - realDiscriminated.reduceMean()
                }
                
                let gradientPenaltyLoss = OperationGroup.capture(named: "GradientPenaltyLoss") { () -> Tensor<Float, CPU> in
                    let criticInputGrads = fakeDiscriminated.gradients(of: [mixed, genLabelInput], retainBackwardsGraph: true)
                    let fakeGeneratedGrad = Tensor(stacking: [criticInputGrads[0].view(as: batchSize, -1), criticInputGrads[1].view(as: batchSize, -1)], along: 1)
                    let partialPenaltyTerm = (fakeGeneratedGrad * fakeGeneratedGrad).reduceSum(along: [1]).sqrt() - 1
                    let gradientPenaltyLoss = lambda * (partialPenaltyTerm * partialPenaltyTerm).reduceMean()
                    return gradientPenaltyLoss
                }
                
                let criticLoss = criticDiscriminationLoss + gradientPenaltyLoss
                let criticGradients = criticLoss.gradients(of: optimCrit.model.parameters)
                
                optimCrit.update(along: criticGradients)
                
                lastCriticDiscriminationLoss = criticDiscriminationLoss.detached()
                lastGradientPenaltyLoss = gradientPenaltyLoss.detached()
            }

            var lastGeneratorLoss: Tensor<Float, CPU> = 0
            
            for _ in 0 ..< n_gen {
                let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
                let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
                    .oneHotEncoded(dim: 10, type: Float.self)
                
                let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
                
                let fakeGenerated = optimGen.model(genInputs)
                let fakeDiscriminated = optimCrit.model((fakeGenerated, genLabelInput))
                let generatorLoss = -fakeDiscriminated.reduceMean()
                
                lastGeneratorLoss = generatorLoss
                let generatorGradients = generatorLoss.gradients(of: optimGen.model.parameters)

                optimGen.update(along: generatorGradients)
            }
            

            if epoch.isMultiple(of: 10) {
                print(" [\(epoch)/\(epochs)] loss c: \(lastCriticDiscriminationLoss), gp: \(lastGradientPenaltyLoss), g: \(lastGeneratorLoss)")
            }
            
            #if canImport(AppKit)
            if epoch.isMultiple(of: 1000) {
                let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
                let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
                    .oneHotEncoded(dim: 10, type: Float.self)
                
                let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
                
                let fakeGenerated = optimGen.model(genInputs).view(as: [-1, 28, 28])
                
                for i in 0 ..< 32 {
                    let slice = fakeGenerated[i].permuted(to: [1, 0]).unsqueezed(at: 0)
                    guard let image = NSImage(slice), let imgData = image.tiffRepresentation else {
                        continue
                    }
                    guard let rep = NSBitmapImageRep(data: imgData) else {
                        continue
                    }
                    let png = rep.representation(using: .png, properties: [:])
                    try? png?.write(to: URL(fileURLWithPath: "/Users/Palle/Desktop/wgan_gp_conv/gen_\(epoch)_\(i).png"))
                }
            }
            #endif
        }
    }
}

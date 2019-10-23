//
//  MNISTTests.swift
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

#if os(Linux)
let MNIST_PATH = "/home/palle/Downloads/MNIST/"
#else
let MNIST_PATH = "/Users/Palle/Developer/DL4S/"
#endif

class MNISTTests: XCTestCase {
    static func loadMNIST<Element, Device>(from path: String, type: Element.Type = Element.self, device: Device.Type = Device.self) -> (train: (Tensor<Element, Device>, Tensor<Int32, Device>), test: (Tensor<Element, Device>, Tensor<Int32, Device>)) {
        do {
            let trainingData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-images.idx3-ubyte"))
            let trainingLabelData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-labels.idx1-ubyte"))
            let testingData = try! Data(contentsOf: URL(fileURLWithPath: path + "t10k-images.idx3-ubyte"))
            let testingLabelData = try! Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels.idx1-ubyte"))
            
            let trainImages = Tensor<Element, Device>(trainingData.dropFirst(16).prefix(28 * 28 * 60_000).map(Element.init)) / 256
            let testImages = Tensor<Element, Device>(testingData.dropFirst(16).prefix(28 * 28 * 10_000).map(Element.init)) / 256
            
            let trainLabels = Tensor<Int32, Device>(trainingLabelData.dropFirst(8).prefix(60_000).map(Int32.init))
            let testLabels = Tensor<Int32, Device>(testingLabelData.dropFirst(8).prefix(10_000).map(Int32.init))
            
            return (
                train: (trainImages.view(as: [-1, 1, 28, 28]), trainLabels),
                test: (testImages.view(as: [-1, 1, 28, 28]), testLabels)
            )
        } catch let error {
            print(error)
            fatalError()
        }
        
    }
    
    func testConvNet() {
        var model = Sequential {
            Convolution2D<Float, CPU>(inputChannels: 1, outputChannels: 6, kernelSize: (5, 5), padding: 0)
            LayerNorm<Float, CPU>(inputSize: [4, 24, 24])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            Convolution2D<Float, CPU>(inputChannels: 6, outputChannels: 16, kernelSize: (5, 5), padding: 0)
            LayerNorm<Float, CPU>(inputSize: [16, 8, 8])
            Relu<Float, CPU>()
            MaxPool2D<Float, CPU>(windowSize: 2, stride: 2)
            Flatten<Float, CPU>()
            Dense<Float, CPU>(inputSize: 256, outputSize: 120)
            LayerNorm<Float, CPU>(inputSize: [120])
            Relu<Float, CPU>()
            Dense<Float, CPU>(inputSize: 120, outputSize: 10)
            Softmax<Float, CPU>()
        }
        
        model.tag = "Classifier"
        var optimizer = Adam(model: model, learningRate: 0.001)
        
        let ((images, labels), (imagesVal, labelsVal)) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: CPU.self)
        
        let epochs = 10_000
        let batchSize = 128
        
        for epoch in 1 ... epochs {
            let (input, target) = Random.minibatch(from: images, labels: labels, count: batchSize)

            let predicted = optimizer.model(input.view(as: [batchSize, 1, 28, 28]))
            let loss = categoricalCrossEntropy(expected: target, actual: predicted)
            
            let gradients = loss.gradients(of: optimizer.model.parameters)
            
            optimizer.update(along: gradients)
            
            if epoch.isMultiple(of: 10) {
                print("[\(epoch)/\(epochs)] loss: \(loss)")
            }
        }
        
        var correctCount = 0
        
        for i in 0 ..< imagesVal.shape[0] {
            let x = imagesVal[i].view(as: [1, 1, 28, 28])
            let pred = optimizer.model(x).squeezed().argmax()
            let actual = Int(labelsVal[i].item)
            
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(imagesVal.shape[0])
        
        print("accuracy: \(accuracy * 100)%")
    }
    
    func testFCN() {
        var model = Sequential {
            Dense<Float, CPU>(inputSize: 28 * 28, outputSize: 500)
            Relu<Float, CPU>()
            
            Dense<Float, CPU>(inputSize: 500, outputSize: 300)
            Relu<Float, CPU>()

            Dense<Float, CPU>(inputSize: 300, outputSize: 10)
            Softmax<Float, CPU>()
        }
        
        model.tag = "Classifier"
        var optimizer = Adam(model: model, learningRate: 0.001)
        
        let ((images, labels), (imagesVal, labelsVal)) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: CPU.self)
        
        let epochs = 10_000
        let batchSize = 128
        
        for epoch in 1 ... epochs {
            let (input, target) = Random.minibatch(from: images, labels: labels, count: batchSize)
            
            let predicted = optimizer.model(input.view(as: [batchSize, 28 * 28]))
            let loss = categoricalCrossEntropy(expected: target, actual: predicted)
            
            let gradients = loss.gradients(of: optimizer.model.parameters)
            optimizer.update(along: gradients)
            
            if epoch.isMultiple(of: 100) {
                print("[\(epoch)/\(epochs)] loss: \(loss)")
            }
        }
        
        var correctCount = 0
        
        for i in 0 ..< imagesVal.shape[0] {
            let x = imagesVal[i].view(as: [1, 28 * 28])
            let y = optimizer.model(x).squeezed()
            let pred = y.argmax()
            
            let actual = Int(labelsVal[i].item)
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(imagesVal.shape[0])
        
        print("accuracy: \(accuracy * 100)%")
    }
    
    func testGRU() {
        let ((images, labels), (imagesVal, labelsVal)) = MNISTTests.loadMNIST(from: MNIST_PATH, type: Float.self, device: CPU.self)
        
        print("Loaded images")

        var model = Sequential {
            GRU<Float, CPU>(inputSize: 28, hiddenSize: 128, direction: .forward)
            Lambda<GRU<Float, CPU>.Outputs, Tensor<Float, CPU>, Float, CPU> { inputs in
                inputs.0
            }
            Dense<Float, CPU>(inputSize: 128, outputSize: 10)
            Softmax<Float, CPU>()
        }
        model.tag = "Classifier"
        
        let epochs = 10_000
        let batchSize = 128
        
        var optimizer = Adam(model: model, learningRate: 0.001)

        print("Created model and optimizer")
        
        let queue = Queue<(Tensor<Float, CPU>, Tensor<Int32, CPU>)>(maxLength: 16)
        let workers = 1

        for i in 0 ..< workers {
            DispatchQueue.global().async {
                print("starting worker \(i)")
                while !queue.isStopped {
                    let (batch, expected) = Random.minibatch(from: images, labels: labels, count: batchSize)
                    let x = batch.view(as: [-1, 28, 28]).permuted(to: [1, 0, 2])
                    queue.enqueue((x, expected))
                }
                print("stopping worker \(i)")
            }
        }
        
        var bar = ProgressBar<Float>(totalUnitCount: epochs, formatUserInfo: {"loss: \($0)"}, label: "training")
        
        for _ in 1 ... epochs {
            let (input, target) = queue.dequeue()!
            
            let predicted = optimizer.model(input)
            let loss = categoricalCrossEntropy(expected: target, actual: predicted)
            
            let gradients = loss.gradients(of: optimizer.model.parameters)
            optimizer.update(along: gradients)
            
            bar.next(userInfo: loss.item)
        }
        
        bar.complete()
        queue.stop()
        
        var correctCount = 0
        
        for i in 0 ..< imagesVal.shape[0] {
            let x = imagesVal[i]
                .view(as: [-1, 28, 28])
                .permuted(to: [1, 0, 2])
            let y = optimizer.model(x).squeezed()
            let pred = y.argmax()
            
            let actual = Int(labelsVal[i].item)
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(imagesVal.shape[0])
        
        print("accuracy: \(accuracy * 100)%")
    }
}

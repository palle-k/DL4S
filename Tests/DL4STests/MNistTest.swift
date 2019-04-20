//
//  MNistTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 28.02.19.
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

class MNistTest: XCTestCase {
    static func readSamples(from bytes: [UInt8], labels: [UInt8], count: Int) -> (Tensor<Float, CPU>, Tensor<Int32, CPU>) {
        let imageOffset = 16
        let labelOffset = 8
        
        let imageWidth = 28
        let imageHeight = 28
        
        var samples: [[Float]] = []
        var labelVectors: [Int32] = []
        
        
        for i in 0 ..< count {
            let offset = imageOffset + imageWidth * imageHeight * i
            let pixelData = bytes[offset ..< (offset + imageWidth * imageHeight)]
                .map{Float($0)/256}
            
            if pixelData.contains(where: {$0.isNaN}) {
                fatalError()
            }
            
            let label = Int(labels[labelOffset + i])
            
            //let sampleMatrix = Matrix3(values: pixelData, width: imageWidth, height: imageHeight, depth: 1)
            //let sampleMatrix = Tensor<Float, CPU>(pixelData, shape: imageHeight, imageWidth)
            //let expectedValue = Tensor<Int32, CPU>(Int32(label))
            
            var e = [Float](repeating: 0, count: 10)
            e[label] = Float(1)
            
            samples.append(pixelData)
            labelVectors.append(Int32(label))
        }
        
        return (Tensor(Array(samples.joined()), shape: samples.count, imageWidth, imageHeight), Tensor(labelVectors))
    }
    
    static func images(from path: String, maxCount: Int? = nil) -> ((Tensor<Float, CPU>, Tensor<Int32, CPU>), (Tensor<Float, CPU>, Tensor<Int32, CPU>)) {
        guard
            let trainingData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-images.idx3-ubyte")),
            let trainingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-labels.idx1-ubyte")),
            let testingData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-images.idx3-ubyte")),
            let testingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels.idx1-ubyte"))
        else {
            fatalError("Data not found")
        }
        
        let trainingBytes = Array<UInt8>(trainingData)
        let trainingLabels = Array<UInt8>(trainingLabelData)
        let testingBytes = Array<UInt8>(testingData)
        let testingLabels = Array<UInt8>(testingLabelData)
        
        let trainingSampleCount = maxCount ?? 60_000
        let testingSampleCount = maxCount ?? 10_000
        
        let trainingSamples = readSamples(from: trainingBytes, labels: trainingLabels, count: trainingSampleCount)
        let testingSamples = readSamples(from: testingBytes, labels: testingLabels, count: testingSampleCount)
        
        return (trainingSamples,testingSamples)
    }
    
    func testMNist() {
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float, CPU>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 5_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            let (batch, expected) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)

            let y_pred = model(batch)
            let y_true = expected
            
            let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
            
            loss.backwards()
            optimizer.step()
            
            if epoch % 100 == 0 {
                let avgLoss = loss.item
                print("[\(epoch)/\(epochs)] loss: \(avgLoss)")
            }
        }
        
        var correctCount = 0
        
        for i in 0 ..< ds_val.0.shape[0] {
            let x = ds_val.0[i].unsqueeze(at: 0)
            let pred = argmax(model(x).squeeze())
            let actual = Int(ds_val.1[i].item)
            
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(ds_val.0.shape[0])
        
        print("Accuracy: \(accuracy)")
        
        // try? model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_params.json"))
    }
    
    func testMNistConvnet() {
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float, CPU>(
            Conv2D(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 0).asAny(), // 4x24x24
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 4x12x12
            Conv2D(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0).asAny(), // 16x8x8
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 16x4x4
            Flatten().asAny(), // 256
            Dense(inputFeatures: 256, outputFeatures: 120).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 120, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 5_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            let (batch, expected) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)
            
            let x = batch.unsqueeze(at: 1)
            let y_pred = model(x)
            let y_true = expected
            
            let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
            
            loss.backwards()
            optimizer.step()
            
            if epoch % 10 == 0 {
                let avgLoss = loss.item
                print("[\(epoch)/\(epochs)] loss: \(avgLoss)")
            }
        }
        
        var correctCount = 0
        
        for i in 0 ..< ds_val.0.shape[0] {
            let x = ds_val.0[i].view(as: 1, 1, 28, 28)
            let pred = argmax(model(x).squeeze())
            let actual = Int(ds_val.1[i].item)
            
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(ds_val.0.shape[0])
        
        print("Accuracy: \(accuracy)")
        
        try! model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_lenet.json"))
    }
    
    func testMNistLstm() {
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float, CPU>(
            GRU(inputSize: 28, hiddenSize: 128).asAny(),
            Dense(inputFeatures: 128, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 10_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)
        
        print("Training...")
        
        let queue = Queue<(Tensor<Float, CPU>, Tensor<Int32, CPU>)>(maxLength: 16)
        let workers = 1

        for i in 0 ..< workers {
            DispatchQueue.global().async {
                print("starting worker \(i)")
                while !queue.isStopped {
                    let (batch, expected) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)
                    let x = batch.permuted(to: 1, 0, 2)
                    queue.enqueue((x, expected))
                }
                print("stopping worker \(i)")
            }
        }
        
        var bar = ProgressBar<Float>(totalUnitCount: epochs, formatUserInfo: {"loss: \($0)"}, label: "training")
        
        for _ in 1 ... epochs {
            optimizer.zeroGradient()
            
            let (x, y_true) = queue.dequeue()!
//            let (batch, y_true) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)
//            let x = batch.permuted(to: 1, 0, 2)

            let y_pred = model(x)
            let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
            loss.backwards()
            
            optimizer.step()
            
            bar.next(userInfo: loss.item)
        }
        bar.complete()
        
        // queue.stop()
        
        var correctCount = 0
        
        for i in 0 ..< ds_val.0.shape[0] {
            let x = ds_val.0[i].unsqueeze(at: 1)
            let pred = argmax(model.forward(x).squeeze())
            let actual = ds_val.1[i].item
            
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(ds_val.0.shape[0])
        
        print("Accuracy: \(accuracy)")
        
        try? model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_gru_params3.json"))
    }
    
    func testGenerative() throws {
        let model = Sequential<Float, CPU>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        try model.loadWeights(from: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_params.json"))
        
        let input = Tensor<Float, CPU>(repeating: 0, shape: [1, 28, 28], requiresGradient: true)
        let expected = Tensor<Int32, CPU>([5])
        
        Random.fill(input, a: 0, b: 1)
        
        let optimizer = Adam(parameters: [input], learningRate: 0.001)
        
        let epochs = 5000
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            
            let pred = model.forward(input)
            let loss = categoricalCrossEntropy(expected: expected, actual: pred) + sum((input - 0.5) * (input - 0.5)) * 0.00001
            loss.backwards()
            
            optimizer.step()
            
            if epoch % 100 == 0 {
                let avgLoss = loss.item
                print("[\(epoch)/\(epochs)] loss: \(avgLoss)")
            }
        }
        
        guard let image = NSImage(input), let imgData = image.tiffRepresentation else {
            fatalError()
        }
        guard let rep = NSBitmapImageRep.init(data: imgData) else {
            fatalError()
        }
        guard let png = rep.representation(using: .png, properties: [:]) else {
            fatalError()
        }
        try png.write(to: URL(fileURLWithPath: "/Users/Palle/Desktop/input.png"))
    }
}


class ResidualBlock<Element: RandomizableType, Device: DeviceType>: Layer {
    let conv1: Conv2D<Element, Device>
    let conv2: Conv2D<Element, Device>
    let bn1: BatchNorm<Element, Device>
    let bn2: BatchNorm<Element, Device>
    
    var isTrainable: Bool = true
    
    var parameters: [Tensor<Element, Device>] {
        return Array([conv1.parameters, conv2.parameters, bn1.parameters, bn2.parameters].joined())
    }
    
    init(inputShape: [Int]) {
        conv1 = Conv2D(inputChannels: inputShape[0], outputChannels: inputShape[0], kernelSize: 3)
        conv2 = Conv2D(inputChannels: inputShape[0], outputChannels: inputShape[0], kernelSize: 3)
        bn1 = BatchNorm(inputSize: inputShape)
        bn2 = BatchNorm(inputSize: inputShape)
    }
    
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        let res = x
        
        x = conv1(x)
        x = bn1(x)
        x = relu(x)
        x = conv2(x)
        x = bn2(x)
        x = x + res
        x = relu(x)
        
        return x
    }
}

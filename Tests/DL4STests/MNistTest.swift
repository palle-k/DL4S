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
    typealias Device = GPU
    
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
            //let sampleMatrix = Tensor<Float, Device>(pixelData, shape: imageHeight, imageWidth)
            //let expectedValue = Tensor<Int32, Device>(Int32(label))
            
            var e = [Float](repeating: 0, count: 10)
            e[label] = Float(1)
            
            samples.append(pixelData)
            labelVectors.append(Int32(label))
        }
        
        return (Tensor(Array(samples.joined()), shape: samples.count, imageWidth, imageHeight), Tensor(labelVectors))
    }
    
    static func images(from path: String, maxCount: Int? = nil) -> ((Tensor<Float, CPU>, Tensor<Int32, CPU>), (Tensor<Float, CPU>, Tensor<Int32, CPU>)) {
        let trainingData: Data
        let trainingLabelData: Data
        let testingData: Data
        let testingLabelData: Data
        
        do {
            trainingData = try Data(contentsOf: URL(fileURLWithPath: path + "train-images-idx3-ubyte"))
            trainingLabelData = try Data(contentsOf: URL(fileURLWithPath: path + "train-labels-idx1-ubyte"))
            testingData = try Data(contentsOf: URL(fileURLWithPath: path + "t10k-images-idx3-ubyte"))
            testingLabelData = try Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels-idx1-ubyte"))
        } catch let error {
            fatalError("Data not found \(error)")
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
        let (ds_train, ds_val) = { () -> ((Tensor<Float, Device>, Tensor<Int32, Device>), (Tensor<Float, Device>, Tensor<Int32, Device>)) in
            let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Developer/DL4S/")
            return ((ds_train.0.copied(to: Device.self), ds_train.1.copied(to: Device.self)), (ds_val.0.copied(to: Device.self), ds_val.1.copied(to: Device.self)))
        }()
        
        let bn1 = BatchNorm<Float, Device>(inputSize: [500])
        let bn2 = BatchNorm<Float, Device>(inputSize: [300])
        
        let model = Sequential<Float, Device>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            // bn1.asAny(),
            // LayerNorm(inputSize: [500]).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            // bn2.asAny(),
            // LayerNorm(inputSize: [500]).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 2_000
        let batchSize = 128
        
        //              1k iterations @ bs 512    |  100 iterations @ bs 512  |  20 iterations @ bs 512   |  5 iterations @ bs 512
        // no norm:     98.28% acc, loss: 0.0141  |  95.06% acc, loss: 0.156  |  89.67% acc, loss: 0.487  |  76.11% acc, loss: 1.293
        // batch norm:  98.06% acc, loss: 0.0084  |  96.39% acc, loss: 0.121  |  91.79% acc, loss: 0.291  |  74.70% acc, loss: 0.824
        // layer norm:  98.12% acc, loss: 0.0083  |  95.90% acc, loss: 0.142  |  90.58% acc, loss: 0.385  |  79.88% acc, loss: 0.987
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.0005)
        // let optimizer = Adagrad(parameters: model.trainableParameters, learningRate: 0.001)
        // let optimizer = GradientDescent(parameters: model.trainableParameters, learningRate: 0.001)
        // let optimizer = Momentum(parameters: model.trainableParameters, learningRate: 0.001)
        // let optimizer = RMSProp(parameters: model.trainableParameters, learningRate: 0.001, gamma: 0.9)
        // let optimizer = Adadelta(parameters: model.trainableParameters, learningRate: 0.001, gamma: 0.9, epsilon: 1e-8)
        
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
        
        bn1.isTraining = false
        bn2.isTraining = false
        
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
        typealias Device = CPU
        
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Developer/DL4S/")
        
        /*
         With Layer Norm:
         [10/10000] loss: 1.8275247
         [100/10000] loss: 0.30241546
         
         [10/10000] loss: 1.9961276
         [100/10000] loss: 0.35611022
         
         Without:
         [10/10000] loss: 2.0360816
         [100/10000] loss: 0.3532066
         
         [10/10000] loss: 2.1360865
         [100/10000] loss: 0.37744054
        */
        
        let model = Sequential<Float, CPU>(
            Conv2D(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 0).asAny(), // 4x24x24
            LayerNorm(inputSize: [4, 24, 24]).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 4x12x12
            Conv2D(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0).asAny(), // 16x8x8
            LayerNorm(inputSize: [16, 8, 8]).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 16x4x4
            Flatten().asAny(), // 256
            Dense(inputFeatures: 256, outputFeatures: 120).asAny(),
            LayerNorm(inputSize: [120]).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 120, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 10_000
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
    
    func testMNistResnet() {
        typealias Device = CPU
        
        var (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        ds_train.0 = pad(ds_train.0, padding: [0, 2, 2])
        ds_val.0 = pad(ds_val.0, padding: [0, 2, 2])
        
        let epochs = 500
        let batchSize = 32
        
        let model = ResNet<Float, Device>(inputShape: [1, 32, 32], classCount: 10)
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
            
            if epoch % 1 == 0 {
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
        try! model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_resnet18.json"))
    }
    
    func testMNistLstm() {
        typealias Device = CPU
        
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float, Device>(
            GRU(inputSize: 28, hiddenSize: 128).asAny(),
            //LSTM(inputSize: 28, hiddenSize: 128).asAny(),
            //BasicRNN(inputSize: 28, hiddenSize: 128).asAny(),
            Dense(inputFeatures: 128, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 10_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)
        
        print("Training...")
        
        let queue = Queue<(Tensor<Float, Device>, Tensor<Int32, Device>)>(maxLength: 16)
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

            let y_pred = model(x)
            let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
            loss.backwards()
            
            optimizer.step()
            
            bar.next(userInfo: loss.item)
        }
        bar.complete()
        
        queue.stop()
        
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
    
    func testMNISTBiRNN() {
        typealias Device = CPU
        
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float, Device>(
            Bidirectional(forwardLayer: GRU(inputSize: 28, hiddenSize: 64, direction: .forward), backwardLayer: GRU(inputSize: 28, hiddenSize: 64, direction: .backward)).asAny(),
            Dense(inputFeatures: 128, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 5_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.trainableParameters, learningRate: 0.001)
        
        print("Training...")
        
        let queue = Queue<(Tensor<Float, Device>, Tensor<Int32, Device>)>(maxLength: 16)
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

            let y_pred = model(x)
            let loss = categoricalCrossEntropy(expected: y_true, actual: y_pred)
            loss.backwards()
            
            optimizer.step()
            
            bar.next(userInfo: loss.item)
        }
        bar.complete()
        
        queue.stop()
        
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
    }
    
    func testGenerative() throws {
        typealias Device = CPU
        
        let model = Sequential<Float, Device>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        try model.loadWeights(from: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_params.json"))
        
        let input = Tensor<Float, Device>(repeating: 0, shape: [1, 28, 28], requiresGradient: true)
        let expected = Tensor<Int32, Device>([5])
        
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
    
    func testIterativeAdversarial() throws {
        let model = Sequential<Float, CPU>(
            Conv2D(inputChannels: 1, outputChannels: 6, kernelSize: 5, padding: 0).asAny(), // 4x24x24
            LayerNorm(inputSize: [4, 24, 24]).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 4x12x12
            Conv2D(inputChannels: 6, outputChannels: 16, kernelSize: 5, padding: 0).asAny(), // 16x8x8
            LayerNorm(inputSize: [16, 8, 8]).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 2, stride: 2).asAny(), // 16x4x4
            Flatten().asAny(), // 256
            Dense(inputFeatures: 256, outputFeatures: 120).asAny(),
            LayerNorm(inputSize: [120]).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 120, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        try model.loadWeights(from: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_lenet.json"))
        
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Developer/DL4S/")
        
        // [1, 28, 28], [1]
        let (sample, target) = Random.minibatch(from: ds_val.0, labels: ds_val.1, count: 1)
        
        save(tensor: sample.permuted(to: 0, 2, 1), to: URL(fileURLWithPath: "/Users/Palle/Desktop/sample_orig.png"))
        
        let targetIndex = target.squeeze().item
        print("Expected index: \(target)")
        let retargeted = Tensor<Int32, CPU>((target.squeeze().item + 1) % 10).view(as: -1) // batchSize
        print("Adversarial target: \(retargeted)")
        
        let epochs = 3000
        
        let x = sample.unsqueeze(at: 1) // [1, 1, 28, 28]
        let y_adv = retargeted
        let noise = Tensor<Float, CPU>(repeating: 0, shape: sample.shape)
        Random.fill(noise, a: -0.001, b: 0.001)
        noise.requiresGradient = true
        
        let optimizer = Adam(parameters: [noise], learningRate: 0.001)
        
        for epoch in 0 ..< epochs {
            optimizer.zeroGradient()
            let y_act = model(x + noise)
            
            let target_loss = categoricalCrossEntropy(expected: y_adv, actual: y_act)
            let decay_loss = l2loss(noise, loss: 1000)
            
            let loss = target_loss + decay_loss
            loss.backwards()
            optimizer.step()
            
            if epoch.isMultiple(of: 10) {
                let prediction = argmax(y_act.squeeze())
                print("[\(epoch)/\(epochs)]: loss: \(loss) | target: \(target_loss) | decay: \(decay_loss) | \(prediction) (\(y_act[0, prediction]) confidence)")
            }
        }
        
        save(tensor: (sample + noise[0]).permuted(to: 0, 2, 1), to: URL(fileURLWithPath: "/Users/Palle/Desktop/sample_adv.png"))
    }
}

func sign<Element, Device>(_ x: Tensor<Element, Device>) -> Tensor<Element, Device> {
    heaviside(x) * 2 - 1
}

func save<Element, Device>(tensor: Tensor<Element, Device>, to url: URL) {
    guard let image = NSImage(tensor), let imgData = image.tiffRepresentation else {
        return
    }
    guard let rep = NSBitmapImageRep.init(data: imgData) else {
        return
    }
    let png = rep.representation(using: .png, properties: [:])
    try? png?.write(to: url)
}

//
//  MNistTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 28.02.19.
//

import XCTest
@testable import DL4S

class MNistTest: XCTestCase {
    static func readSamples(from bytes: [UInt8], labels: [UInt8], count: Int) -> (Tensor<Float>, Tensor<Int32>) {
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
            //let sampleMatrix = Tensor<Float>(pixelData, shape: imageHeight, imageWidth)
            //let expectedValue = Tensor<Int32>(Int32(label))
            
            var e = [Float](repeating: 0, count: 10)
            e[label] = Float(1)
            
            samples.append(pixelData)
            labelVectors.append(Int32(label))
        }
        
        return (Tensor(Array(samples.joined()), shape: samples.count, imageWidth, imageHeight), Tensor(labelVectors))
    }
    
    static func images(from path: String) -> ((Tensor<Float>, Tensor<Int32>), (Tensor<Float>, Tensor<Int32>)) {
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
        
        let trainingSampleCount = 60_000
        let testingSampleCount = 10_000
        
        let trainingSamples = readSamples(from: trainingBytes, labels: trainingLabels, count: trainingSampleCount)
        let testingSamples = readSamples(from: testingBytes, labels: testingLabels, count: testingSampleCount)
        
        return (trainingSamples,testingSamples)
    }
    
    func testMNist() {
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 5_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.parameters, learningRate: 0.001)
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            let (batch, expected) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)

            let y_pred = model.forward(batch)
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
            let pred = argmax(model.forward(x).squeeze())
            let actual = Int(ds_val.1[i].item)
            
            if pred == actual {
                correctCount += 1
            }
        }
        
        let accuracy = Float(correctCount) / Float(ds_val.0.shape[0])
        
        print("Accuracy: \(accuracy)")
        
        try? model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_params.json"))
    }
    
    func testMNistLstm() {
        let (ds_train, ds_val) = MNistTest.images(from: "/Users/Palle/Downloads/")
        
        let model = Sequential<Float>(
            GRU(inputSize: 28, hiddenSize: 128).asAny(),
            Dense(inputFeatures: 128, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        let epochs = 10_000
        let batchSize = 128
        
        let optimizer = Adam(parameters: model.parameters, learningRate: 0.001)
        
        print("Training...")
        
        for epoch in 1 ... epochs {
            optimizer.zeroGradient()
            
            let (batch, expected) = Random.minibatch(from: ds_train.0, labels: ds_train.1, count: batchSize)
            
            let x = batch.permuted(to: 1, 0, 2)
            let y_pred = model.forward(x)
            let loss = categoricalCrossEntropy(expected: expected, actual: y_pred)
            loss.backwards()
            
            optimizer.step()
            
            if epoch % 100 == 0 {
                let avgLoss = loss.item
                print("[\(epoch)/\(epochs)] loss: \(avgLoss)")
            }
        }
        
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
        
        try? model.saveWeights(to: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_gru_params2.json"))
    }
    
    func testGenerative() throws {
        let model = Sequential<Float>(
            Flatten().asAny(),
            Dense(inputFeatures: 28 * 28, outputFeatures: 500).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 500, outputFeatures: 300).asAny(),
            Tanh().asAny(),
            Dense(inputFeatures: 300, outputFeatures: 10).asAny(),
            Softmax().asAny()
        )
        
        try model.loadWeights(from: URL(fileURLWithPath: "/Users/Palle/Desktop/mnist_params.json"))
        
        let input = Tensor<Float>(repeating: 0, shape: [1, 28, 28], requiresGradient: true)
        let expected = Tensor<Int32>([5])
        
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

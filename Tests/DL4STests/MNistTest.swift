//
//  MNistTest.swift
//  DL4STests
//
//  Created by Palle Klewitz on 28.02.19.
//

import XCTest
@testable import DL4S

class MNistTest: XCTestCase {
    static func readSamples(from bytes: [UInt8], labels: [UInt8], count: Int) -> [(Vector<Float>, Vector<Int32>)]
    {
        let imageOffset = 16
        let labelOffset = 8
        
        let imageWidth = 28
        let imageHeight = 28
        
        var samples: [(Vector<Float>, Vector<Int32>)] = []
        
        for i in 0 ..< count
        {
            let offset = imageOffset + imageWidth * imageHeight * i
            let pixelData = bytes[offset ..< (offset + imageWidth * imageHeight)]
                .map{Float($0)/256}
            
            let label = Int(labels[labelOffset + i])
            
            //let sampleMatrix = Matrix3(values: pixelData, width: imageWidth, height: imageHeight, depth: 1)
            let sampleMatrix = Vector<Float>(pixelData, shape: imageHeight, imageWidth)
            let expectedValue = Vector<Int32>(Int32(label))
            samples.append((sampleMatrix, expectedValue))
        }
        
        return samples
    }
    
    static func images(from path: String) -> ([(Vector<Float>, Vector<Int32>)], [(Vector<Float>, Vector<Int32>)])
    {
        guard
            let trainingData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-images-idx3-ubyte")),
            let trainingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "train-labels-idx1-ubyte")),
            let testingData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-images-idx3-ubyte")),
            let testingLabelData = try? Data(contentsOf: URL(fileURLWithPath: path + "t10k-labels-idx1-ubyte"))
            else
        {
            return ([],[])
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
}

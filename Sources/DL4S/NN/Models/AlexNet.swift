//
//  AlexNet.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
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

public struct AlexNet<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            featureNet.parameterPaths.map((\Self.featureNet).appending(path:)),
            avgPool.parameterPaths.map((\Self.avgPool).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:)),
        ].joined())
    }
    
    public var parameters: [Tensor<Element, Device>] {
        Array([
            featureNet.parameters,
            avgPool.parameters,
            classifier.parameters
        ].joined())
    }
    
    var featureNet: Sequential<Sequential<Sequential<Sequential<Convolution2D<Element, Device>, Relu<Element, Device>>, Sequential<MaxPool2D<Element, Device>, Convolution2D<Element, Device>>>, Sequential<Sequential<Relu<Element, Device>, MaxPool2D<Element, Device>>, Sequential<Convolution2D<Element, Device>, Relu<Element, Device>>>>, Sequential<Sequential<Sequential<Convolution2D<Element, Device>, Relu<Element, Device>>, Convolution2D<Element, Device>>, Sequential<Relu<Element, Device>, MaxPool2D<Element, Device>>>>
    
    var avgPool: AdaptiveAvgPool2D<Element, Device>
    
    var classifier: Sequential<Sequential<Sequential<Dropout<Element, Device>, Dense<Element, Device>>, Sequential<Relu<Element, Device>, Dropout<Element, Device>>>, Sequential<Sequential<Dense<Element, Device>, Relu<Element, Device>>, Sequential<Dense<Element, Device>, Softmax<Element, Device>>>>
    
    var isDropoutActive: Bool {
        get {
            classifier.first.first.first.isActive || classifier.first.second.second.isActive
        }
        set {
            classifier.first.first.first.isActive = newValue
            classifier.first.second.second.isActive = newValue
        }
    }
    
    public init(inputChannels: Int, classes: Int) {
        featureNet = Sequential {
            Convolution2D<Element, Device>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (11, 11), padding: 2, stride: 4)
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            Convolution2D<Element, Device>(inputChannels: 64, outputChannels: 192, kernelSize: (5, 5), padding: 2, stride: 1)
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            Convolution2D<Element, Device>(inputChannels: 192, outputChannels: 384, kernelSize: (3, 3), padding: 1, stride: 1)
            Relu<Element, Device>()
            
            Convolution2D<Element, Device>(inputChannels: 384, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            Relu<Element, Device>()
            
            Convolution2D<Element, Device>(inputChannels: 384, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
        }
        
        avgPool = AdaptiveAvgPool2D(targetSize: 6)
        
        classifier = Sequential {
            Dropout<Element, Device>(rate: Float(0.5))
            Dense<Element, Device>(inputSize: 256 * 6 * 6, outputSize: 4096)
            Relu<Element, Device>()
            
            Dropout<Element, Device>(rate: Float(0.5))
            Dense<Element, Device>(inputSize: 4096, outputSize: 4096)
            Relu<Element, Device>()
            
            Dense<Element, Device>(inputSize: 4096, outputSize: classes)
            Softmax<Element, Device>()
        }
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        return classifier(avgPool(featureNet(inputs)))
    }
}

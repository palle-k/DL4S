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

/// AlexNet image classification network.
///
/// Batch normalization has been added to this implementation.
///
/// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
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
    
    var featureNet: Sequential<Sequential<Sequential<Sequential<Convolution2D<Element, Device>, Relu<Element, Device>>, Sequential<MaxPool2D<Element, Device>, Convolution2D<Element, Device>>>, Sequential<Sequential<BatchNorm<Element, Device>, Relu<Element, Device>>, Sequential<MaxPool2D<Element, Device>, Convolution2D<Element, Device>>>>, Sequential<Sequential<Sequential<Relu<Element, Device>, Convolution2D<Element, Device>>, Sequential<BatchNorm<Element, Device>, Relu<Element, Device>>>, Sequential<Sequential<Convolution2D<Element, Device>, BatchNorm<Element, Device>>, Sequential<Relu<Element, Device>, MaxPool2D<Element, Device>>>>>
    
    var avgPool: AdaptiveAvgPool2D<Element, Device>
    
    var classifier: Sequential<Sequential<Sequential<Sequential<Flatten<Element, Device>, Dropout<Element, Device>>, Sequential<Dense<Element, Device>, BatchNorm<Element, Device>>>, Sequential<Sequential<Relu<Element, Device>, Dropout<Element, Device>>, Sequential<Dense<Element, Device>, BatchNorm<Element, Device>>>>, Sequential<Sequential<Relu<Element, Device>, Dense<Element, Device>>, LogSoftmax<Element, Device>>>
    
    /// Determines whether dropout is applied in the classification block
    public var isDropoutActive: Bool {
        get {
            classifier.first.first.first.second.isActive || classifier.first.second.first.second.isActive
        }
        set {
            classifier.first.first.first.second.isActive = newValue
            classifier.first.second.first.second.isActive = newValue
        }
    }
    
    
    /// Creates an AlexNet image classification network with the given number of input channels and classes.
    ///
    /// The network expects images with a resolution of 192x192 or higher.
    ///
    /// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    /// - Parameters:
    ///   - inputChannels: Number of input channesls. Data forwarded through the network must have [batchSize, inputChannels, height, depth] shape.
    ///   - classes: Number of classes / dimensionality of network output
    public init(inputChannels: Int, classes: Int) {
        featureNet = Sequential {
            Convolution2D<Element, Device>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (11, 11), padding: 2, stride: 4)
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            Convolution2D<Element, Device>(inputChannels: 64, outputChannels: 192, kernelSize: (5, 5), padding: 2, stride: 1)
            BatchNorm<Element, Device>(inputSize: [192, 1, 1])
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            Convolution2D<Element, Device>(inputChannels: 192, outputChannels: 384, kernelSize: (3, 3), padding: 1, stride: 1)
            Relu<Element, Device>()
            
            Convolution2D<Element, Device>(inputChannels: 384, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            BatchNorm<Element, Device>(inputSize: [256, 1, 1])
            Relu<Element, Device>()
            
            Convolution2D<Element, Device>(inputChannels: 256, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            BatchNorm<Element, Device>(inputSize: [256, 1, 1])
            Relu<Element, Device>()
            MaxPool2D<Element, Device>(windowSize: 3, stride: 2)
        }
        
        avgPool = AdaptiveAvgPool2D(targetSize: 6)
        
        classifier = Sequential {
            Flatten<Element, Device>()
            
            Dropout<Element, Device>(rate: Float(0.5))
            Dense<Element, Device>(inputSize: 256 * 6 * 6, outputSize: 4096)
            BatchNorm<Element, Device>(inputSize: [4096])
            Relu<Element, Device>()
            
            Dropout<Element, Device>(rate: Float(0.5))
            Dense<Element, Device>(inputSize: 4096, outputSize: 4096)
            BatchNorm<Element, Device>(inputSize: [4096])
            Relu<Element, Device>()
            
            Dense<Element, Device>(inputSize: 4096, outputSize: classes)
            LogSoftmax<Element, Device>()
        }
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        return classifier(avgPool(featureNet(inputs)))
    }
}

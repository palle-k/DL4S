//
//  ResNet.swift
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

extension Optional {
    var forceUnwrapped: Wrapped {
        get {self!}
        set {self = newValue}
    }
}

/// Residual neural network with 18 layers (17 convolutional, 1 dense)
public struct ResNet18<Element: RandomizableType, Device: DeviceType>: LayerType {
    public var parameters: [Tensor<Parameter, Self.Device>] {
        get {
            Array([
                start.parameters,
                l1.parameters,
                l2.parameters,
                l3.parameters,
                l4.parameters,
                classifier.parameters
            ].joined())
        }
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            start.parameterPaths.map((\Self.start).appending(path:)),
            l1.parameterPaths.map((\Self.l1).appending(path:)),
            l2.parameterPaths.map((\Self.l2).appending(path:)),
            l3.parameterPaths.map((\Self.l3).appending(path:)),
            l4.parameterPaths.map((\Self.l4).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:))
        ].joined())
    }
    
    public var start: Sequential<Sequential<Convolution2D<Element, Device>, BatchNorm<Element, Device>>, Relu<Element, Device>>
    public var l1: Sequential<ResidualBlock<Element, Device>, ResidualBlock<Element, Device>>
    public var l2: Sequential<ResidualBlock<Element, Device>, ResidualBlock<Element, Device>>
    public var l3: Sequential<ResidualBlock<Element, Device>, ResidualBlock<Element, Device>>
    public var l4: Sequential<ResidualBlock<Element, Device>, ResidualBlock<Element, Device>>
    public var classifier: Sequential<Sequential<AdaptiveAvgPool2D<Element, Device>, Flatten<Element, Device>>, Sequential<Dense<Element, Device>, LogSoftmax<Element, Device>>>
    
    public init(inputShape: [Int], classes: Int) {
        let startOut = ConvUtil.outputShape(for: inputShape, kernelCount: 64, kernelWidth: 7, kernelHeight: 7, stride: 2, padding: 3)
        
        start = Sequential {
            Convolution2D<Element, Device>(inputChannels: inputShape[0], outputChannels: 64, kernelSize: (7, 7), padding: 3, stride: 2)
            BatchNorm<Element, Device>(inputSize: startOut)
            Relu<Element, Device>()
        }
        
        l1 = Sequential {
            ResidualBlock<Element, Device>(inputShape: startOut, outPlanes: 64, downsample: 1)
            ResidualBlock<Element, Device>(inputShape: startOut, outPlanes: 64, downsample: 1)
        }
        
        l2 = Sequential {
            ResidualBlock<Element, Device>(inputShape: [64, startOut[1], startOut[2]], outPlanes: 128, downsample: 2)
            ResidualBlock<Element, Device>(inputShape: [128, startOut[1] / 2, startOut[2] / 2], outPlanes: 128, downsample: 1)
        }
        
        l3 = Sequential {
            ResidualBlock<Element, Device>(inputShape: [128, startOut[1] / 2, startOut[2] / 2], outPlanes: 256, downsample: 2)
            ResidualBlock<Element, Device>(inputShape: [256, startOut[1] / 4, startOut[2] / 4], outPlanes: 256, downsample: 1)
        }
        
        l4 = Sequential {
            ResidualBlock<Element, Device>(inputShape: [256, startOut[1] / 4, startOut[2] / 4], outPlanes: 512, downsample: 2)
            ResidualBlock<Element, Device>(inputShape: [512, startOut[1] / 8, startOut[2] / 8], outPlanes: 512, downsample: 1)
        }
        
        classifier = Sequential {
            AdaptiveAvgPool2D<Element, Device>(targetSize: 1)
            Flatten<Element, Device>()
            Dense<Element, Device>(inputSize: 512, outputSize: classes)
            LogSoftmax<Element, Device>()
        }
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "ResNet18") {
            var x = inputs
            
            x = start(x)
            x = l1(x)
            x = l2(x)
            x = l3(x)
            x = l4(x)
            x = classifier(x)
            
            return x
        }
    }
}

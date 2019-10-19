//
//  XResNet.swift
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

public struct XResidualBlock<Element: RandomizableType, Device: DeviceType>: XLayer {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {
        return Array([
            conv1.parameterPaths.map((\Self.conv1).appending(path:)),
            conv2.parameterPaths.map((\Self.conv2).appending(path:)),
            bn1.parameterPaths.map((\Self.bn1).appending(path:)),
            bn2.parameterPaths.map((\Self.bn2).appending(path:)),
            downsample?.parameterPaths.map((\Self.downsample.forceUnwrapped).appending(path:)) ?? []
        ].joined())
    }
    
    public var parameters: [XTensor<Element, Device>] {
        Array([
            conv1.parameters,
            conv2.parameters,
            bn1.parameters,
            bn2.parameters,
            downsample?.parameters ?? []
        ].joined())
    }
    
    public var conv1: XConvolution2D<Element, Device>
    public var conv2: XConvolution2D<Element, Device>
    public var bn1: XBatchNorm<Element, Device>
    public var bn2: XBatchNorm<Element, Device>
    public var downsample: XSequential<XConvolution2D<Element, Device>, XBatchNorm<Element, Device>>?
    
    public init(inputShape: [Int], outPlanes: Int, downsample: Int) {
        conv1 = XConvolution2D(inputChannels: inputShape[0], outputChannels: outPlanes, kernelSize: (3, 3), stride: downsample)
        conv2 = XConvolution2D(inputChannels: outPlanes, outputChannels: outPlanes, kernelSize: (3, 3))
        
        let convShape = ConvUtil.outputShape(for: inputShape, kernelCount: outPlanes, kernelWidth: 3, kernelHeight: 3, stride: downsample, padding: 1)
        
        bn1 = XBatchNorm(inputSize: convShape)
        bn2 = XBatchNorm(inputSize: convShape)
        
        if downsample != 1 {
            self.downsample = XSequential {
                XConvolution2D<Element, Device>(inputChannels: inputShape[0], outputChannels: outPlanes, kernelSize: (1, 1), padding: 0, stride: downsample)
                XBatchNorm<Element, Device>(inputSize: [outPlanes, inputShape[1] / downsample, inputShape[2] / downsample])
            }
        } else {
            self.downsample = nil
        }
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "ResidualBlock") {
            var x = inputs
            
            let r = downsample?(x) ?? x
            x = conv1(x)
            x = bn1(x)
            x = relu(x)
            x = conv2(x)
            x = bn2(x)
            x = x + r
            x = relu(x)
            
            return x
        }
    }
}

public struct ResNet18<Element: RandomizableType, Device: DeviceType>: XLayer {
    public var parameters: [XTensor<Parameter, Self.Device>] {
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
    
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {
        Array([
            start.parameterPaths.map((\Self.start).appending(path:)),
            l1.parameterPaths.map((\Self.l1).appending(path:)),
            l2.parameterPaths.map((\Self.l2).appending(path:)),
            l3.parameterPaths.map((\Self.l3).appending(path:)),
            l4.parameterPaths.map((\Self.l4).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:))
        ].joined())
    }
    
    public var start: XSequential<XSequential<XConvolution2D<Element, Device>, XBatchNorm<Element, Device>>, XRelu<Element, Device>>
    public var l1: XSequential<XResidualBlock<Element, Device>, XResidualBlock<Element, Device>>
    public var l2: XSequential<XResidualBlock<Element, Device>, XResidualBlock<Element, Device>>
    public var l3: XSequential<XResidualBlock<Element, Device>, XResidualBlock<Element, Device>>
    public var l4: XSequential<XResidualBlock<Element, Device>, XResidualBlock<Element, Device>>
    public var classifier: XSequential<XSequential<XAdaptiveAvgPool2D<Element, Device>, XFlatten<Element, Device>>, XSequential<XDense<Element, Device>, XSoftmax<Element, Device>>>
    
    public init(inputShape: [Int], classes: Int) {
        let startOut = ConvUtil.outputShape(for: inputShape, kernelCount: 64, kernelWidth: 7, kernelHeight: 7, stride: 2, padding: 3)
        
        start = XSequential {
            XConvolution2D<Element, Device>(inputChannels: inputShape[0], outputChannels: 64, kernelSize: (7, 7), padding: 3, stride: 2)
            XBatchNorm<Element, Device>(inputSize: startOut)
            XRelu<Element, Device>()
        }
        
        l1 = XSequential {
            XResidualBlock<Element, Device>(inputShape: startOut, outPlanes: 64, downsample: 1)
            XResidualBlock<Element, Device>(inputShape: startOut, outPlanes: 64, downsample: 1)
        }
        
        l2 = XSequential {
            XResidualBlock<Element, Device>(inputShape: [64, startOut[1], startOut[2]], outPlanes: 128, downsample: 2)
            XResidualBlock<Element, Device>(inputShape: [128, startOut[1] / 2, startOut[2] / 2], outPlanes: 128, downsample: 1)
        }
        
        l3 = XSequential {
            XResidualBlock<Element, Device>(inputShape: [128, startOut[1] / 2, startOut[2] / 2], outPlanes: 256, downsample: 2)
            XResidualBlock<Element, Device>(inputShape: [256, startOut[1] / 4, startOut[2] / 4], outPlanes: 256, downsample: 1)
        }
        
        l4 = XSequential {
            XResidualBlock<Element, Device>(inputShape: [256, startOut[1] / 4, startOut[2] / 4], outPlanes: 512, downsample: 2)
            XResidualBlock<Element, Device>(inputShape: [512, startOut[1] / 8, startOut[2] / 8], outPlanes: 512, downsample: 1)
        }
        
        classifier = XSequential {
            XAdaptiveAvgPool2D<Element, Device>(targetSize: 1)
            XFlatten<Element, Device>()
            XDense<Element, Device>(inputSize: 512, outputSize: classes)
            XSoftmax<Element, Device>()
        }
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
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

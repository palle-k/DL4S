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


public struct XAlexNet<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {
        Array([
            featureNet.parameterPaths.map((\Self.featureNet).appending(path:)),
            avgPool.parameterPaths.map((\Self.avgPool).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:)),
        ].joined())
    }
    
    public var parameters: [XTensor<Element, Device>] {
        Array([
            featureNet.parameters,
            avgPool.parameters,
            classifier.parameters
        ].joined())
    }
    
    var featureNet: XSequential<XSequential<XSequential<XSequential<XConvolution2D<Element, Device>, XRelu<Element, Device>>, XSequential<XMaxPool2D<Element, Device>, XConvolution2D<Element, Device>>>, XSequential<XSequential<XRelu<Element, Device>, XMaxPool2D<Element, Device>>, XSequential<XConvolution2D<Element, Device>, XRelu<Element, Device>>>>, XSequential<XSequential<XSequential<XConvolution2D<Element, Device>, XRelu<Element, Device>>, XConvolution2D<Element, Device>>, XSequential<XRelu<Element, Device>, XMaxPool2D<Element, Device>>>>
    
    var avgPool: XAdaptiveAvgPool2D<Element, Device>
    
    var classifier: XSequential<XSequential<XSequential<XDropout<Element, Device>, XDense<Element, Device>>, XSequential<XRelu<Element, Device>, XDropout<Element, Device>>>, XSequential<XSequential<XDense<Element, Device>, XRelu<Element, Device>>, XSequential<XDense<Element, Device>, XSoftmax<Element, Device>>>>
    
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
        featureNet = XSequential {
            XConvolution2D<Element, Device>(inputChannels: inputChannels, outputChannels: 64, kernelSize: (11, 11), padding: 2, stride: 4)
            XRelu<Element, Device>()
            XMaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            XConvolution2D<Element, Device>(inputChannels: 64, outputChannels: 192, kernelSize: (5, 5), padding: 2, stride: 1)
            XRelu<Element, Device>()
            XMaxPool2D<Element, Device>(windowSize: 3, stride: 2)
            
            XConvolution2D<Element, Device>(inputChannels: 192, outputChannels: 384, kernelSize: (3, 3), padding: 1, stride: 1)
            XRelu<Element, Device>()
            
            XConvolution2D<Element, Device>(inputChannels: 384, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            XRelu<Element, Device>()
            
            XConvolution2D<Element, Device>(inputChannels: 384, outputChannels: 256, kernelSize: (3, 3), padding: 1, stride: 1)
            XRelu<Element, Device>()
            XMaxPool2D<Element, Device>(windowSize: 3, stride: 2)
        }
        
        avgPool = XAdaptiveAvgPool2D(targetSize: 6)
        
        classifier = XSequential {
            XDropout<Element, Device>(rate: Float(0.5))
            XDense<Element, Device>(inputSize: 256 * 6 * 6, outputSize: 4096)
            XRelu<Element, Device>()
            
            XDropout<Element, Device>(rate: Float(0.5))
            XDense<Element, Device>(inputSize: 4096, outputSize: 4096)
            XRelu<Element, Device>()
            
            XDense<Element, Device>(inputSize: 4096, outputSize: classes)
            XSoftmax<Element, Device>()
        }
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        return classifier(avgPool(featureNet(inputs)))
    }
}

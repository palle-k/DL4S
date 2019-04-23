//
//  AlexNet.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.04.19.
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


public class AlexNet<Element: RandomizableType, Device: DeviceType>: Layer {
    public var isTrainable: Bool = true
    
    let featureNet: Sequential<Element, Device>
    let avgPool: AdaptiveAvgPool2D<Element, Device>
    let classifier: Sequential<Element, Device>
    
    let dropout1: Dropout<Element, Device>
    let dropout2: Dropout<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return Array([featureNet.parameters, avgPool.parameters, classifier.parameters].joined())
    }
    
    public var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return Array([featureNet.trainableParameters, avgPool.trainableParameters, classifier.trainableParameters].joined())
    }
    
    public var isDropoutActive: Bool {
        get {
            return dropout1.isActive || dropout2.isActive
        }
        set {
            dropout1.isActive = newValue
            dropout2.isActive = newValue
        }
    }
    
    public init(numChannels: Int = 3, numClasses: Int) {
        featureNet = Sequential(
            Conv2D(inputChannels: numChannels, outputChannels: 64, kernelSize: 11, stride: 4, padding: 2).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 3, stride: 2).asAny(),
            Conv2D(inputChannels: 64, outputChannels: 192, kernelSize: 5, stride: 1, padding: 2).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 3, stride: 2).asAny(),
            Conv2D(inputChannels: 192, outputChannels: 384, kernelSize: 3, stride: 1, padding: 1).asAny(),
            Relu().asAny(),
            Conv2D(inputChannels: 384, outputChannels: 256, kernelSize: 3, stride: 1, padding: 1).asAny(),
            Relu().asAny(),
            Conv2D(inputChannels: 384, outputChannels: 256, kernelSize: 3, stride: 1, padding: 1).asAny(),
            Relu().asAny(),
            MaxPool2D(windowSize: 3, stride: 2).asAny()
        )
        
        avgPool = AdaptiveAvgPool2D(targetSize: 6)
        
        dropout1 = Dropout(rate: 0.5)
        dropout2 = Dropout(rate: 0.5)

        classifier = Sequential(
            dropout1.asAny(),
            Dense(inputFeatures: 256 * 6 * 6, outputFeatures: 4096).asAny(),
            Relu().asAny(),
            dropout2.asAny(),
            Dense(inputFeatures: 4096, outputFeatures: 4096).asAny(),
            Relu().asAny(),
            Dense(inputFeatures: 4096, outputFeatures: numClasses).asAny()
        )
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        
        x = featureNet(x)
        x = avgPool(x)
        x = classifier(x)
        
        return x
    }
}

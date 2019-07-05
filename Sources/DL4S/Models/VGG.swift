//
//  VGG.swift
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


public class VGG<Element: RandomizableType, Device: DeviceType>: Layer {
    public enum Configuration: String, Hashable {
        case vgg11
        case vgg13
        case vgg16
        case vgg19
        
        fileprivate var operationSequence: [Operation] {
            switch self {
            case .vgg11:
                return [
                    .conv(64),
                    .pool,
                    .conv(128),
                    .pool,
                    .conv(256), .conv(256),
                    .pool,
                    .conv(512), .conv(512),
                    .pool,
                    .conv(512), .conv(512),
                    .pool
                ]
            case .vgg13:
                return [
                    .conv(64), .conv(64),
                    .pool,
                    .conv(128), .conv(128),
                    .pool,
                    .conv(256), .conv(256),
                    .pool,
                    .conv(512), .conv(512),
                    .pool,
                    .conv(512), .conv(512),
                    .pool
                ]
            case .vgg16:
                return [
                    .conv(64), .conv(64),
                    .pool,
                    .conv(128), .conv(128),
                    .pool,
                    .conv(256), .conv(256), .conv(256),
                    .pool,
                    .conv(512), .conv(512), .conv(512),
                    .pool,
                    .conv(512), .conv(512), .conv(512),
                    .pool
                ]
            case .vgg19:
                return [
                    .conv(64), .conv(64),
                    .pool,
                    .conv(128), .conv(128),
                    .pool,
                    .conv(256), .conv(256), .conv(256), .conv(256),
                    .pool,
                    .conv(512), .conv(512), .conv(512), .conv(512),
                    .pool,
                    .conv(512), .conv(512), .conv(512), .conv(512),
                    .pool
                ]
            }
        }
    }
    
    fileprivate enum Operation {
        case pool
        case conv(Int)
    }
    
    public var isTrainable: Bool = true
    
    let features: Sequential<Element, Device>
    let pool: AdaptiveAvgPool2D<Element, Device>
    let classifier: Sequential<Element, Device>
    
    let dropout1: Dropout<Element, Device>
    let dropout2: Dropout<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return Array([features.parameters, classifier.parameters].joined())
    }
    
    public var trainableParameters: [Tensor<Element, Device>] {
        guard isTrainable else {
            return []
        }
        return Array([features.trainableParameters, classifier.trainableParameters].joined())
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
    
    public init(numChannels: Int = 3, numClasses: Int, configuration: VGG.Configuration) {
        features = VGG<Element, Device>.featureLayers(for: configuration, channels: numChannels)
        pool = AdaptiveAvgPool2D(targetSize: 7)
        
        dropout1 = Dropout(rate: 0.5)
        dropout2 = Dropout(rate: 0.5)
        
        classifier = Sequential(
            Dense(inputFeatures: 512 * 7 * 7, outputFeatures: 4096).asAny(),
            Relu().asAny(),
            dropout1.asAny(),
            Dense(inputFeatures: 4096, outputFeatures: 4096).asAny(),
            Relu().asAny(),
            dropout2.asAny(),
            Dense(inputFeatures: 4096, outputFeatures: numClasses).asAny()
        )
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        var x = inputs[0]
        
        x = features(x)
        x = pool(x)
        x = classifier(x)
        
        return x
    }
    
    private static func featureLayers(for configuration: Configuration, channels: Int) -> Sequential<Element, Device> {
        let featureLayers = Sequential<Element, Device>()
        
        var channels = channels
        
        for op in configuration.operationSequence {
            switch op {
            case .pool:
                featureLayers.append(MaxPool2D(windowSize: 2))
            case .conv(let outChannels):
                featureLayers.append(Conv2D(inputChannels: channels, outputChannels: outChannels, kernelSize: 3))
                channels = outChannels
            }
        }
        
        return featureLayers
    }
}

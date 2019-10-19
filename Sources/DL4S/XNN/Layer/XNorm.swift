//
//  XNorm.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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

public struct XBatchNorm<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public static var parameters: [WritableKeyPath<XBatchNorm<Element, Device>, XTensor<Element, Device>>] {[
        \.shift,
        \.scale
    ]}
    public var parameters: [XTensor<Element, Device>] {
        get {[shift, scale]}
        set {
            shift = newValue[0]
            scale = newValue[1]
        }
    }
    
    public var isTraining = true
    
    public var shift: XTensor<Element, Device>
    public var scale: XTensor<Element, Device>
    
    public var momentum: Element
    
    public init(inputSize: [Int], momentum: Element = 0.9) {
        shift = XTensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = XTensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        self.momentum = momentum
        
        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "BatchNorm") {
            let x = inputs
            let mean = x.reduceMean(along: [0])
            let variance = x.variance(along: [0])
            let normalized = (x - mean) / (sqrt(variance) + 1e-5)
            return normalized * scale + shift
        }
    }
}


public struct XLayerNorm<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public static var parameters: [WritableKeyPath<Self, XTensor<Element, Device>>] {[
        \.shift,
        \.scale
    ]}
    public var parameters: [XTensor<Element, Device>] {
        get {[shift, scale]}
        set {
            shift = newValue[0]
            scale = newValue[1]
        }
    }
    
    public var isTraining = true
    
    public var shift: XTensor<Element, Device>
    public var scale: XTensor<Element, Device>
    
    public var momentum: Element
    
    public init(inputSize: [Int], momentum: Element = 0.9) {
        shift = XTensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = XTensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        self.momentum = momentum
        
        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "LayerNorm") {
            let x = inputs
            let axes = Array(1 ..< x.dim)
            let mean = x
                .reduceMean(along: axes)
                .view(as: [x.shape[0]] + Array(repeating: 1, count: axes.count))
            
            let variance = x
                .variance(along: axes)
                .view(as: mean.shape)
            
            let normalized = (x - mean) / (sqrt(variance) + 1e-5)
            return normalized // * scale + shift
        }
    }
}

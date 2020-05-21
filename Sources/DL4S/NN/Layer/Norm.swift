//
//  Norm.swift
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

/// A layer that normalizes its inputs along the batch dimension
public struct BatchNorm<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        [\.shift, \.scale]
    }
    public var parameters: [Tensor<Element, Device>] {
        get {[shift, scale]}
    }
    
    /// Whether the layer is training, currently ignored.
    public var isTraining = true
    
    /// Learned shift vector
    public var shift: Tensor<Element, Device>
    
    /// Learned scale vector
    public var scale: Tensor<Element, Device>
    
    /// Momentum with which to update mean and variance. Currently ignored
    public var momentum: Element

    /// A layer that normalizes its inputs along the batch dimension.
    public init(inputSize: [Int], momentum: Element = 0.9) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        self.momentum = momentum
        
        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "BatchNorm") {
            let x = inputs
            let mean = x.reduceMean(along: [0])
            let variance = x.variance(along: [0])
            let normalized = (x - mean) / (sqrt(variance) + 1e-5)
            return normalized * scale + shift
        }
    }
}

/// A layer that normalizes its inputs along all dimensions except the batch dimension
public struct LayerNorm<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        [\.shift, \.scale]
    }
    public var parameters: [Tensor<Element, Device>] {
        get {[shift, scale]}
    }
    
    /// Whether the layer is training, currently ignored.
    public var isTraining = true
    
    /// Learned shift vector
    public var shift: Tensor<Element, Device>
    
    /// Learned scale vector
    public var scale: Tensor<Element, Device>
    
    /// Momentum with which to update mean and variance. Currently ignored
    public var momentum: Element

    /// A layer that normalizes its inputs along all dimensions except the batch dimension
    public init(inputSize: [Int], momentum: Element = 0.9) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        self.momentum = momentum
        
        #if DEBUG
        shift.tag = "shift"
        scale.tag = "scale"
        #endif
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
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
            return normalized * scale + shift
        }
    }
}

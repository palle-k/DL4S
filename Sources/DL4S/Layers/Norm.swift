//
//  Norm.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
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


public class BatchNorm<Element: NumericType, Device: DeviceType>: Layer, Codable {
    
    public typealias Input = Element
    
    public var isTrainable: Bool = true
    public var isTraining: Bool = true
    
    public var parameters: [Tensor<Element, Device>] {
        return [shift, scale]
    }
    
    let shift: Tensor<Element, Device>
    let scale: Tensor<Element, Device>
    
    var runningMean: Tensor<Element, Device>
    var runningVar: Tensor<Element, Device>
    
    public var momentum: Element
    
    public init(inputSize: [Int], momentum: Element = 0.9) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        runningMean = Tensor(repeating: 0, shape: inputSize)
        runningVar = Tensor(repeating: 1, shape: inputSize)
        
        self.momentum = momentum
        
        shift.tag = "shift"
        scale.tag = "scale"
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let x = inputs[0]
        
        if self.isTraining {
            let mean = x.mean(axes: 0)
            let variance = x.variance(axes: 0)
            
            runningMean = Tensor(momentum) * runningMean + Tensor(1 - momentum) * mean.detached()
            runningVar = Tensor(momentum) * runningVar + Tensor(1 - momentum) * variance.detached()
            
            let normalized = (x - mean) / (sqrt(variance) + 1e-5)
            return normalized * scale + shift
        } else {
            let normalized = (x - runningMean) / (sqrt(runningVar) + 1e-5)
            return normalized * scale + shift
        }
    }
}


public class LayerNorm<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var isTrainable: Bool = true
    
    let shift: Tensor<Element, Device>
    let scale: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return [shift, scale]
    }
    
    public init(inputSize: [Int]) {
        shift = Tensor(repeating: 0, shape: inputSize, requiresGradient: true)
        scale = Tensor(repeating: 1, shape: inputSize, requiresGradient: true)
        
        shift.tag = "shift"
        scale.tag = "scale"
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let x = inputs[0]
        
        let axes = Array(1 ..< x.dim)
        
        // reshaping mean & variance so that both are broadcasted over batch dimension
        
        let mean = x
            .mean(axes: axes)
            .view(as: [x.shape[0]] + Array(repeating: 1, count: axes.count))
        
        let variance = x
            .variance(axes: axes)
            .view(as: mean.shape)
        
        let normalized = (x - mean) / (sqrt(variance) + 1e-5)
        return normalized // * scale + shift
    }
}

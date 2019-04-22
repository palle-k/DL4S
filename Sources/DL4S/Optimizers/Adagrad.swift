//
//  AdaGrad.swift
//  DL4S
//
//  Created by Palle Klewitz on 22.04.19.
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


public class Adagrad<Element: NumericType, Device: DeviceType>: Optimizer {
    public var learningRate: Element
    public var epsilon: Element
    public let parameters: [Tensor<Element, Device>]
    
    private var gradientSums: [Buffer<Element, Device>]
    private var cache: [Buffer<Element, Device>]
    
    public init(parameters: [Tensor<Element, Device>], learningRate: Element, epsilon: Element = 1e-8) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gradientSums = []
        self.cache = []
        
        reset()
    }
    
    deinit {
        gradientSums.forEach(Device.Memory.free)
        cache.forEach(Device.Memory.free)
    }
    
    public func step() {
        for i in 0 ..< parameters.count {
            let param = parameters[i]
            let sum = gradientSums[i]
            let c = cache[i]
            
            guard let gradient = param.gradient else {
                continue
            }
            
            // Updating sum of squared gradients
            Device.Engine.vMA(lhs: gradient, rhs: gradient, add: sum, result: sum, count: param.count)
            
            // Computing adapted learning rate:
            // -learningRate / sqrt(sum + epsilon)
            Device.Engine.vsAdd(lhs: sum, rhs: epsilon, result: c, count: param.count)
            Device.Engine.sqrt(val: c, result: c, count: param.count)
            Device.Engine.svDiv(lhs: -learningRate, rhs: c, result: c, count: param.count)
            
            // Performing weight update
            Device.Engine.vMA(lhs: c, rhs: gradient, add: param.values, result: param.values, count: param.count)
        }
    }
    
    public func reset() {
        gradientSums.forEach(Device.Memory.free)
        cache.forEach(Device.Memory.free)
        
        gradientSums = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        cache = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        
        for gs in gradientSums {
            Device.Engine.fill(value: 0, result: gs, count: gs.count)
        }
        for c in cache {
            Device.Engine.fill(value: 0, result: c, count: c.count)
        }
    }
    
    public func zeroGradient() {
        parameters.forEach {
            $0.zeroGradient()
        }
    }
}

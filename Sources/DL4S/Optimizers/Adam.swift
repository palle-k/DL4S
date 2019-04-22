//
//  Adam.swift
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


public class Adam<Element: NumericType, Device: DeviceType>: Optimizer {
    public let parameters: [Tensor<Element, Device>]
    
    public var learningRate: Element
    public var beta1: Element
    public var beta2: Element
    public var epsilon: Element
    
    private var beta1t: Element
    private var beta2t: Element
    
    private let firstMoment: [Buffer<Element, Device>]
    private let secondMoment: [Buffer<Element, Device>]
    private let cache: [Buffer<Element, Device>]
    private let cache2: [Buffer<Element, Device>]
    
    public private(set) var iteration: Int = 0
    
    public init(parameters: [Tensor<Element, Device>], learningRate: Element, beta1: Element = 0.9, beta2: Element = 0.999, epsilon: Element = 1e-8) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.firstMoment = parameters.map {Device.Memory.allocateBuffer(withCapacity: $0.count, type: Element.self)}
        self.secondMoment = parameters.map {Device.Memory.allocateBuffer(withCapacity: $0.count, type: Element.self)}
        self.cache = parameters.map {Device.Memory.allocateBuffer(withCapacity: $0.count, type: Element.self)}
        self.cache2 = parameters.map {Device.Memory.allocateBuffer(withCapacity: $0.count, type: Element.self)}
        
        self.beta1t = beta1
        self.beta2t = beta2
        
        for m in [firstMoment, secondMoment, cache, cache2].joined() {
            //m.assign(repeating: 0)
            Device.Engine.fill(value: 0, result: m, count: m.count)
        }
    }
    
    deinit {
        [firstMoment, secondMoment, cache, cache2].joined().forEach(Device.Memory.free)
    }
    
    public func reset() {
        beta1t = beta1
        beta2t = beta2
        for m in [firstMoment, secondMoment, cache, cache2].joined() {
            Device.Engine.fill(value: 0, result: m, count: m.count)
        }
    }
    
    public func step() {
        for (i, param) in parameters.enumerated() {
            guard let g = param.gradient else {
                continue
            }
            
            let m = firstMoment[i]
            let v = secondMoment[i]
            let c = cache[i]
            let c2 = cache2[i]
            
            // cache = (1 - beta1) * gradient
            Device.Engine.vsMul(lhs: g, rhs: (1 - beta1), result: c, count: param.count)
            // m = beta1 * m + cache = beta1 * m + (1 - beta1) * gradient
            Device.Engine.vsMulVAdd(lhs: m, rhs: beta1, add: c, result: m, count: param.count)
            
            // cache = gradient ^ 2
            Device.Engine.vSquare(values: g, result: c, count: param.count)
            // cache = (1 - beta2) * gradient ^ 2
            Device.Engine.vsMul(lhs: c, rhs: (1 - beta2), result: c, count: param.count)
            // v = beta2 * v + (1 - beta2) * gradient ^ 2
            Device.Engine.vsMulVAdd(lhs: v, rhs: beta2, add: c, result: v, count: param.count)
            
            // c (m^) = m / (1 - beta1)
            Device.Engine.vsMul(lhs: m, rhs: 1 / (1 - beta1t), result: c, count: param.count)
            
            // c2 (^v) = v / (1 - beta2)
            Device.Engine.vsMul(lhs: v, rhs: 1 / (1 - beta2t), result: c2, count: param.count)
            
            // c2 = sqrt(c2)
            Device.Engine.sqrt(val: c2, result: c2, count: param.count)
            // c2 = c2 + eps
            Device.Engine.vsAdd(lhs: c2, rhs: epsilon, result: c2, count: param.count)
            // c2 = -lr / c2
            Device.Engine.svDiv(lhs: -learningRate, rhs: c2, result: c2, count: param.count)
            
            // weights = c * c2 + weights = -lr / (sqrt(c2) + eps) * c + weights
            Device.Engine.vMA(lhs: c, rhs: c2, add: param.values, result: param.values, count: param.count)
        }
        
        // beta_t = pow(beta_, iteration)
        beta1t = beta1t * beta1
        beta2t = beta2t * beta2
    }
    
    public func zeroGradient() {
        for parameter in self.parameters {
            parameter.zeroGradient()
        }
    }
    
}

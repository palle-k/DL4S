//
//  SGD.swift
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


public class GradientDescent<Element: NumericType, Device: DeviceType>: Optimizer {
    public var learningRate: Element
    public var regularizationL2: Element = 0
    
    public let parameters: [Tensor<Element, Device>]
    
    public init(parameters: [Tensor<Element, Device>], learningRate: Element) {
        self.learningRate = learningRate
        self.parameters = parameters
    }
    
    public func step() {
        for parameter in self.parameters {
            guard let gradient = parameter.gradient else {
                continue
            }
            Device.Engine.vsMulVAdd(lhs: gradient, rhs: -self.learningRate, add: parameter.values, result: parameter.values, count: parameter.count)
        }
    }
    
    public func zeroGradient() {
        for parameter in self.parameters {
            parameter.zeroGradient()
        }
    }
    
    public func reset() {
        // noop
    }
}


public class Momentum<Element: NumericType, Device: DeviceType>: Optimizer {
    public var learningRate: Element
    public var momentum: Element
    
    public let parameters: [Tensor<Element, Device>]
    private let momentumParams: [Buffer<Element, Device>]
    
    public init(parameters: [Tensor<Element, Device>], learningRate: Element, momentum: Element = 0.8) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.parameters = parameters
        self.momentumParams = parameters.map {Device.Memory.allocateBuffer(withCapacity: $0.count, type: Element.self)}
        
        for m in self.momentumParams {
            Device.Engine.fill(value: 0, result: m, count: m.count)
        }
    }
    
    deinit {
        self.momentumParams.forEach(Device.Memory.free)
    }
    
    public func reset() {
        for m in self.momentumParams {
            Device.Engine.fill(value: 0, result: m, count: m.count)
        }
    }
    
    public func step() {
        for (param, momentumParam) in zip(parameters, momentumParams) {
            guard let gradient = param.gradient else {
                continue
            }
            // m = momentum * m
            Device.Engine.vsMul(lhs: momentumParam, rhs: self.momentum, result: momentumParam, count: param.count)
            // m = lr * grad + m = lr * grad + momentum * m
            Device.Engine.vsMulVAdd(lhs: gradient, rhs: self.learningRate, add: momentumParam, result: momentumParam, count: param.count)
            // values = values - m
            Device.Engine.vSub(lhs: param.values, rhs: momentumParam, result: param.values, count: param.count)
            
            
        }
    }
    
    public func zeroGradient() {
        for parameter in self.parameters {
            parameter.zeroGradient()
        }
    }
}


//
//  AdaDelta.swift
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


public class Adadelta<Element: NumericType, Device: DeviceType>: Optimizer {
    public var learningRate: Element
    public var gamma: Element
    public var epsilon: Element
    public let parameters: [Tensor<Element, Device>]
    
    private var gradientSums: [Buffer<Element, Device>]
    private var updateSums: [Buffer<Element, Device>]
    private var cache1: [Buffer<Element, Device>]
    private var cache2: [Buffer<Element, Device>]
    private var cache3: [Buffer<Element, Device>]
    private var isInitialized = false
    
    /// Adadelta optimizer
    ///
    /// - Parameters:
    ///   - parameters: Parameters that should be optimized
    ///   - learningRate: Initial learning rate. The learning rate is only applied in the first step and ignored in every subsequent step
    ///   - gamma: Decay rate for exponential decaying filter applied to sum of squared gradients and sum of squared weight updates
    ///   - epsilon: Smoothing value applied to divisions to prevent NaNs
    public init(parameters: [Tensor<Element, Device>], learningRate: Element = 0.001, gamma: Element = 0.9, epsilon: Element = 1e-8) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
        self.gradientSums = []
        self.cache1 = []
        self.cache2 = []
        self.cache3 = []
        self.updateSums = []
        
        reset()
    }
    
    deinit {
        gradientSums.forEach(Device.Memory.free)
        updateSums.forEach(Device.Memory.free)
        cache1.forEach(Device.Memory.free)
        cache2.forEach(Device.Memory.free)
        cache3.forEach(Device.Memory.free)
    }
    
    public func step() {
        for i in 0 ..< parameters.count {
            let param = parameters[i]
            let sum = gradientSums[i]
            let updateSum = updateSums[i]
            let c1 = cache1[i]
            let c2 = cache2[i]
            let c3 = cache3[i]
            
            guard let gradient = param.gradient else {
                continue
            }
            
            // Compute exponentially decaying sum of squared gradients
            Device.Engine.vSquare(values: gradient, result: c1, count: param.count)
            Device.Engine.vsMul(lhs: c1, rhs: (1 - gamma), result: c1, count: param.count)
            Device.Engine.vsMulVAdd(lhs: sum, rhs: gamma, add: c1, result: sum, count: param.count)
            
            // Compute value update
            if isInitialized {
                // If initialized, use normal Adadelta update rule
                // -sqrt(updateSum+eps) / sqrt(sum+eps) * gradient
                
                // sqrt(sum + eps) -> c2
                Device.Engine.vsAdd(lhs: sum, rhs: epsilon, result: c2, count: param.count)
                Device.Engine.sqrt(val: c2, result: c2, count: param.count)
                
                // -sqrt(updateSum + eps) -> c3
                Device.Engine.vsAdd(lhs: updateSum, rhs: epsilon, result: c3, count: param.count)
                Device.Engine.sqrt(val: c3, result: c3, count: param.count)
                Device.Engine.vNeg(val: c3, result: c3, count: param.count)
                
                // c3 / c2 * gradient
                Device.Engine.vDiv(lhs: c3, rhs: c2, result: c2, count: param.count)
                Device.Engine.vMul(lhs: c2, rhs: gradient, result: c2, count: param.count)
                
                // Weight update
                Device.Engine.vAdd(lhs: c2, rhs: param.values, result: param.values, count: param.count)
                
                // Update updateSum
                Device.Engine.vSquare(values: c2, result: c2, count: param.count)
                Device.Engine.vsMul(lhs: c2, rhs: (1 - gamma), result: c2, count: param.count)
                Device.Engine.vsMulVAdd(lhs: updateSum, rhs: gamma, add: c2, result: updateSum, count: param.count)
            } else {
                // Otherwise use initial learning rate and default to RMSProp
                // -learningRate / sqrt(sum + epsilon)
                Device.Engine.vsAdd(lhs: sum, rhs: epsilon, result: c1, count: param.count)
                Device.Engine.sqrt(val: c1, result: c1, count: param.count)
                Device.Engine.svDiv(lhs: -learningRate, rhs: c1, result: c1, count: param.count)
                
                // Computing weight update
                Device.Engine.vMul(lhs: c1, rhs: gradient, result: c1, count: param.count)
                
                // Squared weight update -> updateSum
                Device.Engine.vSquare(values: c1, result: updateSum, count: param.count)
                
                // Perform weight update
                Device.Engine.vAdd(lhs: c1, rhs: param.values, result: param.values, count: param.count)
            }
        }
        
        isInitialized = true
    }
    
    public func reset() {
        gradientSums.forEach(Device.Memory.free)
        updateSums.forEach(Device.Memory.free)
        cache1.forEach(Device.Memory.free)
        cache2.forEach(Device.Memory.free)
        cache3.forEach(Device.Memory.free)
        
        
        gradientSums = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        cache1 = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        cache2 = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        cache3 = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        updateSums = parameters.map { param in
            Device.Memory.allocateBuffer(withCapacity: param.count, type: Element.self)
        }
        
        for gs in [gradientSums, updateSums, cache1, cache2, cache3].joined() {
            Device.Engine.fill(value: 0, result: gs, count: gs.count)
        }
    }
    
    public func zeroGradient() {
        parameters.forEach {
            $0.zeroGradient()
        }
    }
}

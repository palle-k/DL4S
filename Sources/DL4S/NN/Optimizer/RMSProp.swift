//
//  RMSProp.swift
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

public struct RMSProp<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    public var learningRate: ParamTensor
    public var gamma: ParamTensor
    public var epsilon: ParamTensor
    
    private var gradientSums: [ParamTensor]
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]
    
    public init(model: Layer, learningRate: ParamTensor = 0.001, gamma: ParamTensor = 0.9, epsilon: ParamTensor = 1e-8) {
        self.model = model
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
    }
    
    public mutating func reset() {
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let path = paths[i]
            let grad = gradients[i].detached()
            
            gradientSums[i] = gamma * gradientSums[i] + (1 - gamma) * (grad * grad)
            
            let a = learningRate / sqrt(gradientSums[i] + epsilon)
            let delta = a * grad
            
            model[keyPath: path] -= delta
            model[keyPath: path].discardContext()
        }
    }
}

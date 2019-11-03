//
//  Adadelta.swift
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

/// Adadelta Optimizer
///
/// Follows [Matthew D. Zeiler - Adadelta: An adaptive learning rate method](https://arxiv.org/pdf/1212.5701.pdf)
public struct Adadelta<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    /// Initial learning rate scaling factor. Only used in first optimization step after initialization or reset.
    public var learningRate: ParamTensor
    
    /// Exponential decay rate for squared gradient history
    public var gamma: ParamTensor
    
    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor
    
    private var gradientSums: [ParamTensor]
    private var updateSums: [ParamTensor]
    
    private var isInitialized = false
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]

    /// Adadelta Optimizer
    ///
    /// Follows [Matthew D. Zeiler - Adadelta: An adaptive learning rate method](https://arxiv.org/pdf/1212.5701.pdf)
    /// - Parameters:
    ///   - model: Model to optimize
    ///   - learningRate: Initial learning rate, ignored after first step
    ///   - gamma: Exponential decay rate for squared gradient history
    ///   - epsilon: Normalization scalar added to divisors
    public init(model: Layer, learningRate: ParamTensor = 0.001, gamma: ParamTensor = 0.9, epsilon: ParamTensor = 1e-8) {
        self.model = model
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.updateSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
    }
    
    /// Resets the state of the optimizer
    public mutating func reset() {
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.updateSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        isInitialized = false
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let path = paths[i]
            let grad = gradients[i].detached()
            
            let addedToGradSum = (1 - gamma) * (grad * grad)
            gradientSums[i] = gamma * gradientSums[i] + addedToGradSum
            
            if isInitialized {
                let a = sqrt(gradientSums[i] + epsilon)
                let b = sqrt(updateSums[i] + epsilon)
                
                let delta = b / a * grad
                model[keyPath: path] -= delta
                
                let addedToUpdateSum = (1 - gamma) * (delta * delta)
                updateSums[i] = gamma * updateSums[i] + addedToUpdateSum
                
            } else {
                let a = learningRate / sqrt(gradientSums[i] + epsilon)
                let delta = a * grad
                model[keyPath: path] -= delta
                
                updateSums[i] = delta * delta
                
                isInitialized = true
            }
            
            model[keyPath: path].discardContext()
        }
    }
}

extension Adadelta: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.model = try container.decode(Layer.self, forKey: .model)
        self.learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        self.gamma = try container.decode(ParamTensor.self, forKey: .gamma)
        self.epsilon = try container.decode(ParamTensor.self, forKey: .epsilon)
        self.gradientSums = try container.decode([ParamTensor].self, forKey: .gradientSums)
        self.updateSums = try container.decode([ParamTensor].self, forKey: .updateSums)
        self.isInitialized = try container.decode(Bool.self, forKey: .isInitialized)
        self.paths = self.model.parameterPaths
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(gamma, forKey: .gamma)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(gradientSums, forKey: .gradientSums)
        try container.encode(updateSums, forKey: .updateSums)
        try container.encode(isInitialized, forKey: .isInitialized)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case learningRate
        case gamma
        case epsilon
        case gradientSums
        case updateSums
        case isInitialized
    }
}

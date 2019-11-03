//
//  Adagrad.swift
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


/// Adagrad optimizer
///
/// Follows [Duchi et al - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
public struct Adagrad<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    /// Learning rate scaling factor
    public var learningRate: ParamTensor
    
    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor
    private var gradientSums: [ParamTensor]
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]

    /// Adagrad optimizer
    ///
    /// Follows [Duchi et al - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    /// - Parameters:
    ///   - model: Model to optimize
    ///   - learningRate: Learning rate scaling factor
    ///   - epsilon: Normalization scalar added to divisors
    public init(model: Layer, learningRate: ParamTensor, epsilon: ParamTensor = 1e-8) {
        self.model = model
        self.learningRate = learningRate
        self.epsilon = epsilon
        
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
    }
    
    /// Resets the state of the optimizer
    public mutating func reset() {
        self.gradientSums = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let path = paths[i]
            let grad = gradients[i].detached()
            
            gradientSums[i] += grad * grad
            
            let adaptiveLearningRate = learningRate / sqrt(gradientSums[i] + epsilon)
            
            model[keyPath: path] -= adaptiveLearningRate * grad
            model[keyPath: path].discardContext()
        }
    }
}


extension Adagrad: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.model = try container.decode(Layer.self, forKey: .model)
        self.learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        self.epsilon = try container.decode(ParamTensor.self, forKey: .epsilon)
        self.gradientSums = try container.decode([ParamTensor].self, forKey: .gradientSums)
        self.paths = self.model.parameterPaths
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(gradientSums, forKey: .gradientSums)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case learningRate
        case epsilon
        case gradientSums
    }
}

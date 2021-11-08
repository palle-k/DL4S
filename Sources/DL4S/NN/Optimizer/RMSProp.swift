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

/// Root mean square optimizer
///
/// Unpublished, proposed by [Geoffrey Hinton - Neural Networks for Machine Learning](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
public struct RMSProp<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    /// Learning rate scaling factor
    public var learningRate: ParamTensor
    
    /// Exponential decay rate for gradient history
    public var gamma: ParamTensor
    
    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor
    
    private var gradientSums: [ParamTensor]
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]

    /// Root mean square optimizer
    ///
    /// Unpublished, proposed by [Geoffrey Hinton - Neural Networks for Machine Learning](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    /// - Parameters:
    ///   - model: Model to optimize
    ///   - learningRate: Learning rate scaling factor
    ///   - gamma: Exponential decay rate for gradient history
    ///   - epsilon: Normalization scalar added to divisors
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
            
            let addedToGradientSum = (1 - gamma) * (grad * grad)
            gradientSums[i] = gamma * gradientSums[i] + addedToGradientSum
            
            let a = learningRate / sqrt(gradientSums[i] + epsilon)
            let delta = a * grad
            
            model[keyPath: path] -= delta
            model[keyPath: path].discardContext()
        }
    }
}

extension RMSProp: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.model = try container.decode(Layer.self, forKey: .model)
        self.learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        self.gamma = try container.decode(ParamTensor.self, forKey: .gamma)
        self.epsilon = try container.decode(ParamTensor.self, forKey: .epsilon)
        self.gradientSums = try container.decode([ParamTensor].self, forKey: .gradientSums)
        self.paths = self.model.parameterPaths
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(gamma, forKey: .gamma)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(gradientSums, forKey: .gradientSums)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case learningRate
        case gamma
        case epsilon
        case gradientSums
    }
}

//
//  File.swift
//  
//
//  Created by Palle Klewitz on 20.09.20.
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

/// Gradient descent optimizer with momentum
public struct Momentum<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    private var velocities: [ParamTensor]
    
    /// Learning rate with which to move along the gradient
    public var learningRate: ParamTensor
    
    /// Decay rate of momentum that is built up, when subsequent gradient updates move in the same direction
    public var momentum: ParamTensor
    private var paths: [WritableKeyPath<Layer, ParamTensor>]
    
    /// Gradient descent optimizer with momentum
    /// - Parameters:
    ///   - model: Model to optimize
    ///   - learningRate: Learning rate with which to move along the gradient
    ///   - momentum: Decay rate of momentum that is built up, when subsequent gradient updates move in the same direction
    public init(model: Layer, learningRate: ParamTensor, momentum: ParamTensor = 0.8) {
        self.model = model
        self.learningRate = learningRate
        self.momentum = momentum
        
        self.velocities = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
    }
    
    /// Resets the state of the optimizer
    public mutating func reset() {
        self.velocities = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let keyPath = paths[i]
            velocities[i] = velocities[i] * momentum + learningRate * gradients[i]
            model[keyPath: keyPath] -= velocities[i]
            model[keyPath: keyPath].discardContext()
        }
    }
}

extension Momentum: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        model = try container.decode(Layer.self, forKey: .model)
        momentum = try container.decode(ParamTensor.self, forKey: .momentum)
        learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        velocities = try container.decode([ParamTensor].self, forKey: .velocities)
        
        paths = model.parameterPaths
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(momentum, forKey: .momentum)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(velocities, forKey: .velocities)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case velocities
        case learningRate
        case momentum
    }
}

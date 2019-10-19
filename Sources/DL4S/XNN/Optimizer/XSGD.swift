//
//  XSGD.swift
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

public struct XSGD<Layer: XLayer>: XOptimizer {
    public typealias ParamTensor = XTensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    public var learningRate: ParamTensor
    private var paths: [WritableKeyPath<Layer, ParamTensor>]
    
    public init(model: Layer, learningRate: ParamTensor) {
        self.model = model
        self.learningRate = learningRate
        self.paths = model.parameterPaths
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for (keyPath, grad) in zip(paths, gradients) {
            model[keyPath: keyPath] -= learningRate * grad
            model[keyPath: keyPath].discardContext()
        }
    }
}

public struct XMomentum<Layer: XLayer>: XOptimizer {
    public typealias ParamTensor = XTensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    private var velocities: [ParamTensor]
    
    public var learningRate: ParamTensor
    public var momentum: ParamTensor
    private var paths: [WritableKeyPath<Layer, ParamTensor>]
    
    public init(model: Layer, learningRate: ParamTensor, momentum: ParamTensor = 0.8) {
        self.model = model
        self.learningRate = learningRate
        self.momentum = momentum
        
        self.velocities = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
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

extension XSGD: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        model = try container.decode(Layer.self, forKey: .model)
        learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        
        paths = model.parameterPaths
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(learningRate, forKey: .learningRate)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case learningRate
    }
}

extension XMomentum: Codable where Layer: Codable {
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

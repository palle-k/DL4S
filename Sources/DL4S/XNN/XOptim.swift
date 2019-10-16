//
//  XOptim.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
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

public protocol XOptimizer {
    associatedtype Layer: XLayer
    
    var model: Layer { get }
    mutating func update(along gradients: [XTensor<Layer.Parameter, Layer.Device>])
}

public struct XSGD<Layer: XLayer>: XOptimizer {
    public private(set) var model: Layer
    public var learningRate: XTensor<Layer.Parameter, Layer.Device>
    
    public init(model: Layer, learningRate: XTensor<Layer.Parameter, Layer.Device>) {
        self.model = model
        self.learningRate = learningRate
    }
    
    public mutating func update(along gradients: [XTensor<Layer.Parameter, Layer.Device>]) {
        for (keyPath, grad) in zip(Layer.parameters, gradients) {
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
        self.paths = Layer.parameters
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

public struct XAdam<Layer: XLayer>: XOptimizer {
    public typealias ParamTensor = XTensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    public var learningRate: ParamTensor
    public var beta1: ParamTensor
    public var beta2: ParamTensor
    public var epsilon: ParamTensor
    
    private var beta1t: ParamTensor
    private var beta2t: ParamTensor
    
    private var firstMoments: [ParamTensor]
    private var secondMoments: [ParamTensor]
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]
    
    public init(model: Layer, learningRate: ParamTensor, beta1: ParamTensor = 0.9, beta2: ParamTensor = 0.999, epsilon: ParamTensor = 1e-8) {
        self.model = model
        
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.beta1t = beta1
        self.beta2t = beta2
        
        self.epsilon = epsilon
        
        self.firstMoments = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
        self.secondMoments = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
        self.paths = Layer.parameters
    }
    
    public mutating func reset() {
        beta1t = beta1
        beta2t = beta2
        
        self.firstMoments = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
        self.secondMoments = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let path = paths[i]
            let grad = gradients[i].detached()
            
            firstMoments[i] = firstMoments[i] * beta1 + grad * (1 - beta1)
            secondMoments[i] = secondMoments[i] * beta2 + (grad * grad) * (1 - beta2)
            
            let m_car_t = firstMoments[i] / (1 - beta1t)
            let v_car_t = secondMoments[i] / (1 - beta2t)
            
            let delta = learningRate / (v_car_t.sqrt() + epsilon) * m_car_t
            model[keyPath: path] -= delta
            model[keyPath: path].discardContext()
        }
        
        beta1t *= beta1
        beta2t *= beta2
    }
}

extension XSGD: Codable where Layer: Codable {}

extension XMomentum: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        model = try container.decode(Layer.self, forKey: .model)
        momentum = try container.decode(ParamTensor.self, forKey: .momentum)
        learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        velocities = try container.decode([ParamTensor].self, forKey: .velocities)
        
        paths = Layer.parameters
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

extension XAdam: Codable where Layer: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        model = try container.decode(Layer.self, forKey: .model)
        learningRate = try container.decode(ParamTensor.self, forKey: .learningRate)
        beta1 = try container.decode(ParamTensor.self, forKey: .beta1)
        beta2 = try container.decode(ParamTensor.self, forKey: .beta2)
        beta1t = try container.decode(ParamTensor.self, forKey: .beta1t)
        beta2t = try container.decode(ParamTensor.self, forKey: .beta2t)
        epsilon = try container.decode(ParamTensor.self, forKey: .epsilon)
        firstMoments = try container.decode([ParamTensor].self, forKey: .firstMoments)
        secondMoments = try container.decode([ParamTensor].self, forKey: .secondMoments)
        
        paths = Layer.parameters
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        try container.encode(model, forKey: .model)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(beta1, forKey: .beta1)
        try container.encode(beta2, forKey: .beta2)
        try container.encode(beta1t, forKey: .beta1t)
        try container.encode(beta2t, forKey: .beta2t)
        try container.encode(epsilon, forKey: .epsilon)
        try container.encode(firstMoments, forKey: .firstMoments)
        try container.encode(secondMoments, forKey: .secondMoments)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case firstMoments
        case secondMoments
        case learningRate
        case beta1
        case beta2
        case beta1t
        case beta2t
        case epsilon
    }
}

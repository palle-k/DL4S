//
//  Adam.swift
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

/// Adam optimizer (Adaptive moment estimation)
///
/// Follows [Kingma et al. - Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
public struct Adam<Layer: LayerType>: Optimizer {
    public typealias ParamTensor = Tensor<Layer.Parameter, Layer.Device>
    
    public private(set) var model: Layer
    
    public let useAMSGrad: Bool
    
    /// Learning rate scaling factor
    public var learningRate: ParamTensor
    
    /// Exponential decay rate for first moment
    public var beta1: ParamTensor
    
    /// Exponential decay rate for second moment
    public var beta2: ParamTensor
    
    /// Normalization scalar added to divisors
    public var epsilon: ParamTensor
    
    private var beta1t: ParamTensor
    private var beta2t: ParamTensor
    
    private var firstMoments: [ParamTensor]
    private var secondMoments: [ParamTensor]
    private var secondMomentMax: [ParamTensor]?
    
    private var paths: [WritableKeyPath<Layer, ParamTensor>]

    /// Adam optimizer (Adaptive moment estimation)
    ///
    /// Follows [Kingma et al. - Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)
    /// - Parameters:
    ///   - model: Model to optimize
    ///   - learningRate: Learning rate scaling factor
    ///   - beta1: Exponential decay rate for first moment
    ///   - beta2: Exponential decay rate for second moment
    ///   - epsilon: Normalization scalar added to divisors
    public init(model: Layer, learningRate: ParamTensor, useAMSGrad: Bool = false, beta1: ParamTensor = 0.9, beta2: ParamTensor = 0.999, epsilon: ParamTensor = 1e-8) {
        self.model = model
        
        self.useAMSGrad = useAMSGrad
        
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.beta1t = beta1
        self.beta2t = beta2
        
        self.epsilon = epsilon
        
        self.firstMoments = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.secondMoments = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.paths = model.parameterPaths
        
        if useAMSGrad {
            self.secondMomentMax = model.parameters.map {
                Tensor(repeating: 0, shape: $0.shape)
            }
        }
    }
    
    /// Resets the state of the optimizer
    public mutating func reset() {
        beta1t = beta1
        beta2t = beta2
        
        self.firstMoments = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
        self.secondMoments = model.parameters.map {
            Tensor(repeating: 0, shape: $0.shape)
        }
    }
    
    public mutating func update(along gradients: [ParamTensor]) {
        for i in paths.indices {
            let path = paths[i]
            let grad = gradients[i].detached()
            
            let addedToFirstMoment = grad * (1 - beta1)
            firstMoments[i] = firstMoments[i] * beta1 + addedToFirstMoment
            
            let addedToSecondMoment = (grad * grad) * (1 - beta2)
            secondMoments[i] = secondMoments[i] * beta2 + addedToSecondMoment
            
            let v_t_norm: ParamTensor
            if useAMSGrad, let secondMomentMax = self.secondMomentMax {
                let v_t_max = secondMomentMax[i]
                v_t_norm = Tensor.max(v_t_max, secondMoments[i])
            } else {
                v_t_norm = secondMoments[i]
            }
            
            let m_car_t = firstMoments[i] / (1 - beta1t)
            let v_car_t = v_t_norm / (1 - beta2t)
            
            let delta = learningRate / (v_car_t.sqrt() + epsilon) * m_car_t
            model[keyPath: path] -= delta
            model[keyPath: path].discardContext()
        }
        
        beta1t *= beta1
        beta2t *= beta2
    }
}


extension Adam: Codable where Layer: Codable {
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
        if container.contains(.useAMSGrad), try container.decode(Bool.self, forKey: .useAMSGrad) {
            useAMSGrad = true
            secondMomentMax = try container.decode([ParamTensor].self, forKey: .secondMomentMax)
        } else {
            useAMSGrad = false
            secondMomentMax = nil
        }
        
        paths = model.parameterPaths
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
        try container.encode(useAMSGrad, forKey: .useAMSGrad)
        try container.encode(secondMomentMax, forKey: .secondMomentMax)
    }
    
    private enum CodingKeys: String, CodingKey {
        case model
        case firstMoments
        case secondMoments
        case useAMSGrad
        case secondMomentMax
        case learningRate
        case beta1
        case beta2
        case beta1t
        case beta2t
        case epsilon
    }
}

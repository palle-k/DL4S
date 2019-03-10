//
//  Optim.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


public protocol Optimizer {
    associatedtype Element: NumericType
    
    var parameters: [Tensor<Element>] { get }
    
    func step()
    func reset()
    func zeroGradient()
}

public class SGDOptimizer<Element: NumericType>: Optimizer {
    public var learningRate: Element
    public var regularizationL2: Element = 0
    
    public let parameters: [Tensor<Element>]
    
    public init(parameters: [Tensor<Element>], learningRate: Element) {
        self.learningRate = learningRate
        self.parameters = parameters
    }
    
    public func step() {
        for parameter in self.parameters {
            guard let gradient = parameter.gradient else {
                continue
            }
            Element.vsMulVAdd(lhs: gradient.immutable, rhs: -self.learningRate, add: parameter.values.immutable, result: parameter.values, count: parameter.count)
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


public class MomentumOptimizer<Element: NumericType>: Optimizer {
    public var learningRate: Element
    public var momentum: Element
    
    public let parameters: [Tensor<Element>]
    private let momentumParams: [UnsafeMutableBufferPointer<Element>]
    
    public init(parameters: [Tensor<Element>], learningRate: Element, momentum: Element = 0.8) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.parameters = parameters
        self.momentumParams = parameters.map {Allocator.allocate(count: $0.count)}
        
        for m in self.momentumParams {
            m.assign(repeating: 0)
        }
    }
    
    deinit {
        self.momentumParams.forEach(Allocator.free)
    }
    
    public func reset() {
        for m in self.momentumParams {
            m.assign(repeating: 0)
        }
    }
    
    public func step() {
        for (param, momentumParam) in zip(parameters, momentumParams) {
            guard let gradient = param.gradient else {
                continue
            }
            // m = momentum * m
            Element.vsMul(lhs: momentumParam.immutable, rhs: self.momentum, result: momentumParam, count: param.count)
            // m = lr * grad + m = lr * grad + momentum * m
            Element.vsMulVAdd(lhs: gradient.immutable, rhs: self.learningRate, add: momentumParam.immutable, result: momentumParam, count: param.count)
            // values = values - m
            Element.vSub(lhs: param.values.immutable, rhs: momentumParam.immutable, result: param.values, count: param.count)
        }
    }
    
    public func zeroGradient() {
        for parameter in self.parameters {
            parameter.zeroGradient()
        }
    }
}


public class Adam<Element: NumericType>: Optimizer {
    public let parameters: [Tensor<Element>]
    
    public var learningRate: Element
    public var beta1: Element
    public var beta2: Element
    public var epsilon: Element
    
    private var beta1t: Element
    private var beta2t: Element
    
    private let firstMoment: [UnsafeMutableBufferPointer<Element>]
    private let secondMoment: [UnsafeMutableBufferPointer<Element>]
    private let cache: [UnsafeMutableBufferPointer<Element>]
    private let cache2: [UnsafeMutableBufferPointer<Element>]
    
    public private(set) var iteration: Int = 0
    
    public init(parameters: [Tensor<Element>], learningRate: Element, beta1: Element = 0.9, beta2: Element = 0.999, epsilon: Element = 1e-8) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.firstMoment = parameters.map {Allocator.allocate(count: $0.count)}
        self.secondMoment = parameters.map {Allocator.allocate(count: $0.count)}
        self.cache = parameters.map {Allocator.allocate(count: $0.count)}
        self.cache2 = parameters.map {Allocator.allocate(count: $0.count)}
        
        self.beta1t = beta1
        self.beta2t = beta2
        
        for m in [firstMoment, secondMoment, cache, cache2].joined() {
            m.assign(repeating: 0)
        }
    }
    
    deinit {
        [firstMoment, secondMoment, cache, cache2].joined().forEach(Allocator.free)
    }
    
    public func reset() {
        beta1t = beta1
        beta2t = beta2
        for m in [firstMoment, secondMoment, cache, cache2].joined() {
            m.assign(repeating: 0)
        }
    }
    
    public func step() {
        for (i, param) in parameters.enumerated() {
            guard let g = param.gradient else {
                continue
            }
            
            let m = firstMoment[i]
            let v = secondMoment[i]
            let c = cache[i]
            let c2 = cache2[i]
            
            // cache = (1 - beta1) * gradient
            Element.vsMul(lhs: g.immutable, rhs: (1 - beta1), result: c, count: param.count)
            // m = beta1 * m + cache = beta1 * m + (1 - beta1) * gradient
            Element.vsMulVAdd(lhs: m.immutable, rhs: beta1, add: c.immutable, result: m, count: param.count)
            
            // cache = gradient ^ 2
            Element.vSquare(values: g.immutable, result: c, count: param.count)
            // cache = (1 - beta2) * gradient ^ 2
            Element.vsMul(lhs: c.immutable, rhs: (1 - beta2), result: c, count: param.count)
            // v = beta2 * v + (1 - beta2) * gradient ^ 2
            Element.vsMulVAdd(lhs: v.immutable, rhs: beta2, add: c.immutable, result: v, count: param.count)
            
            // c (m^) = m / (1 - beta1)
            Element.vsMul(lhs: m.immutable, rhs: 1 / (1 - beta1t), result: c, count: param.count)
            
            // c2 (^v) = v / (1 - beta2)
            Element.vsMul(lhs: v.immutable, rhs: 1 / (1 - beta2t), result: c2, count: param.count)
            
            // c2 = sqrt(c2)
            Element.sqrt(val: c2.immutable, result: c2, count: param.count)
            // c2 = c2 + eps
            Element.vsAdd(lhs: c2.immutable, rhs: epsilon, result: c2, count: param.count)
            // c2 = -lr / c2
            Element.svDiv(lhs: -learningRate, rhs: c2.immutable, result: c2, count: param.count)
            
            // weights = c * c2 + weights = -lr / (sqrt(c2) + eps) * c + weights
            Element.vMA(lhs: c.immutable, rhs: c2.immutable, add: param.values.immutable, result: param.values, count: param.count)
        }
        
        // beta_t = pow(beta_, iteration)
        beta1t = beta1t * beta1
        beta2t = beta2t * beta2
    }
    
    public func zeroGradient() {
        for parameter in self.parameters {
            parameter.zeroGradient()
        }
    }
    
}

//
//  XOptim.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

import Foundation

public protocol XOptimizer {
    associatedtype Layer: XLayer
    
    var model: Layer { get }
    mutating func update(along gradients: [XTensor<Layer.Parameter, Layer.Device>])
}

public struct XSGD<Layer: XLayer>: XOptimizer {
    public private(set) var model: Layer
    public var learningRate: XTensor<Layer.Parameter, Layer.Device>
    
    public mutating func update(along gradients: [XTensor<Layer.Parameter, Layer.Device>]) {
        for (keyPath, grad) in zip(type(of: model).parameters, gradients) {
            model[keyPath: keyPath] -= learningRate * grad
            model[keyPath: keyPath].discardContext()
        }
    }
}

public struct XMomentum<Layer: XLayer>: XOptimizer {
    public private(set) var model: Layer
    private var velocities: [XTensor<Layer.Parameter, Layer.Device>]
    
    public var learningRate: XTensor<Layer.Parameter, Layer.Device>
    public var momentum: XTensor<Layer.Parameter, Layer.Device>
    private var paths: [WritableKeyPath<Layer, XTensor<Layer.Parameter, Layer.Device>>]
    
    public init(model: Layer, learningRate: XTensor<Layer.Parameter, Layer.Device>, momentum: XTensor<Layer.Parameter, Layer.Device> = 0.8) {
        self.model = model
        self.learningRate = learningRate
        self.momentum = momentum
        
        self.velocities = model.parameters.map {
            XTensor(repeating: 0, shape: $0.shape)
        }
        self.paths = type(of: model).parameters
    }
    
    public mutating func update(along gradients: [XTensor<Layer.Parameter, Layer.Device>]) {
        for i in 0 ..< paths.count {
            let keyPath = paths[i]
            velocities[i] = velocities[i] * momentum + learningRate * gradients[i]
            model[keyPath: keyPath] -= velocities[i]
            model[keyPath: keyPath].discardContext()
        }
    }
}

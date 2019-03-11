//
//  LayerTypes.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
//

import Foundation


public class Relu<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        return relu(inputs[0])
    }
}


public class Tanh<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        return tanh(inputs[0])
    }
}


public class Sigmoid<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let out = sigmoid(inputs[0])
        assert(out.shape == inputs[0].shape)
        return out
    }
}


public class Softmax<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        // TODO: Normalize inputs to make exp more stable
        let norm = inputs[0] - max(inputs[0]).detached()
        let e = exp(norm)
        let s = sum(e, axis: 1)
        return (e.T / s).T
    }
}


public class Flatten<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return inputs[0].view(as: inputs[0].shape[0], -1)
    }
}


public class Reshape<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var trainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var outputShape: [Int]
    
    public init(shape: [Int]) {
        self.outputShape = shape
    }
    
    public convenience init(shape: Int...) {
        self.init(shape: shape)
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        let x = inputs[0]
        return x.view(as: [x.shape[0]] + self.outputShape)
    }
}

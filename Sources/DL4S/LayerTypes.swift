//
//  LayerTypes.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
//

import Foundation




public class Dense<Element: RandomizableType>: Layer {
    let w: Vector<Element>
    let b: Vector<Element>
    
    public var parameters: [Vector<Dense.Element>] {
        return trainable ? [w, b] : []
    }
    
    public var trainable: Bool = true
    
    public var inputFeatures: Int {
        return w.shape[0]
    }
    
    public var outputFeatures: Int {
        return w.shape[1]
    }
    
    public init(inputFeatures: Int, outputFeatures: Int) {
        w = Vector(repeating: 0.5, shape: inputFeatures, outputFeatures)
        b = Vector(repeating: 0, shape: outputFeatures)
        
        Random.fillNormal(w, mean: 0, stdev: 1 / Element(inputFeatures))
    }
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        precondition(inputs.count == 1)
        return mmul(inputs[0].view(as: -1, inputFeatures), w) + b
    }
}

public class Relu<Element: NumericType>: Layer {
    public var parameters: [Vector<Element>] {
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
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        precondition(inputs.count == 1)
        return relu(inputs[0])
    }
}


public class Tanh<Element: NumericType>: Layer {
    public var parameters: [Vector<Element>] {
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
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        precondition(inputs.count == 1)
        return tanh(inputs[0])
    }
}


public class Sigmoid<Element: NumericType>: Layer {
    public var parameters: [Vector<Element>] {
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
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        precondition(inputs.count == 1)
        return 1 / (1 + exp(-inputs[0]))
    }
}


public class Softmax<Element: NumericType>: Layer {
    public var parameters: [Vector<Element>] {
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
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        precondition(inputs.count == 1)
        let norm = inputs[0] - max(inputs[0])
        let e = exp(norm)
        let s = sum(e)
        return e / s
    }
}


public class Sequential<Element: NumericType>: Layer {
    public let layers: [AnyLayer<Element>]
    
    public var trainable: Bool {
        get {
            return layers.contains(where: {$0.trainable})
        }
        set {
            layers.forEach {$0.trainable = newValue}
        }
    }
    
    public var parameters: [Vector<Element>] {
        return layers.flatMap {$0.parameters}
    }
    
    public init(_ layers: AnyLayer<Element>...) {
        self.layers = layers
    }
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        return layers.reduce(inputs[0]) {$1.forward([$0])}
    }
}

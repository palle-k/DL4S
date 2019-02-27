//
//  Layers.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation

public protocol Layer {
    var allParameters: [Variable] { get }
    
    func forward(_ input: [Variable]) -> [Variable]
}

public class Dense: Layer {
    public var allParameters: [Variable] {
        return Array(params.joined())
    }
    
    let params: [[Variable]]
    let bias: [Variable]
    
    init(inputs: Int, outputs: Int, weightScale: Float = 0.01) {
        params = [[Variable]](repeating: 0, rows: outputs, columns: inputs).fillRandomly(-weightScale ... weightScale)
        bias = [Variable](repeating: Float(0), count: outputs)
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return params * input .+ bias
    }
}

public class Tanh: Layer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return tanh(input)
    }
}

public class Sigmoid: Layer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return sigmoid(input)
    }
}

public class Relu: Layer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return relu(input)
    }
}

public class Sequential: Layer {
    public var allParameters: [Variable] {
        return layers.flatMap {$0.allParameters}
    }
    
    let layers: [Layer]
    
    init(_ layers: Layer...) {
        self.layers = layers
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return layers.reduce(input, {$1.forward($0)})
    }
}

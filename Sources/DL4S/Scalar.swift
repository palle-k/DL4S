//
//  Layers.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation

public protocol ScalarLayer {
    var allParameters: [Variable] { get }
    
    func forward(_ input: [Variable]) -> [Variable]
}

public class ScalarDense: ScalarLayer {
    public var allParameters: [Variable] {
        return Array(params.joined())
    }
    
    let params: [[Variable]]
    let bias: [Variable]
    
    init(inputs: Int, outputs: Int, weightScale: Float = 0.01) {
        params = [[Variable]](repeating: 0.5, rows: outputs, columns: inputs).fillRandomly(-weightScale ... weightScale)
        bias = [Variable](repeating: Float(0), count: outputs)
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return params * input .+ bias
    }
}

public class ScalarTanh: ScalarLayer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return tanh(input)
    }
}

public class ScalarSigmoid: ScalarLayer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return sigmoid(input)
    }
}

public class ScalarRelu: ScalarLayer {
    public var allParameters: [Variable] {
        return []
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return relu(input)
    }
}

public class ScalarSequential: ScalarLayer {
    public var allParameters: [Variable] {
        return layers.flatMap {$0.allParameters}
    }
    
    let layers: [ScalarLayer]
    
    init(_ layers: ScalarLayer...) {
        self.layers = layers
    }
    
    public func forward(_ input: [Variable]) -> [Variable] {
        return layers.reduce(input, {$1.forward($0)})
    }
}

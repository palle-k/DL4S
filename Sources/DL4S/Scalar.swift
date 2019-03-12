//
//  Layers.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
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

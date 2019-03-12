//
//  LayerTypes.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
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


public class Relu<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var isTrainable: Bool {
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
    
    public var isTrainable: Bool {
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
    
    public var isTrainable: Bool {
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
    
    public var isTrainable: Bool {
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
    
    public var isTrainable: Bool {
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
    
    public var isTrainable: Bool {
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

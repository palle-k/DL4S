//
//  Activation.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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

/// Element-wise hyperbolic tangent activation layer.
public struct Tanh<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    
    /// Element-wise hyperbolic tangent activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.tanh()
    }
}

/// Element-wise sigmoid activation layer.
public struct Sigmoid<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    
    /// Element-wise sigmoid activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Sigmoid") {
            inputs.sigmoid()
        }
    }
}

/// Element-wise rectified linear unit activation layer.
public struct Relu<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}

    /// Element-wise rectified linear unit activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.rectifiedLinear()
    }
}

/// Element-wise leaky linear rectified unit activation layer.
public struct LeakyRelu<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    public var leakage: Element
    
    /// Element-wise leaky rectified linear unit activation layer.
    public init(leakage: Element) {
        self.leakage = leakage
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "LeakyRelu") {
            inputs.leakyRectifiedLinear(leakage: Tensor(leakage))
        }
    }
}

/// Log Softmax activation layer
public struct LogSoftmax<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Softmax activation layer
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "LogSoftmax") {
            inputs.logSoftmax()
        }
    }
}

/// Softmax activation layer
public struct Softmax<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    
    /// Softmax activation layer
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Softmax") {
            inputs.softmax()
        }
    }
}

/// Element-wise gaussian error linear unit activation layer.
public struct Gelu<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }

    /// Element-wise Gaussian error linear unit activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "gelu") {
            inputs.gaussianErrorLinear()
        }
    }
}

/// Element-wise Swish activation layer.
public struct Swish<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        beta.requiresGradient ? [\.beta] : []
    }
    public var parameters: [Tensor<Element, Device>] {
        beta.requiresGradient ? [beta] : []
    }
    
    public var beta: Tensor<Element, Device>

    /// Element-wise Swish activation layer with learnable beta parameter
    public init(trainableWithChannels channels: Int) {
        beta = Tensor(repeating: 1, shape: [channels], requiresGradient: true)
    }
    
    /// Element-wise Swish activation layer with fixed beta parameter
    public init(fixedWithBeta beta: Element = 1) {
        self.beta = Tensor(beta)
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "swish") {
            inputs.swishActivated(beta: beta)
        }
    }
}

/// Element-wise Mish activation layer.
public struct Mish<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}

    /// Element-wise Mish activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "mish") {
            inputs.mishActivated()
        }
    }
}

/// Element-wise LiSHT activation layer.
public struct LiSHT<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}

    /// Element-wise LiSHT activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "lisht") {
            inputs.lishtActivated()
        }
    }
}

/// Layer wrapping an arbitrary transform provided by a closure.
public struct Lambda<Inputs, Outputs, Element: NumericType, Device: DeviceType>: LayerType {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] {[]}
    
    /// Transformation performed by the layer
    public var transform: (Inputs) -> Outputs
    
    /// Creates a layer that performs the given transformation on its inputs
    /// - Parameter transform: Transformation to perform
    public init(_ transform: @escaping (Inputs) -> Outputs) {
        self.transform = transform
    }
    
    public func callAsFunction(_ inputs: Inputs) -> Outputs {
        transform(inputs)
    }
}

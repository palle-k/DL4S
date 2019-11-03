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
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Element-wise hyperbolic tangent activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.tanh()
    }
}

/// Element-wise sigmoid activation layer.
public struct Sigmoid<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
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
    public var parameters: [Tensor<Element, Device>] { get {[]} }

    /// Element-wise rectified linear unit activation layer.
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.rectifiedLinear()
    }
}

/// Element-wise leaky linear rectified unit activation layer.
public struct LeakyRelu<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
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

/// Softmax activation layer
public struct Softmax<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Softmax activation layer
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Softmax") {
            inputs.softmax()
        }
    }
}

/// Layer wrapping an arbitrary transform provided by a closure.
public struct Lambda<Inputs, Outputs, Element: NumericType, Device: DeviceType>: LayerType {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
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

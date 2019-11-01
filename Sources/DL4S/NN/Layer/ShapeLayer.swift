//
//  ShapeLayer.swift
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

/// Layer that reshapes its inputs to a given target size, except the batch size
public struct Reshape<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Target size (except batch size)
    public var outputShape: [Int]
    
    /// Layer that reshapes its inputs to a given target size, except the batch size
    /// - Parameter outputShape: Target size (except batch size)
    public init(outputShape: [Int]) {
        self.outputShape = outputShape
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        // retain batch dimension
        inputs.view(as: [inputs.shape[0]] + outputShape)
    }
}

/// Layer that flattens its inputs into a tensor of shape [batchSize, -1]
public struct Flatten<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }

    /// Layer that flattens its inputs into a tensor of shape [batchSize, -1]
    public init() {}
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        // retain batch dimension
        inputs.view(as: [inputs.shape[0], -1])
    }
}

/// Layer that concatenates a list of input tensors along their second dimension
public struct Concat<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }

    /// Layer that concatenates a list of input tensors along their second dimension
    public init() {}
    
    public func callAsFunction(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        // 0th axis is batch dimension
        Tensor(stacking: inputs, along: 1)
    }
}

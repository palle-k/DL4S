//
//  Pooling.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
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


/// A 2D max pooling layer
public struct MaxPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Pooling window size
    public let windowSize: Int
    
    /// Pooling window stride
    public let stride: Int
    
    /// Padding applied around the edges of the input of the layer.
    public let padding: Int?
    
    /// Creates a 2D max pooling layer.
    /// - Parameters:
    ///   - windowSize: Size of the window
    ///   - stride: Stride, with which the window moves over the input tensor >= 1.
    ///   - padding: Padding applied around the edges of the input of the layer.
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.maxPooled2d(windowSize: windowSize, padding: padding, stride: stride)
    }
}

/// A 2D average pooling layer
public struct AvgPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Pooling window size
    public let windowSize: Int
    
    /// Pooling window stride
    public let stride: Int
    
    /// Padding applied around the edges of the input of the layer.
    public let padding: Int?
    
    /// Creates a 2D average pooling layer.
    /// - Parameters:
    ///   - windowSize: Size of the window
    ///   - stride: Stride, with which the window moves over the input tensor >= 1.
    ///   - padding: Padding applied around the edges of the input of the layer.
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.averagePooled2d(windowSize: windowSize, padding: padding, stride: stride)
    }
}

/// A 2D adaptive max pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
public struct AdaptiveMaxPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    /// Width and height of the output tensor
    public let targetSize: Int
    
    /// A 2D adaptive max pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
    /// - Parameter targetSize: Width and height of the output tensor
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let height = inputs.shape[2]
        let windowSize = height / targetSize
        return inputs.maxPooled2d(windowSize: windowSize, padding: 0, stride: windowSize)
    }
}

/// A 2D adaptive average pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
public struct AdaptiveAvgPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }

    /// Width and height of the output tensor
    public let targetSize: Int
    
    /// A 2D adaptive average pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
    /// - Parameter targetSize: Width and height of the output tensor
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let height = inputs.shape[2]
        let windowSize = height / targetSize
        return inputs.averagePooled2d(windowSize: windowSize, padding: 0, stride: windowSize)
    }
}

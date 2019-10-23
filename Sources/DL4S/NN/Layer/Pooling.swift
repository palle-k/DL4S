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


public struct MaxPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    public let windowSize: Int
    public let stride: Int
    public let padding: Int?
    
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MaxPool2D") {
            inputs.maxPooled2d(windowSize: windowSize, padding: padding, stride: stride)
        }
    }
}

public struct AvgPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    public let windowSize: Int
    public let stride: Int
    public let padding: Int?
    
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "AvgPool2D") {
            inputs.averagePooled2d(windowSize: windowSize, padding: padding, stride: stride)
        }
    }
}

public struct AdaptiveMaxPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    public let targetSize: Int
    
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let height = inputs.shape[2]
        let windowSize = height / targetSize
        return inputs.maxPooled2d(windowSize: windowSize, padding: 0, stride: windowSize)
    }
}

public struct AdaptiveAvgPool2D<Element: NumericType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[]}
    public var parameters: [Tensor<Element, Device>] { get {[]} }
    
    public let targetSize: Int
    
    public init(targetSize: Int) {
        self.targetSize = targetSize
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let height = inputs.shape[2]
        let windowSize = height / targetSize
        return inputs.averagePooled2d(windowSize: windowSize, padding: 0, stride: windowSize)
    }
}

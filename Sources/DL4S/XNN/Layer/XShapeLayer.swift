//
//  XShapeLayer.swift
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

public struct XReshape<Element: NumericType, Device: DeviceType>: XLayer, Codable {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} }
    
    public var outputShape: [Int]
    
    public init(outputShape: [Int]) {
        self.outputShape = outputShape
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        // retain batch dimension
        inputs.view(as: [inputs.shape[0]] + outputShape)
    }
}

public struct XFlatten<Element: NumericType, Device: DeviceType>: XLayer, Codable {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} }
    
    public init() {}
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        // retain batch dimension
        inputs.view(as: [inputs.shape[0], -1])
    }
}

public struct XConcat<Element: NumericType, Device: DeviceType>: XLayer, Codable {
    public var parameterPaths: [WritableKeyPath<Self, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} }
    
    public init() {}
    
    public func callAsFunction(_ inputs: [XTensor<Element, Device>]) -> XTensor<Element, Device> {
        // 0th axis is batch dimension
        XTensor(stacking: inputs, along: 1)
    }
}

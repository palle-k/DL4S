//
//  XDense.swift
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

public struct XDense<Element: RandomizableType, Device: DeviceType>: XLayer, Codable {
    public static var parameters: [WritableKeyPath<XDense<Element, Device>, XTensor<Element, Device>>] {[
        \.weights,
        \.bias
    ]}
    
    public var weights: XTensor<Element, Device>
    public var bias: XTensor<Element, Device>
    
    public var parameters: [XTensor<Element, Device>] {
        get {
            [weights, bias]
        }
        set {
            (weights, bias) = (newValue[0], newValue[1])
        }
    }
    
    public init(inputSize: Int, outputSize: Int) {
        weights = XTensor(xavierNormalWithShape: [inputSize, outputSize], requiresGradient: true)
        bias = XTensor(repeating: 0, shape: [outputSize], requiresGradient: true)
        
        #if DEBUG
        weights.tag = "W"
        bias.tag = "b"
        #endif
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        OperationGroup.capture(named: "Dense") {
            inputs.matMul(weights) + bias
        }
    }
}

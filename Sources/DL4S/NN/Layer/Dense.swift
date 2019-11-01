//
//  Dense.swift
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

/// Dense (Linear, Fully connected) layer with no activation function.
public struct Dense<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \.weights,
        \.bias
    ]}
    
    /// Weights, shape [inputSize, outputSize]
    public var weights: Tensor<Element, Device>
    
    /// Bias, shape [outputSize]
    public var bias: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        get {
            [weights, bias]
        }
    }
    
    /// Creates a dense / linear / fully connected layer with no output activation function.
    ///
    /// The layer expects inputs to have a shape of [batchSize, inputSize].
    ///
    /// - Parameters:
    ///   - inputSize: Number of elements in an input vector
    ///   - outputSize: Number of elements in an output vector
    public init(inputSize: Int, outputSize: Int) {
        weights = Tensor(xavierNormalWithShape: [inputSize, outputSize], requiresGradient: true)
        bias = Tensor(repeating: 0, shape: [outputSize], requiresGradient: true)
        
        #if DEBUG
        weights.tag = "W"
        bias.tag = "b"
        #endif
    }
    
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Dense") {
            inputs.matrixMultiplied(with: weights) + bias
        }
    }
}

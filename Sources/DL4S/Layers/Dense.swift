//
//  Dense.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.03.19.
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


/// Dense / Linear / Fully connected layer
public class Dense<Element: RandomizableType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    /// Weight matrix
    let w: Tensor<Element, Device>
    
    /// Bias vector
    let b: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return [w, b]
    }
    
    public var isTrainable: Bool = true
    
    /// Number of features in each input
    public var inputFeatures: Int {
        return w.shape[0]
    }
    
    /// Number of features in each output of the layer
    public var outputFeatures: Int {
        return w.shape[1]
    }
    
    
    /// Initializes a dense layer with the given number of input and output features and initializes the weights using Xavier 2/n initialization.
    ///
    /// - Parameters:
    ///   - inputFeatures: Number of input features
    ///   - outputFeatures: Number of output features
    public init(inputFeatures: Int, outputFeatures: Int) {
        w = Tensor(repeating: 0.5, shape: [inputFeatures, outputFeatures], requiresGradient: true)
        b = Tensor(repeating: 0, shape: [outputFeatures], requiresGradient: true)
        
        Random.fillNormal(w, mean: 0, stdev: (2 / Element(inputFeatures)).sqrt())
        
        w.tag = "W"
        b.tag = "b"
    }
    
    
    /// Performs a feed forward operation on a batch of samples with the shape [batchSize x inputSize] and returns
    /// a batch of samples with the shape [batchSize x outputSize].
    ///
    /// - Parameter inputs: Batch of input samples
    /// - Returns: Batch of forwarded samples.
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(inputs.count == 1)
        let out = mmul(inputs[0], w) + b
        return out
    }
}

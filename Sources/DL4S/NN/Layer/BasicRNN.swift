//
//  BasicRNN.swift
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

/// A 'vanilla' RNN.
/// In each step, the RNN performs the transformation matMul(x\_t, W) + matMul(h\_t-1, U) + b
public struct BasicRNN<Element: RandomizableType, Device: DeviceType>: RNN, Codable {
    public typealias Inputs = Tensor<Element, Device>
    public typealias Outputs = (Tensor<Element, Device>, () -> Tensor<Element, Device>)
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {[
        \.W, \.U, \.b
    ]}
    
    public let direction: RNNDirection
    
    public var W: Tensor<Element, Device>
    public var U: Tensor<Element, Device>
    public var b: Tensor<Element, Device>
    
    public var inputSize: Int {
        return W.shape[0]
    }
    public var hiddenSize: Int {
        return W.shape[1]
    }
    
    public var parameters: [Tensor<Element, Device>] {
        get {[W, U, b]}
    }

    /// A 'vanilla' RNN.
    /// In each step, the RNN performs the transformation matMul(x\_t, W) + matMul(h\_t-1, U) + b.
    ///
    /// - Parameters:
    ///  - inputSize: Number of elements in each input vector of the RNN. The RNN expects inputs to have a shape of [sequence length, batch size, input size].
    ///  - hiddenSize: Number of elements in each output vector of the RNN.
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward) {
        self.direction = direction
        
        W = Tensor(normalDistributedWithShape: [inputSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(inputSize)).sqrt(), requiresGradient: true)
        U = Tensor(normalDistributedWithShape: [hiddenSize, hiddenSize], mean: 0, stdev: (Element(1) / Element(hiddenSize)).sqrt(), requiresGradient: true)
        b = Tensor(repeating: 0, shape: [hiddenSize], requiresGradient: true)
    }
    
    public func initialState(for inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        Tensor(repeating: 0, shape: [inputs.shape[1], hiddenSize]) // [batchSize, hiddenSize]
    }
    
    public func prepare(inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "BasicRNNPrepare") {
            let seqlen = inputs.shape[0]
            let batchSize = inputs.shape[1]
            
            let multiplied = inputs
                .view(as: [seqlen * batchSize, inputSize])
                .matrixMultiplied(with: W)
                .view(as: [seqlen, batchSize, hiddenSize])
            
            return multiplied + b
        }
    }
    
    public func input(at step: Int, using preparedInput: Tensor<Element, Device>) -> Tensor<Element, Device> {
        preparedInput[step]
    }
    
    public func step(_ preparedInput: Tensor<Element, Device>, previousState: Tensor<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "BasicRNNCell") {
            tanh(preparedInput + previousState.matrixMultiplied(with: U))
        }
    }
    
    public func concatenate(_ states: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        Tensor(stacking: states.map {$0.unsqueezed(at: 0)}, along: 0)
    }
    
    public func numberOfSteps(for inputs: Tensor<Element, Device>) -> Int {
        inputs.shape[0]
    }
}

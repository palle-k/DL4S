//
//  BasicRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.05.19.
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

public class BasicRNN<Element: RandomizableType, Device: DeviceType>: RNN {
    public typealias State = Tensor<Element, Device>
    public typealias Input = Element
    public typealias Element = Element
    
    public var isTrainable: Bool = true
    public let shouldReturnFullSequence: Bool
    
    public let direction: RNNDirection
    
    public var W: Tensor<Element, Device>
    public var U: Tensor<Element, Device>
    public var b: Tensor<Element, Device>
    
    public let inputSize: Int
    public let hiddenSize: Int
    
    public var parameters: [Tensor<Element, Device>] {
        return [W, U, b]
    }
    
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward, shouldReturnFullSequence: Bool = false) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.direction = direction
        self.shouldReturnFullSequence = shouldReturnFullSequence
        
        W = Tensor(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        U = Tensor(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        b = Tensor(repeating: 0, shape: hiddenSize, requiresGradient: true)
        
        Random.fillNormal(W, stdev: (Element(1) / Element(inputSize)).sqrt())
        Random.fillNormal(U, stdev: (Element(1) / Element(hiddenSize)).sqrt())
    }
    
    public func createInitialState(fromLayerInputs inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        precondition(1 ... 2 ~= inputs.count)
        
        let x = inputs[0]
        
        let batchSize = x.shape[1]
        
        if inputs.count == 1 {
            return Tensor(repeating: 0, shape: batchSize, hiddenSize)
        } else {
            return inputs[1]
        }
    }
    
    public func prepare(input: Tensor<Element, Device>) -> [Tensor<Element, Device>] {
        let seqlen = input.shape[0]
        let batchSize = input.shape[1]
        
        return [
            mmul(input.view(as: seqlen * batchSize, inputSize), W).view(as: seqlen, batchSize, hiddenSize) + b
        ]
    }
    
    public func step(preparedInputs: [Tensor<Element, Device>], state: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let x = preparedInputs[0]
        
        return tanh(x + mmul(state, U))
    }
    
    public func output(for state: Tensor<Element, Device>) -> Tensor<Element, Device> {
        return state
    }
    
}

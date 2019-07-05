//
//  GRU.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.04.19.
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

/// Gated recurrent unit layer (mono-directional) for sequence to sequence transformation with arbitrary length.
public class GRU<Element: RandomizableType, Device: DeviceType>: RNN, Codable {
    public typealias Input = Element
    
    // GRU weights
    
    let W_z: Tensor<Element, Device>
    let W_r: Tensor<Element, Device>
    let W_h: Tensor<Element, Device>
    
    let U_z: Tensor<Element, Device>
    let U_r: Tensor<Element, Device>
    let U_h: Tensor<Element, Device>
    
    let b_z: Tensor<Element, Device>
    let b_r: Tensor<Element, Device>
    let b_h: Tensor<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {
        return [W_z, W_r, W_h, U_z, U_r, U_h, b_z, b_r, b_h]
    }
    
    public var isTrainable: Bool = true
    
    
    /// Number of elements in each hidden state
    public let hiddenSize: Int
    
    /// Size of each input in the input sequence
    public let inputSize: Int
    
    /// Indicates whether the GRU should return its full state sequence or only the last hidden state
    public let shouldReturnFullSequence: Bool
    
    
    public var direction: RNNDirection
    
    
    /// Initializes a gated recurrent unit layer with the given input and hidden size at each timestep.
    ///
    /// If the GRU is instructed to return full sequences, the GRU hidden state sequence
    /// is returned by the forward operation.
    /// If the GRU is not instructed to return full sequences, only the last hidden state is returned.
    /// The latter may be computationally less intensive and should be preferred if possible.
    ///
    /// - Parameters:
    ///   - inputSize: Number of inputs at each timestep
    ///   - hiddenSize: Number of elements in each hidden
    ///   - shouldReturnFullSequence: Indicates whether the GRU should return its full state sequence or only the last hidden state
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward, shouldReturnFullSequence: Bool = false) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.shouldReturnFullSequence = shouldReturnFullSequence
        self.direction = direction
        
        W_z = Tensor(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        W_r = Tensor(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        W_h = Tensor(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        
        U_z = Tensor(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        U_r = Tensor(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        U_h = Tensor(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        
        b_z = Tensor(repeating: 0, shape: hiddenSize, requiresGradient: true)
        b_r = Tensor(repeating: 0, shape: hiddenSize, requiresGradient: true)
        b_h = Tensor(repeating: 0, shape: hiddenSize, requiresGradient: true)
        
        W_z.tag = "W_z"
        W_r.tag = "W_r"
        W_h.tag = "W_h"
        U_z.tag = "U_z"
        U_r.tag = "U_r"
        U_h.tag = "U_h"
        b_z.tag = "b_z"
        b_r.tag = "b_r"
        b_h.tag = "b_h"
        
        for W in [W_z, W_r, W_h] {
            Random.fillNormal(W, stdev: (Element(1) / Element(inputSize)).sqrt())
        }
        
        for U in [U_z, U_r, U_h] {
            Random.fillNormal(U, stdev: (Element(1) / Element(hiddenSize)).sqrt())
        }
    }
    
    
    public func step(preparedInputs: [Tensor<Element, Device>], state: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let x_z = preparedInputs[0]
        let x_r = preparedInputs[1]
        let x_h = preparedInputs[2]
        
        let h_p = state.view(as: x_z.shape[0], hiddenSize)
        
        let z_t = sigmoid(x_z + mmul(h_p, U_z))
        let r_t = sigmoid(x_r + mmul(h_p, U_r))
        
        let h_t_partial_1 = (1 - z_t) * h_p
        let h_t_partial_2 = tanh(x_h + mmul(r_t * h_p, U_h))
        
        let h_t = h_t_partial_1 + z_t * h_t_partial_2
        
        return h_t
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
            input.view(as: seqlen * batchSize, inputSize).mmul(W_z).view(as: seqlen, batchSize, hiddenSize) + b_z,
            input.view(as: seqlen * batchSize, inputSize).mmul(W_r).view(as: seqlen, batchSize, hiddenSize) + b_r,
            input.view(as: seqlen * batchSize, inputSize).mmul(W_h).view(as: seqlen, batchSize, hiddenSize) + b_h
        ]
    }
    
    
    public func output(for state: Tensor<Element, Device>) -> Tensor<Element, Device> {
        return state
    }
}

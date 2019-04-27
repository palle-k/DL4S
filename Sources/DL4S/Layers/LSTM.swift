//
//  LSTM.swift
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


/// Long short-term memory layer (mono-directional) for sequence to sequence transformation with arbitrary length.
public class LSTM<Element: RandomizableType, Device: DeviceType>: RNN, Codable {
    //    public typealias Element = Element
    //    public typealias Device = Device
    public typealias Input = Element
    //    public typealias State = (h: Tensor<Element, Device>, c: Tensor<Element, Device>)
    
    public var isTrainable: Bool = true
    
    // LSTM weights
    
    let W_i: Tensor<Element, Device>
    let W_o: Tensor<Element, Device>
    let W_f: Tensor<Element, Device>
    let W_c: Tensor<Element, Device>
    
    let U_i: Tensor<Element, Device>
    let U_o: Tensor<Element, Device>
    let U_f: Tensor<Element, Device>
    let U_c: Tensor<Element, Device>
    
    let b_i: Tensor<Element, Device>
    let b_o: Tensor<Element, Device>
    let b_f: Tensor<Element, Device>
    let b_c: Tensor<Element, Device>
    
    /// Number of elements in each hidden state
    public let hiddenSize: Int
    
    /// Size of each input in the input sequence
    public let inputSize: Int
    
    
    public let direction: RNNDirection
    
    
    /// Indicates whether the LSTM should return its full state sequence or only the last hidden state
    public let shouldReturnFullSequence: Bool
    
    public var parameters: [Tensor<Element, Device>] {
        return [
            W_i, U_i, b_i,
            W_o, U_o, b_o,
            W_f, U_f, b_f,
            W_c, U_c, b_c
        ]
    }
    
    
    /// Initializes an LSTM layer with the given input and hidden size at each timestep.
    ///
    /// If the LSTM is instructed to return full sequences, the LSTM hidden state sequence and cell state sequence
    /// is returned by the forward operation.
    /// If the LSTM is not instructed to return full sequences, only the last hidden state is returned.
    /// The latter may be computationally less intensive and should be preferred if possible.
    ///
    /// - Parameters:
    ///   - inputSize: Number of inputs at each timestep
    ///   - hiddenSize: Number of elements in each hidden and cell state
    ///   - shouldReturnFullSequence: Indicates whether the LSTM should return its full state sequence or only the last hidden state
    public init(inputSize: Int, hiddenSize: Int, direction: RNNDirection = .forward, shouldReturnFullSequence: Bool = false) {
        W_i = Tensor<Element, Device>(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        W_o = Tensor<Element, Device>(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        W_f = Tensor<Element, Device>(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        W_c = Tensor<Element, Device>(repeating: 0, shape: inputSize, hiddenSize, requiresGradient: true)
        
        U_i = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        U_o = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        U_f = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        U_c = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, hiddenSize, requiresGradient: true)
        
        b_i = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, requiresGradient: true)
        b_o = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, requiresGradient: true)
        b_f = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, requiresGradient: true)
        b_c = Tensor<Element, Device>(repeating: 0, shape: hiddenSize, requiresGradient: true)
        
        W_i.tag = "W_i"
        W_o.tag = "W_o"
        W_f.tag = "W_f"
        W_c.tag = "W_c"
        U_i.tag = "U_i"
        U_o.tag = "U_o"
        U_f.tag = "U_f"
        U_c.tag = "U_c"
        b_i.tag = "b_i"
        b_o.tag = "b_o"
        b_f.tag = "b_f"
        b_c.tag = "b_c"
        
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.direction = direction
        self.shouldReturnFullSequence = shouldReturnFullSequence
        
        for W in [W_i, W_o, W_f, W_c] {
            Random.fillNormal(W, stdev: (Element(1) / Element(inputSize)).sqrt())
        }
        
        for U in [U_i, U_o, U_f, U_c] {
            Random.fillNormal(U, stdev: (Element(1) / Element(hiddenSize)).sqrt())
        }
    }
    
    
    public func step(x: Tensor<Element, Device>, state: (h: Tensor<Element, Device>, c: Tensor<Element, Device>)) -> (h: Tensor<Element, Device>, c: Tensor<Element, Device>) {
        let x_t = x
        let h_p = state.h
        let c_p = state.c
        
        // TODO: Unify W_* matrics, U_* matrices and b_* vectors, perform just two matrix multiplications and one addition, then select slices
        let f_t = sigmoid(mmul(x_t, W_f) + mmul(h_p, U_f) + b_f)
        let i_t = sigmoid(mmul(x_t, W_i) + mmul(h_p, U_i) + b_i)
        let o_t = sigmoid(mmul(x_t, W_o) + mmul(h_p, U_o) + b_o)
        
        let c_t_partial_1 = f_t * c_p + i_t
        let c_t_partial_2 = tanh(mmul(x_t, W_c) + mmul(h_p, U_c) + b_c)
        let c_t = c_t_partial_1 * c_t_partial_2
        let h_t = o_t * tanh(c_t)
        
        return (h_t, c_t)
    }
    
    public func output(for state: (h: Tensor<Element, Device>, c: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        return state.h
    }
    
    
    public func createInitialState(fromLayerInputs inputs: [Tensor<Element, Device>]) -> (h: Tensor<Element, Device>, c: Tensor<Element, Device>) {
        let x = inputs[0]
        let batchSize = x.shape[1]
        
        let h0: Tensor<Element, Device>
        let c0: Tensor<Element, Device>
        
        if inputs.count == 1 {
            h0 = Tensor(repeating: 0, shape: batchSize, hiddenSize)
            c0 = Tensor(repeating: 0, shape: batchSize, hiddenSize)
        } else {
            h0 = inputs[1][0]
            c0 = inputs[1][1]
        }
        
        return (h0, c0)
    }
    
}

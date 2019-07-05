//
//  RecurrentLayerTypes.swift
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


public enum RNNDirection: Int, Codable, Hashable {
    case forward = 0
    case backward = 1
}


public protocol RNN: Layer {
    associatedtype State
    
    var shouldReturnFullSequence: Bool { get }
    var direction: RNNDirection { get }
    
    func createInitialState(fromLayerInputs inputs: [Tensor<Element, Device>]) -> State
    func step(preparedInputs: [Tensor<Element, Device>], state: State) -> State
    func output(for state: State) -> Tensor<Element, Device>
    func prepare(input: Tensor<Element, Device>) -> [Tensor<Element, Device>]
}


public extension RNN {
    func process(_ inputs: [Tensor<Element, Device>]) -> ([Tensor<Element, Device>], State) {
        precondition([1, 3].contains(inputs.count))
        let x = prepare(input: inputs[0])
        let seqlen = inputs[0].shape[0]
        
        var state = createInitialState(fromLayerInputs: inputs)
        
        var outputSequence: [Tensor<Element, Device>] = []
        
        for i in 0 ..< seqlen {
            let x_t = x.map {$0[direction == .forward ? i : (seqlen - i - 1)]}
            
            state = step(preparedInputs: x_t, state: state)
            
            outputSequence.append(output(for: state).unsqueeze(at: 0))
        }
        
        if direction == .backward {
            outputSequence.reverse()
        }
        
        return (outputSequence, state)
    }
    
    func prepare(input: Tensor<Element, Device>) -> [Tensor<Element, Device>] {
        return [input]
    }
    
//    func process(_ inputs: PackedSequence<Element, Device>) -> PackedSequence<Element, Device> {
//        guard let maxLength = inputs.lengths.max() else {
//            // As inputs is empty at this point, it can be returned as the result instead of allocating a new empty packed sequence.
//            return inputs
//        }
//        let x = self.prepare(input: inputs.storage).map {
//            PackedSequence(from: $0, withLengths: inputs.lengths)
//        }
//        let batchSizePlaceholder = Tensor<Element, Device>(repeating: 0, shape: 0, inputs.batchSize)
//        var state = createInitialState(fromLayerInputs: [batchSizePlaceholder])
//        
//        var outputSequence: [Tensor<Element, Device>] = []
//        
//        for i in 0 ..< maxLength {
//            let x_t = x.map {$0[direction == .forward ? i : (maxLength - i - 1)]}
//            
//            
//            
//            state = step(preparedInputs: x_t, state: state)
//            outputSequence.append(output(for: state))
//        }
//        
//        if direction == .backward {
//            outputSequence.reverse()
//        }
//        
//        fatalError("TODO")
//    }
    
    
    /// Forwards the given input sequence through the LSTM.
    ///
    /// Expects either one or more inputs depending on whether an initial state is given or not.
    /// The first input must be the input sequence.
    /// Additional parameters are RNN states and depend on the RNN implementation
    ///
    /// The input sequence must be in the shape [SequenceLength x BatchSize x InputSize].
    ///
    /// If the RNN should return full sequences, the output has the shape [SequenceLength x BatchSize x HiddenSize].
    ///
    /// If the RNN should not return full sequences, the output has the shape [BatchSize x HiddenSize] and only contains the last hidden state.
    ///
    /// - Parameter inputs: Input sequence and optional initial RNN state.
    /// - Returns: If the RNN should return full sequences, all RNN states, otherwise the last RNN state.
    func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        // Expects one or three inputs
        // Either:
        // - input sequence, [initial hidden state and cell state] vector
        // - input sequence
        
        // Produces one hidden state vector for every input vector
        
        // Input shape: SequencLength x BatchSize x NumFeatures
        // Output shape: SequenceLength x BatchSize x HiddenSize
        
        let (outputSequence, state) = process(inputs)
        
        if shouldReturnFullSequence {
            return stack(outputSequence)
        } else {
            return output(for: state)
        }
    }
    
    func step(x: Tensor<Element, Device>, state: State) -> State {
        return step(preparedInputs: prepare(input: x.unsqueeze(at: 0)).map {$0.squeeze(at: 0)}, state: state)
    }
}


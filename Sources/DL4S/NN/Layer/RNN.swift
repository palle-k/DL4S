//
//  RNN.swift
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


public enum RNNDirection: String, Codable {
    case forward
    case backward
}


/// Base protocol for recurrent neural networks.
public protocol RNN: LayerType where Outputs == (State, () -> StateSequence) {
    associatedtype State
    associatedtype StateSequence
    associatedtype PreparedInput
    associatedtype StepInput
    
    var direction: RNNDirection { get }
    
    /// Number of steps to perform given the inputs of the RNN
    /// - Parameter inputs: Inputs of the RNN
    func numberOfSteps(for inputs: Inputs) -> Int
    
    /// Creates the initial state of the RNN for processing the given sequence
    /// - Parameter inputs: Sequence to process
    func initialState(for inputs: Inputs) -> State
    
    /// Performs the input transformation on all timesteps of the input at once
    /// - Parameter inputs: Sequence to process
    func prepare(inputs: Inputs) -> PreparedInput
    
    /// Concatenates the given array of states into a state sequence
    /// - Parameter states: States to concatenate
    func concatenate(_ states: [State]) -> StateSequence
    
    /// Extracts the inputs of the RNN at a given timestep from the preprocessed input sequence
    /// - Parameters:
    ///   - step: Timestep
    ///   - preparedInput: Prepared input sequence
    func input(at step: Int, using preparedInput: PreparedInput) -> StepInput
    
    /// Performs a single RNN timestep
    /// - Parameters:
    ///   - preparedInput: Preprocessed input for the current timestep
    ///   - previousState: Previous hidden state
    func step(_ preparedInput: StepInput, previousState: State) -> State
    
    /// Applies the RNN to the given input sequence using the provided initial state
    /// - Parameters:
    ///   - inputs: Input sequence
    ///   - state: Initial state, optional
    func callAsFunction(_ inputs: Inputs, state: State?) -> (State, () -> StateSequence)
}

extension RNN {
    public func callAsFunction(_ inputs: Inputs) -> Outputs {
        callAsFunction(inputs, state: initialState(for: inputs))
    }
    
    public func callAsFunction(_ inputs: Inputs, state: State? = nil) -> Outputs {
        OperationGroup.capture(named: "RNN") {
            let initState = state ?? initialState(for: inputs)
            let prepared = prepare(inputs: inputs)
            
            var currentState = initState
            var stateSequence: [State] = []
            
            let range: AnySequence<Int>
            switch direction {
            case .forward:
                range = AnySequence(0 ..< numberOfSteps(for: inputs))
            case .backward:
                range = AnySequence((0 ..< numberOfSteps(for: inputs)).reversed())
            }
            
            for i in range {
                let stepInput = input(at: i, using: prepared)
                currentState = step(stepInput, previousState: currentState)
                stateSequence.append(currentState)
            }
            
            if direction == .backward {
                stateSequence.reverse()
            }
            
            return (currentState, {self.concatenate(stateSequence)})
        }
    }
}

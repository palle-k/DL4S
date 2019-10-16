//
//  XRNN.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
//

import Foundation


public protocol XRNN: XLayer where Outputs == (State, () -> StateSequence) {
    associatedtype State
    associatedtype StateSequence
    associatedtype PreparedInput
    associatedtype StepInput
    
    var direction: RNNDirection { get }
    
    func numberOfSteps(for inputs: Inputs) -> Int
    func initialState(for inputs: Inputs) -> State
    func prepare(inputs: Inputs) -> PreparedInput
    func concatenate(_ states: [State]) -> StateSequence
    func input(at step: Int, using preparedInput: PreparedInput) -> StepInput
    func step(_ preparedInput: StepInput, previousState: State) -> State
    func callAsFunction(_ inputs: Inputs, state: State?) -> (State, () -> StateSequence)
}

extension XRNN {
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

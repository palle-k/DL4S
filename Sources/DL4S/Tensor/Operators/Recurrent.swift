//
//  Recurrent.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

// MARK: Recurrent cells

/// Computes one step of a gated recurrent unit.
///
/// With `z = sigmoid(updateInput + state × updateWeights)`, `r = sigmoid(resetInput + state × resetWeights)`, and
/// `c = tanh(candidateInput + (r * state) × candidateWeights)`, the new state is `(1 - z) * state + z * c`.
///
/// - Parameters:
///   - updateInput: Projection of the input for the update gate, including its bias, shape [batchSize, hiddenSize]
///   - resetInput: Projection of the input for the reset gate, including its bias, shape [batchSize, hiddenSize]
///   - candidateInput: Projection of the input for the candidate state, including its bias, shape [batchSize, hiddenSize]
///   - state: Previous state, shape [batchSize, hiddenSize]
///   - updateWeights: Weights of the state for the update gate, shape [hiddenSize, hiddenSize]
///   - resetWeights: Weights of the state for the reset gate, shape [hiddenSize, hiddenSize]
///   - candidateWeights: Weights of the reset state for the candidate state, shape [hiddenSize, hiddenSize]
/// - Returns: New state, shape [batchSize, hiddenSize]
public func gatedRecurrentUnitStep<Element, Device>(
    updateInput: Tensor<Element, Device>,
    resetInput: Tensor<Element, Device>,
    candidateInput: Tensor<Element, Device>,
    state: Tensor<Element, Device>,
    updateWeights: Tensor<Element, Device>,
    resetWeights: Tensor<Element, Device>,
    candidateWeights: Tensor<Element, Device>,
) -> Tensor<Element, Device> {
    precondition(state.dim == 2 && [updateInput, resetInput, candidateInput].allSatisfy { $0.shape == state.shape }, "The inputs must have the shape of the state, [batchSize, hiddenSize].")
    precondition([updateWeights, resetWeights, candidateWeights].allSatisfy { $0.shape == [state.shape[1], state.shape[1]] }, "The weights must have the shape [hiddenSize, hiddenSize].")
    let result = Device.FusedOperations.gatedRecurrentUnitStep(
        updateInput: updateInput,
        resetInput: resetInput,
        candidateInput: candidateInput,
        state: state,
        updateWeights: updateWeights,
        resetWeights: resetWeights,
        candidateWeights: candidateWeights,
    )
    let sources = [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights]
    return result.attachingContext(tag: "gruStep", sources: sources) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            let computed = Composed.gatedRecurrentUnitGradients(
                updateInput: updateInput,
                resetInput: resetInput,
                candidateInput: candidateInput,
                state: state,
                updateWeights: updateWeights,
                resetWeights: resetWeights,
                candidateWeights: candidateWeights,
                outputGradient: resultGradient,
                computes: sources.map(\.requiresGradient),
            )
            for (index, gradient) in computed.inSourceOrder.enumerated() {
                Tensor.accumulate(gradient, into: &gradients[index])
            }
        } else {
            // The array gives up its references, so the accumulated gradients stay uniquely referenced.
            let taken = gradients
            gradients = Array(repeating: nil, count: taken.count)
            var accumulated = GatedRecurrentUnitGradients(inSourceOrder: consume taken)
            Device.FusedOperations.gatedRecurrentUnitStepBackward(
                updateInput: updateInput,
                resetInput: resetInput,
                candidateInput: candidateInput,
                state: state,
                updateWeights: updateWeights,
                resetWeights: resetWeights,
                candidateWeights: candidateWeights,
                outputGradient: resultGradient,
                accumulating: &accumulated,
            )
            gradients = accumulated.inSourceOrder
        }
    }
}

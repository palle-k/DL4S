//
//  FusedRecurrent.swift
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

// MARK: Default implementations

public extension FusedOperationsType {
    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
    ) -> Tensor<N, Device> {
        let state = state.detached()
        let update = (updateInput.detached() + state.matrixMultiplied(with: updateWeights.detached())).sigmoid()
        let reset = (resetInput.detached() + state.matrixMultiplied(with: resetWeights.detached())).sigmoid()
        let candidate = (candidateInput.detached() + (reset * state).matrixMultiplied(with: candidateWeights.detached())).tanh()
        return (1 - update) * state + update * candidate
    }

    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        accumulating gradients: inout GatedRecurrentUnitGradients<N, Device>,
    ) {
        gradients.accumulate(Composed.gatedRecurrentUnitGradients(
            updateInput: updateInput.detached(),
            resetInput: resetInput.detached(),
            candidateInput: candidateInput.detached(),
            state: state.detached(),
            updateWeights: updateWeights.detached(),
            resetWeights: resetWeights.detached(),
            candidateWeights: candidateWeights.detached(),
            outputGradient: outputGradient.detached(),
            computes: [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights].map(\.requiresGradient),
        ))
    }
}

// MARK: Composed gradients

extension Composed {
    /// Computes the gradients of one step of a gated recurrent unit.
    ///
    /// The flags in `computes` select the gradients in the order of ``GatedRecurrentUnitGradients/inSourceOrder``.
    static func gatedRecurrentUnitGradients<N, Device>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computes: [Bool],
    ) -> GatedRecurrentUnitGradients<N, Device> {
        // The gates are computed again instead of being kept alive between the forward and the backward pass.
        let update = (updateInput + state.matrixMultiplied(with: updateWeights)).sigmoid()
        let reset = (resetInput + state.matrixMultiplied(with: resetWeights)).sigmoid()
        let resetState = reset * state
        let candidate = (candidateInput + resetState.matrixMultiplied(with: candidateWeights)).tanh()

        // The new state is state + update * (candidate - state).
        let updateActivationGradient = outputGradient * (candidate - state) * update * (1 - update)
        let candidateActivationGradient = outputGradient * update * (1 - candidate * candidate)

        var gradients = GatedRecurrentUnitGradients<N, Device>()
        gradients.updateInput = computes[0] ? updateActivationGradient : nil
        gradients.candidateInput = computes[2] ? candidateActivationGradient : nil
        gradients.updateWeights = computes[4] ? state.matrixMultiplied(with: updateActivationGradient, transposeSelf: true) : nil
        gradients.candidateWeights = computes[6] ? resetState.matrixMultiplied(with: candidateActivationGradient, transposeSelf: true) : nil

        guard computes[1] || computes[3] || computes[5] else {
            return gradients
        }
        let resetStateGradient = candidateActivationGradient.matrixMultiplied(with: candidateWeights, transposeOther: true)
        let resetActivationGradient = resetStateGradient * state * reset * (1 - reset)
        gradients.resetInput = computes[1] ? resetActivationGradient : nil
        gradients.resetWeights = computes[5] ? state.matrixMultiplied(with: resetActivationGradient, transposeSelf: true) : nil
        if computes[3] {
            gradients.state = outputGradient * (1 - update)
                + resetStateGradient * reset
                + resetActivationGradient.matrixMultiplied(with: resetWeights, transposeOther: true)
                + updateActivationGradient.matrixMultiplied(with: updateWeights, transposeOther: true)
        }
        return gradients
    }
}

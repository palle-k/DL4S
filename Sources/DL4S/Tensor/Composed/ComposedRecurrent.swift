//
//  ComposedRecurrent.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.09.26.
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

// MARK: Composed gradients

extension Composed {
    static func gatedRecurrentUnitStepBackward<N, Device>(
        updateInput: Tensor<N, Device>,
        resetInput: Tensor<N, Device>,
        candidateInput: Tensor<N, Device>,
        state: Tensor<N, Device>,
        updateWeights: Tensor<N, Device>,
        resetWeights: Tensor<N, Device>,
        candidateWeights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        gradients: inout GatedRecurrentUnitGradients<GradientAccumulator<N, Device>>,
    ) {
        // The gates are computed again instead of being kept alive between the forward and the backward pass.
        let update = (updateInput + state.matrixMultiplied(with: updateWeights)).sigmoid()
        let reset = (resetInput + state.matrixMultiplied(with: resetWeights)).sigmoid()
        let resetState = reset * state
        let candidate = (candidateInput + resetState.matrixMultiplied(with: candidateWeights)).tanh()

        // The new state is state + update * (candidate - state).
        let updateActivationGradient = outputGradient * (candidate - state) * update * (1 - update)
        let candidateActivationGradient = outputGradient * update * (1 - candidate * candidate)

        // The weight gradients are products, which are added in place without a gradient graph,
        // so a weight that every time step uses needs no temporary gradient.
        if gradients.updateInput.isRequested {
            gradients.updateInput.add(updateActivationGradient)
        }
        if gradients.candidateInput.isRequested {
            gradients.candidateInput.add(candidateActivationGradient)
        }
        if gradients.updateWeights.isRequested {
            gradients.updateWeights.add(state.matrixMultiplied(with: updateActivationGradient, transposeSelf: true))
        }
        if gradients.candidateWeights.isRequested {
            gradients.candidateWeights.add(resetState.matrixMultiplied(with: candidateActivationGradient, transposeSelf: true))
        }
        guard gradients.resetInput.isRequested || gradients.state.isRequested || gradients.resetWeights.isRequested else {
            return
        }
        let resetStateGradient = candidateActivationGradient.matrixMultiplied(with: candidateWeights, transposeOther: true)
        let resetActivationGradient = resetStateGradient * state * reset * (1 - reset)
        if gradients.resetInput.isRequested {
            gradients.resetInput.add(resetActivationGradient)
        }
        if gradients.resetWeights.isRequested {
            gradients.resetWeights.add(state.matrixMultiplied(with: resetActivationGradient, transposeSelf: true))
        }
        if gradients.state.isRequested {
            gradients.state.add(
                outputGradient * (1 - update)
                    + resetStateGradient * reset
                    + resetActivationGradient.matrixMultiplied(with: resetWeights, transposeOther: true)
                    + updateActivationGradient.matrixMultiplied(with: updateWeights, transposeOther: true),
            )
        }
    }
}

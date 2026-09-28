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
        updateInput: ShapedBuffer<N, Device>,
        resetInput: ShapedBuffer<N, Device>,
        candidateInput: ShapedBuffer<N, Device>,
        state: ShapedBuffer<N, Device>,
        updateWeights: ShapedBuffer<N, Device>,
        resetWeights: ShapedBuffer<N, Device>,
        candidateWeights: ShapedBuffer<N, Device>,
        result: MutableShapedBuffer<N, Device>,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let gates = GatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, candidate: result, math: math)
        // The new state is state + update * (candidate - state), and the candidate is in the result.
        math.subtract(result, state, into: result)
        math.multiply(result, gates.update, into: result)
        math.add(result, state, into: result)
    }

    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: ShapedBuffer<N, Device>,
        resetInput: ShapedBuffer<N, Device>,
        candidateInput: ShapedBuffer<N, Device>,
        state: ShapedBuffer<N, Device>,
        updateWeights: ShapedBuffer<N, Device>,
        resetWeights: ShapedBuffer<N, Device>,
        candidateWeights: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        gradients: GatedRecurrentUnitGradients<GradientBuffer<N, Device>?>,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The gates are computed again instead of being kept alive between the forward and the backward pass.
        let candidate = math.temporary(state.shape)
        let gates = GatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, candidate: candidate, math: math)
        let (update, reset, resetState) = (gates.update, gates.reset, gates.resetState)
        let slope = math.temporary(state.shape)

        // The new state is state + update * (candidate - state).
        // The gradient of the update activation is outputGradient * (candidate - state) * update * (1 - update).
        let updateActivationGradient = math.temporary(state.shape)
        math.subtract(candidate, state, into: updateActivationGradient)
        math.multiply(updateActivationGradient, outputGradient, into: updateActivationGradient)
        math.subtract(1, update, into: slope)
        math.multiply(slope, update, into: slope)
        math.multiply(updateActivationGradient, slope, into: updateActivationGradient)
        // The gradient of the candidate activation is outputGradient * update * (1 - candidate * candidate).
        let candidateActivationGradient = candidate
        math.multiply(candidate, candidate, into: candidateActivationGradient)
        math.subtract(1, candidateActivationGradient, into: candidateActivationGradient)
        math.multiply(candidateActivationGradient, update, into: candidateActivationGradient)
        math.multiply(candidateActivationGradient, outputGradient, into: candidateActivationGradient)

        // The weight gradients are added with GEMMs, so a weight that every time step uses needs no temporary gradient.
        math.write(gradients.updateInput) { math.copy(updateActivationGradient, into: $0) }
        math.write(gradients.candidateInput) { math.copy(candidateActivationGradient, into: $0) }
        if let weightGradient = gradients.updateWeights {
            math.multiplyMatrices(state, updateActivationGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        if let weightGradient = gradients.candidateWeights {
            math.multiplyMatrices(resetState, candidateActivationGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        guard gradients.resetInput != nil || gradients.state != nil || gradients.resetWeights != nil else {
            return
        }
        let resetStateGradient = math.temporary(state.shape)
        math.multiplyMatrices(candidateActivationGradient, candidateWeights, rhsTransposed: true, into: resetStateGradient)
        // The gradient of the reset activation is resetStateGradient * state * reset * (1 - reset).
        let resetActivationGradient = math.temporary(state.shape)
        math.multiply(resetStateGradient, state, into: resetActivationGradient)
        math.subtract(1, reset, into: slope)
        math.multiply(slope, reset, into: slope)
        math.multiply(resetActivationGradient, slope, into: resetActivationGradient)
        math.write(gradients.resetInput) { math.copy(resetActivationGradient, into: $0) }
        if let weightGradient = gradients.resetWeights {
            math.multiplyMatrices(state, resetActivationGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        // outputGradient * (1 - update) + resetStateGradient * reset, and the products with the weights of the gates
        math.write(gradients.state) { stateGradient in
            math.subtract(1, update, into: stateGradient)
            math.multiply(stateGradient, outputGradient, into: stateGradient)
            math.multiply(resetStateGradient, reset, into: slope)
            math.add(stateGradient, slope, into: stateGradient)
            math.multiplyMatrices(resetActivationGradient, resetWeights, rhsTransposed: true, into: stateGradient, beta: 1)
            math.multiplyMatrices(updateActivationGradient, updateWeights, rhsTransposed: true, into: stateGradient, beta: 1)
        }
    }
}

/// The gates of one step of a gated recurrent unit, in intermediate buffers.
struct GatedRecurrentUnitGates<N: NumericType, Device: DeviceType> {
    /// Update gate, `sigmoid(updateInput + state × updateWeights)`
    let update: MutableShapedBuffer<N, Device>
    /// Reset gate, `sigmoid(resetInput + state × resetWeights)`
    let reset: MutableShapedBuffer<N, Device>
    /// Reset state, `reset * state`
    let resetState: MutableShapedBuffer<N, Device>

    /// Computes the gates, and the candidate state `tanh(candidateInput + resetState × candidateWeights)` into the given buffer.
    init(
        updateInput: ShapedBuffer<N, Device>,
        resetInput: ShapedBuffer<N, Device>,
        candidateInput: ShapedBuffer<N, Device>,
        state: ShapedBuffer<N, Device>,
        updateWeights: ShapedBuffer<N, Device>,
        resetWeights: ShapedBuffer<N, Device>,
        candidateWeights: ShapedBuffer<N, Device>,
        candidate: MutableShapedBuffer<N, Device>,
        math: BufferMath<N, Device>,
    ) {
        update = math.temporary(state.shape)
        reset = math.temporary(state.shape)
        resetState = math.temporary(state.shape)
        math.copy(updateInput, into: update)
        math.multiplyMatrices(state, updateWeights, into: update, beta: 1)
        math.sigmoid(update, into: update)
        math.copy(resetInput, into: reset)
        math.multiplyMatrices(state, resetWeights, into: reset, beta: 1)
        math.sigmoid(reset, into: reset)
        math.multiply(reset, state, into: resetState)
        math.copy(candidateInput, into: candidate)
        math.multiplyMatrices(resetState, candidateWeights, into: candidate, beta: 1)
        math.tanh(candidate, into: candidate)
    }
}

//
//  CPUFusedRecurrent.swift
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

// A step of a gated recurrent unit writes the input projections into the gate buffers and adds the products with the
// state weights with GEMMs (beta 1), so the sums need no separate pass. The activations and the new state are
// element-wise loops.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: ShapedBuffer<N, CPU>,
        resetInput: ShapedBuffer<N, CPU>,
        candidateInput: ShapedBuffer<N, CPU>,
        state: ShapedBuffer<N, CPU>,
        updateWeights: ShapedBuffer<N, CPU>,
        resetWeights: ShapedBuffer<N, CPU>,
        candidateWeights: ShapedBuffer<N, CPU>,
        result: MutableShapedBuffer<N, CPU>,
    ) {
        let geometry = GatedRecurrentUnitGeometry(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        let output = result.elementPointer
        let gates = GatedRecurrentUnitScratch<N>(count: geometry.count)
        defer {
            gates.deallocate()
        }
        // The candidate state is written into the result, which becomes the new state.
        geometry.computeGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, into: gates, candidate: output)
        let (h, z) = (state.elementPointer, gates.update)
        for i in 0 ..< geometry.count {
            let (previous, candidate) = (h[i], output[i])
            output[i] = previous + z[i] * (candidate - previous)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: ShapedBuffer<N, CPU>,
        resetInput: ShapedBuffer<N, CPU>,
        candidateInput: ShapedBuffer<N, CPU>,
        state: ShapedBuffer<N, CPU>,
        updateWeights: ShapedBuffer<N, CPU>,
        resetWeights: ShapedBuffer<N, CPU>,
        candidateWeights: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        gradients: GatedRecurrentUnitGradients<GradientBuffer<N, CPU>?>,
    ) {
        let geometry = GatedRecurrentUnitGeometry(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        precondition(outputGradient.shape == state.shape, "The gradient of the new state must have the shape of the state.")
        let (count, batchSize, hiddenSize) = (geometry.count, geometry.batchSize, geometry.hiddenSize)
        let (h, g) = (state.elementPointer, outputGradient.elementPointer)
        let (uz, ur, uh) = (updateWeights.elementPointer, resetWeights.elementPointer, candidateWeights.elementPointer)
        let gates = GatedRecurrentUnitScratch<N>(count: count)
        let candidate = UnsafeMutablePointer<N>.allocate(capacity: count)
        let updateActivationGradient = UnsafeMutablePointer<N>.allocate(capacity: count)
        let candidateActivationGradient = UnsafeMutablePointer<N>.allocate(capacity: count)
        defer {
            gates.deallocate()
            candidate.deallocate()
            updateActivationGradient.deallocate()
            candidateActivationGradient.deallocate()
        }
        // The gates are computed again instead of being kept alive between the forward and the backward pass.
        geometry.computeGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, into: gates, candidate: candidate)
        let (update, reset, resetState) = (gates.update, gates.reset, gates.resetState)
        // The new state is state + update * (candidate - state).
        for i in 0 ..< count {
            let (gradient, z, c, previous) = (g[i], update[i], candidate[i], h[i])
            updateActivationGradient[i] = gradient * (c - previous) * z * (1 - z)
            candidateActivationGradient[i] = gradient * z * (1 - c * c)
        }
        // The weight gradients are added to the accumulated gradients with GEMMs, so a weight that every time step uses needs no temporary gradient.
        let matrixShape = (batchSize, hiddenSize)
        let weightShape = (hiddenSize, hiddenSize)
        if let (target, beta) = gradients.updateInput?.elementsToWrite() {
            CPUKernels.store(updateActivationGradient, into: target, beta: beta, count: count)
        }
        if let (target, beta) = gradients.candidateInput?.elementsToWrite() {
            CPUKernels.store(candidateActivationGradient, into: target, beta: beta, count: count)
        }
        if let (target, beta) = gradients.updateWeights?.elementsToWrite() {
            CPUKernels.gemm(h, shape: matrixShape, lhsTransposed: true, updateActivationGradient, shape: matrixShape, into: target, beta: beta)
        }
        if let (target, beta) = gradients.candidateWeights?.elementsToWrite() {
            CPUKernels.gemm(resetState, shape: matrixShape, lhsTransposed: true, candidateActivationGradient, shape: matrixShape, into: target, beta: beta)
        }
        guard gradients.resetInput != nil || gradients.state != nil || gradients.resetWeights != nil else {
            return
        }
        // The candidate and the reset state are not needed anymore, so their buffers take the gradients of the reset path.
        let resetStateGradient = candidate
        CPUKernels.gemm(candidateActivationGradient, shape: matrixShape, uh, shape: weightShape, rhsTransposed: true, into: resetStateGradient)
        let resetActivationGradient = resetState
        for i in 0 ..< count {
            let (gradient, r) = (resetStateGradient[i], reset[i])
            resetActivationGradient[i] = gradient * h[i] * r * (1 - r)
        }
        if let (target, beta) = gradients.resetInput?.elementsToWrite() {
            CPUKernels.store(resetActivationGradient, into: target, beta: beta, count: count)
        }
        if let (target, beta) = gradients.resetWeights?.elementsToWrite() {
            CPUKernels.gemm(h, shape: matrixShape, lhsTransposed: true, resetActivationGradient, shape: matrixShape, into: target, beta: beta)
        }
        if let (target, beta) = gradients.state?.elementsToWrite() {
            // The element-wise part of the state gradient replaces the gradient of the reset state, which it reads first.
            let elementwise = resetStateGradient
            for i in 0 ..< count {
                let (gradient, z, fromResetState, r) = (g[i], update[i], resetStateGradient[i], reset[i])
                elementwise[i] = gradient * (1 - z) + fromResetState * r
            }
            CPUKernels.store(elementwise, into: target, beta: beta, count: count)
            CPUKernels.gemm(resetActivationGradient, shape: matrixShape, ur, shape: weightShape, rhsTransposed: true, into: target, beta: 1)
            CPUKernels.gemm(updateActivationGradient, shape: matrixShape, uz, shape: weightShape, rhsTransposed: true, into: target, beta: 1)
        }
    }
}

/// Scratch buffers of the gates of one step of a gated recurrent unit.
struct GatedRecurrentUnitScratch<N: NumericType> {
    /// Update gate
    let update: UnsafeMutablePointer<N>
    /// Reset gate
    let reset: UnsafeMutablePointer<N>
    /// Reset state, `reset * state`
    let resetState: UnsafeMutablePointer<N>

    init(count: Int) {
        update = .allocate(capacity: count)
        reset = .allocate(capacity: count)
        resetState = .allocate(capacity: count)
    }

    func deallocate() {
        update.deallocate()
        reset.deallocate()
        resetState.deallocate()
    }
}

/// Shapes of one step of a gated recurrent unit.
struct GatedRecurrentUnitGeometry {
    let batchSize: Int
    let hiddenSize: Int

    var count: Int {
        batchSize * hiddenSize
    }

    /// The shapes of a step. The arguments must have the shapes that ``FusedOperationsType/gatedRecurrentUnitStep(updateInput:resetInput:candidateInput:state:updateWeights:resetWeights:candidateWeights:result:)`` states.
    init<N>(updateInput: ShapedBuffer<N, CPU>, resetInput: ShapedBuffer<N, CPU>, candidateInput: ShapedBuffer<N, CPU>, state: ShapedBuffer<N, CPU>, updateWeights: ShapedBuffer<N, CPU>, resetWeights: ShapedBuffer<N, CPU>, candidateWeights: ShapedBuffer<N, CPU>) {
        precondition(state.dim == 2, "The state must be a matrix.")
        precondition([updateInput, resetInput, candidateInput].allSatisfy { $0.shape == state.shape }, "The inputs of the gates must have the shape of the state.")
        let weightShape = [state.shape[1], state.shape[1]]
        precondition([updateWeights, resetWeights, candidateWeights].allSatisfy { $0.shape == weightShape }, "The weights must have the shape [hiddenSize, hiddenSize].")
        batchSize = state.shape[0]
        hiddenSize = state.shape[1]
    }

    /// Computes the update gate, the reset gate, the reset state, and the candidate state.
    @inline(__always)
    func computeGates<N: NumericType>(
        updateInput: ShapedBuffer<N, CPU>,
        resetInput: ShapedBuffer<N, CPU>,
        candidateInput: ShapedBuffer<N, CPU>,
        state: ShapedBuffer<N, CPU>,
        updateWeights: ShapedBuffer<N, CPU>,
        resetWeights: ShapedBuffer<N, CPU>,
        candidateWeights: ShapedBuffer<N, CPU>,
        into gates: GatedRecurrentUnitScratch<N>,
        candidate: UnsafeMutablePointer<N>,
    ) {
        let shape = (batchSize, hiddenSize)
        let weightShape = (hiddenSize, hiddenSize)
        let h = state.elementPointer
        let (update, reset, resetState) = (gates.update, gates.reset, gates.resetState)
        update.update(from: updateInput.elementPointer, count: count)
        CPUKernels.gemm(h, shape: shape, updateWeights.elementPointer, shape: weightShape, into: update, beta: 1)
        reset.update(from: resetInput.elementPointer, count: count)
        CPUKernels.gemm(h, shape: shape, resetWeights.elementPointer, shape: weightShape, into: reset, beta: 1)
        // The activations run in blocks, so that the passes of the sigmoid over a block stay in the cache.
        CPUKernels.forEachBlock(count: count) { offset, length in
            CPUKernels.sigmoid(update + offset, into: update + offset, count: length)
            CPUKernels.sigmoid(reset + offset, into: reset + offset, count: length)
        }
        for i in 0 ..< count {
            resetState[i] = reset[i] * h[i]
        }
        // The candidate starts as its pre-activation.
        candidate.update(from: candidateInput.elementPointer, count: count)
        CPUKernels.gemm(resetState, shape: shape, candidateWeights.elementPointer, shape: weightShape, into: candidate, beta: 1)
        CPUKernels.tanh(candidate, into: candidate, count: count)
    }
}

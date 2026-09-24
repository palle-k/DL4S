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
// element-wise loops over blocks.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: Tensor<N, CPU>,
        resetInput: Tensor<N, CPU>,
        candidateInput: Tensor<N, CPU>,
        state: Tensor<N, CPU>,
        updateWeights: Tensor<N, CPU>,
        resetWeights: Tensor<N, CPU>,
        candidateWeights: Tensor<N, CPU>,
    ) -> Tensor<N, CPU> {
        guard let geometry = GatedRecurrentUnitGeometry(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights) else {
            return DefaultFusedOperations<CPU>.gatedRecurrentUnitStep(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        }
        let (result, output) = CPUKernels.makeTensor(shape: state.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let count = geometry.count
        CPUKernels.withScratch(N.self, count: 3 * count + CPUKernels.blockSize) { scratch in
            let (update, reset, resetState) = (scratch, scratch + count, scratch + 2 * count)
            let blockScratch = scratch + 3 * count
            geometry.gates(
                updateInput: updateInput.elementPointer, resetInput: resetInput.elementPointer, candidateInput: candidateInput.elementPointer,
                state: state.elementPointer, updateWeights: updateWeights.elementPointer, resetWeights: resetWeights.elementPointer, candidateWeights: candidateWeights.elementPointer,
                update: update, reset: reset, resetState: resetState, candidate: output, scratch: blockScratch,
            )
            let h = state.elementPointer
            for i in 0 ..< count {
                let (previous, candidate) = (h[i], output[i])
                output[i] = previous + update[i] * (candidate - previous)
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: Tensor<N, CPU>,
        resetInput: Tensor<N, CPU>,
        candidateInput: Tensor<N, CPU>,
        state: Tensor<N, CPU>,
        updateWeights: Tensor<N, CPU>,
        resetWeights: Tensor<N, CPU>,
        candidateWeights: Tensor<N, CPU>,
        outputGradient: Tensor<N, CPU>,
    ) -> GatedRecurrentUnitGradients<N, CPU> {
        guard let geometry = GatedRecurrentUnitGeometry(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights),
              outputGradient.shape == state.shape
        else {
            return DefaultFusedOperations<CPU>.gatedRecurrentUnitStepBackward(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, outputGradient: outputGradient)
        }
        let (count, batchSize, hiddenSize) = (geometry.count, geometry.batchSize, geometry.hiddenSize)
        let computes = [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights].map(\.requiresGradient)
        let (h, g) = (state.elementPointer, outputGradient.elementPointer)
        let (uz, ur, uh) = (updateWeights.elementPointer, resetWeights.elementPointer, candidateWeights.elementPointer)
        var gradients = GatedRecurrentUnitGradients<N, CPU>()

        CPUKernels.withScratch(N.self, count: 6 * count + CPUKernels.blockSize) { scratch in
            let (update, reset, resetState, candidate) = (scratch, scratch + count, scratch + 2 * count, scratch + 3 * count)
            let (updateActivationGradient, candidateActivationGradient) = (scratch + 4 * count, scratch + 5 * count)
            // The gates are computed again instead of being kept alive between the forward and the backward pass.
            geometry.gates(
                updateInput: updateInput.elementPointer, resetInput: resetInput.elementPointer, candidateInput: candidateInput.elementPointer,
                state: h, updateWeights: uz, resetWeights: ur, candidateWeights: uh,
                update: update, reset: reset, resetState: resetState, candidate: candidate, scratch: scratch + 6 * count,
            )
            // The new state is state + update * (candidate - state).
            for i in 0 ..< count {
                let (gradient, z, c, previous) = (g[i], update[i], candidate[i], h[i])
                updateActivationGradient[i] = gradient * (c - previous) * z * (1 - z)
                candidateActivationGradient[i] = gradient * z * (1 - c * c)
            }
            if computes[0] {
                gradients.updateInput = copy(updateActivationGradient, shape: state.shape)
            }
            if computes[2] {
                gradients.candidateInput = copy(candidateActivationGradient, shape: state.shape)
            }
            if computes[4] {
                gradients.updateWeights = product(state: h, gradient: updateActivationGradient, batchSize: batchSize, hiddenSize: hiddenSize)
            }
            if computes[6] {
                gradients.candidateWeights = product(state: resetState, gradient: candidateActivationGradient, batchSize: batchSize, hiddenSize: hiddenSize)
            }
            guard computes[1] || computes[3] || computes[5] else {
                return
            }
            // The gate buffers of the reset state and the candidate are free now.
            let resetStateGradient = candidate
            CPUKernels.gemm(candidateActivationGradient, shape: (batchSize, hiddenSize), uh, shape: (hiddenSize, hiddenSize), rhsTransposed: true, into: resetStateGradient)
            let resetActivationGradient = resetState
            for i in 0 ..< count {
                let (gradient, r) = (resetStateGradient[i], reset[i])
                resetActivationGradient[i] = gradient * h[i] * r * (1 - r)
            }
            if computes[1] {
                gradients.resetInput = copy(resetActivationGradient, shape: state.shape)
            }
            if computes[5] {
                gradients.resetWeights = product(state: h, gradient: resetActivationGradient, batchSize: batchSize, hiddenSize: hiddenSize)
            }
            if computes[3] {
                let (stateGradient, dh) = CPUKernels.makeTensor(shape: state.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
                for i in 0 ..< count {
                    let (gradient, z, fromResetState, r) = (g[i], update[i], resetStateGradient[i], reset[i])
                    dh[i] = gradient * (1 - z) + fromResetState * r
                }
                CPUKernels.gemm(resetActivationGradient, shape: (batchSize, hiddenSize), ur, shape: (hiddenSize, hiddenSize), rhsTransposed: true, into: dh, beta: 1)
                CPUKernels.gemm(updateActivationGradient, shape: (batchSize, hiddenSize), uz, shape: (hiddenSize, hiddenSize), rhsTransposed: true, into: dh, beta: 1)
                gradients.state = stateGradient
            }
        }
        return gradients
    }
}

private extension CPUFusedOperations {
    /// Returns a tensor without context with a copy of the elements.
    static func copy<N: NumericType>(_ values: UnsafePointer<N>, shape: [Int]) -> Tensor<N, CPU> {
        let (tensor, pointer) = CPUKernels.makeTensor(shape: shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        pointer.update(from: values, count: tensor.count)
        return tensor
    }

    /// Returns `stateᵀ × gradient`, the gradient of the weights of a gate, shape [hiddenSize, hiddenSize].
    static func product<N: NumericType>(state: UnsafePointer<N>, gradient: UnsafePointer<N>, batchSize: Int, hiddenSize: Int) -> Tensor<N, CPU> {
        let (tensor, pointer) = CPUKernels.makeTensor(shape: [hiddenSize, hiddenSize]) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        CPUKernels.gemm(state, shape: (batchSize, hiddenSize), lhsTransposed: true, gradient, shape: (batchSize, hiddenSize), into: pointer)
        return tensor
    }
}

/// Shapes of one step of a gated recurrent unit.
struct GatedRecurrentUnitGeometry {
    let batchSize: Int
    let hiddenSize: Int

    var count: Int {
        batchSize * hiddenSize
    }

    /// Returns nil for shapes that the kernels do not support.
    init?<N>(updateInput: Tensor<N, CPU>, resetInput: Tensor<N, CPU>, candidateInput: Tensor<N, CPU>, state: Tensor<N, CPU>, updateWeights: Tensor<N, CPU>, resetWeights: Tensor<N, CPU>, candidateWeights: Tensor<N, CPU>) {
        guard state.dim == 2, state.count > 0, [updateInput, resetInput, candidateInput].allSatisfy({ $0.shape == state.shape }) else {
            return nil
        }
        let weightShape = [state.shape[1], state.shape[1]]
        guard [updateWeights, resetWeights, candidateWeights].allSatisfy({ $0.shape == weightShape }) else {
            return nil
        }
        batchSize = state.shape[0]
        hiddenSize = state.shape[1]
    }

    /// Computes the update gate, the reset gate, the reset state `reset * state`, and the candidate state.
    /// `scratch` holds `CPUKernels.blockSize` elements.
    @inline(__always)
    func gates<N: NumericType>(
        updateInput: UnsafePointer<N>,
        resetInput: UnsafePointer<N>,
        candidateInput: UnsafePointer<N>,
        state: UnsafePointer<N>,
        updateWeights: UnsafePointer<N>,
        resetWeights: UnsafePointer<N>,
        candidateWeights: UnsafePointer<N>,
        update: UnsafeMutablePointer<N>,
        reset: UnsafeMutablePointer<N>,
        resetState: UnsafeMutablePointer<N>,
        candidate: UnsafeMutablePointer<N>,
        scratch: UnsafeMutablePointer<N>,
    ) {
        let shape = (batchSize, hiddenSize)
        let weightShape = (hiddenSize, hiddenSize)
        update.update(from: updateInput, count: count)
        CPUKernels.gemm(state, shape: shape, updateWeights, shape: weightShape, into: update, beta: 1)
        reset.update(from: resetInput, count: count)
        CPUKernels.gemm(state, shape: shape, resetWeights, shape: weightShape, into: reset, beta: 1)
        applySigmoid(update, scratch: scratch)
        applySigmoid(reset, scratch: scratch)
        for i in 0 ..< count {
            resetState[i] = reset[i] * state[i]
        }
        // The candidate starts as its pre-activation, and tanh writes through the scratch buffer.
        candidate.update(from: candidateInput, count: count)
        CPUKernels.gemm(resetState, shape: shape, candidateWeights, shape: weightShape, into: candidate, beta: 1)
        CPUKernels.forEachBlock(count: count) { offset, length in
            let block = candidate + offset
            scratch.update(from: block, count: length)
            CPUKernels.tanh(scratch, into: block, count: length)
        }
    }

    /// Replaces the values with their sigmoid, `tanh(x / 2) / 2 + 1 / 2`.
    @inline(__always)
    private func applySigmoid<N: NumericType>(_ values: UnsafeMutablePointer<N>, scratch: UnsafeMutablePointer<N>) {
        let half = N(0.5)
        CPUKernels.forEachBlock(count: count) { offset, length in
            let block = values + offset
            for i in 0 ..< length {
                scratch[i] = block[i] * half
            }
            CPUKernels.tanh(scratch, into: block, count: length)
            for i in 0 ..< length {
                block[i] = block[i] * half + half
            }
        }
    }
}

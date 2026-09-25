//
//  GPUFusedTraining.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation

public extension GPUFusedOperations {
    static func adamUpdate<N: NumericType>(
        parameter: Tensor<N, GPU>,
        gradient: Tensor<N, GPU>,
        firstMoment: inout Tensor<N, GPU>,
        secondMoment: inout Tensor<N, GPU>,
        secondMomentMax: inout Tensor<N, GPU>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
    ) -> Tensor<N, GPU> {
        let shape = parameter.shape
        guard gradient.shape == shape, firstMoment.shape == shape, secondMoment.shape == shape, secondMomentMax.map({ $0.shape == shape }) ?? true,
              GPUFused.runsKernel(N.self, elements: parameter.count, reading: [parameter, gradient, firstMoment, secondMoment])
        else {
            return DefaultFusedOperations<GPU>.adamUpdate(
                parameter: parameter, gradient: gradient, firstMoment: &firstMoment, secondMoment: &secondMoment, secondMomentMax: &secondMomentMax,
                learningRate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon, beta1Power: beta1Power, beta2Power: beta2Power,
            )
        }
        let result: Tensor<N, GPU> = GPUFused.makeTensor(shape: shape)
        // The optimizer owns the moments, so the writes do not copy them.
        let (p, g, updated) = (parameter.gpuBuffer, gradient.gpuBuffer, result.gpuBuffer)
        let (m, v) = (firstMoment.mutableGPUBuffer, secondMoment.mutableGPUBuffer)
        let maximum = secondMomentMax != nil ? secondMomentMax!.mutableGPUBuffer : nil
        let parameters = AdamParameters(
            count: UInt32(parameter.count),
            amsgrad: maximum != nil ? 1 : 0,
            learningRate: learningRate.floatValue,
            beta1: beta1.floatValue,
            beta2: beta2.floatValue,
            epsilon: epsilon.floatValue,
            firstCorrection: 1 / (1 - beta1Power.floatValue),
            secondCorrection: 1 / (1 - beta2Power.floatValue),
        )
        let moments = [m, v] + (maximum.map { [$0] } ?? [])
        let pipeline = GPUKernels.pipeline("adam_update", in: .fused)
        GPUContext.current.compute(pipeline, reading: [p, g] + moments, writing: moments + [updated]) { arguments in
            arguments.buffer(p)
            arguments.buffer(g)
            arguments.buffer(m)
            arguments.buffer(v)
            arguments.buffer(maximum ?? v)
            arguments.buffer(updated)
            arguments.value(parameters)
            arguments.dispatch(count: parameter.count)
        }
        return result
    }

    // One step of a gated recurrent unit is three matrix products and two element-wise kernels.
    // The backward pass computes the gates again, then the gradients with three element-wise kernels and the products.

    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: Tensor<N, GPU>,
        resetInput: Tensor<N, GPU>,
        candidateInput: Tensor<N, GPU>,
        state: Tensor<N, GPU>,
        updateWeights: Tensor<N, GPU>,
        resetWeights: Tensor<N, GPU>,
        candidateWeights: Tensor<N, GPU>,
    ) -> Tensor<N, GPU> {
        let inputs = [updateInput, resetInput, candidateInput, state]
        guard state.dim == 2, inputs.allSatisfy({ $0.shape == state.shape }),
              GPUFused.runsKernel(N.self, elements: state.count, reading: inputs + [updateWeights, resetWeights, candidateWeights])
        else {
            return DefaultFusedOperations<GPU>.gatedRecurrentUnitStep(
                updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state,
                updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights,
            )
        }
        let gates = GatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        let result: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        let (z, input, product, h, y) = (gates.update.gpuBuffer, candidateInput.gpuBuffer, gates.candidateProduct.gpuBuffer, state.gpuBuffer, result.gpuBuffer)
        let count = UInt32(state.count)
        let pipeline = GPUKernels.pipeline("gru_state", in: .fused)
        GPUContext.current.compute(pipeline, reading: [z, input, product, h], writing: [y]) { arguments in
            arguments.buffer(z)
            arguments.buffer(input)
            arguments.buffer(product)
            arguments.buffer(h)
            arguments.buffer(y)
            arguments.value(count)
            arguments.dispatch(count: Int(count))
        }
        return result
    }

    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: Tensor<N, GPU>,
        resetInput: Tensor<N, GPU>,
        candidateInput: Tensor<N, GPU>,
        state: Tensor<N, GPU>,
        updateWeights: Tensor<N, GPU>,
        resetWeights: Tensor<N, GPU>,
        candidateWeights: Tensor<N, GPU>,
        outputGradient: Tensor<N, GPU>,
        accumulating gradients: inout GatedRecurrentUnitGradients<N, GPU>,
    ) {
        let inputs = [updateInput, resetInput, candidateInput, state, outputGradient]
        guard state.dim == 2, inputs.allSatisfy({ $0.shape == state.shape }),
              GPUFused.runsKernel(N.self, elements: state.count, reading: inputs + [updateWeights, resetWeights, candidateWeights])
        else {
            DefaultFusedOperations<GPU>.gatedRecurrentUnitStepBackward(
                updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state,
                updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights,
                outputGradient: outputGradient, accumulating: &gradients,
            )
            return
        }
        let gates = GatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        let count = UInt32(state.count)
        let updateGradient: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        let candidateGradient: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        let stateGradient: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        do {
            let (z, input, product, h, g) = (gates.update.gpuBuffer, candidateInput.gpuBuffer, gates.candidateProduct.gpuBuffer, state.gpuBuffer, outputGradient.gpuBuffer)
            let (dz, dc, dh) = (updateGradient.gpuBuffer, candidateGradient.gpuBuffer, stateGradient.gpuBuffer)
            let pipeline = GPUKernels.pipeline("gru_backward_gates", in: .fused)
            GPUContext.current.compute(pipeline, reading: [z, input, product, h, g], writing: [dz, dc, dh]) { arguments in
                for buffer in [z, input, product, h, g, dz, dc, dh] {
                    arguments.buffer(buffer)
                }
                arguments.value(count)
                arguments.dispatch(count: Int(count))
            }
        }
        if updateWeights.requiresGradient {
            Tensor.accumulateProduct(state.detached(), updateGradient, transposeLhs: true, into: &gradients.updateWeights)
        }
        if candidateWeights.requiresGradient {
            Tensor.accumulateProduct(gates.resetState, candidateGradient, transposeLhs: true, into: &gradients.candidateWeights)
        }
        if updateInput.requiresGradient {
            Tensor.accumulate(updateGradient, into: &gradients.updateInput)
        }
        if candidateInput.requiresGradient {
            Tensor.accumulate(candidateGradient, into: &gradients.candidateInput)
        }
        guard resetInput.requiresGradient || state.requiresGradient || resetWeights.requiresGradient else {
            return
        }
        let resetStateGradient = candidateGradient.matrixMultiplied(with: candidateWeights.detached(), transposeOther: true)
        let resetGradient: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        do {
            let (r, h, drs, dr, dh) = (gates.reset.gpuBuffer, state.gpuBuffer, resetStateGradient.gpuBuffer, resetGradient.gpuBuffer, stateGradient.gpuBuffer)
            let pipeline = GPUKernels.pipeline("gru_backward_reset", in: .fused)
            GPUContext.current.compute(pipeline, reading: [r, h, drs, dh], writing: [dr, dh]) { arguments in
                for buffer in [r, h, drs, dr, dh] {
                    arguments.buffer(buffer)
                }
                arguments.value(count)
                arguments.dispatch(count: Int(count))
            }
        }
        if resetWeights.requiresGradient {
            Tensor.accumulateProduct(state.detached(), resetGradient, transposeLhs: true, into: &gradients.resetWeights)
        }
        if resetInput.requiresGradient {
            Tensor.accumulate(resetGradient, into: &gradients.resetInput)
        }
        if state.requiresGradient {
            // The products of the gate gradients with the weights are added to the state gradient in place.
            var accumulated: Tensor<N, GPU>? = consume stateGradient
            Tensor.accumulateProduct(resetGradient, resetWeights.detached(), transposeRhs: true, into: &accumulated)
            Tensor.accumulateProduct(updateGradient, updateWeights.detached(), transposeRhs: true, into: &accumulated)
            Tensor.accumulate(accumulated, into: &gradients.state)
        }
    }
}

/// Gates of one step of a gated recurrent unit on the GPU.
private struct GatedRecurrentUnitGates<N: NumericType> {
    /// Update gate
    let update: Tensor<N, GPU>
    /// Reset gate
    let reset: Tensor<N, GPU>
    /// Reset gate times the previous state
    let resetState: Tensor<N, GPU>
    /// Product of the reset state with the candidate weights, without the candidate input
    let candidateProduct: Tensor<N, GPU>

    init(updateInput: Tensor<N, GPU>, resetInput: Tensor<N, GPU>, candidateInput _: Tensor<N, GPU>, state: Tensor<N, GPU>, updateWeights: Tensor<N, GPU>, resetWeights: Tensor<N, GPU>, candidateWeights: Tensor<N, GPU>) {
        let state = state.detached()
        let update = state.matrixMultiplied(with: updateWeights.detached())
        let reset = state.matrixMultiplied(with: resetWeights.detached())
        let resetState: Tensor<N, GPU> = GPUFused.makeTensor(shape: state.shape)
        let (zInput, z, rInput, r, h, rs) = (updateInput.gpuBuffer, update.gpuBuffer, resetInput.gpuBuffer, reset.gpuBuffer, state.gpuBuffer, resetState.gpuBuffer)
        let count = UInt32(state.count)
        let pipeline = GPUKernels.pipeline("gru_gates", in: .fused)
        GPUContext.current.compute(pipeline, reading: [zInput, z, rInput, r, h], writing: [z, r, rs]) { arguments in
            for buffer in [zInput, z, rInput, r, h, rs] {
                arguments.buffer(buffer)
            }
            arguments.value(count)
            arguments.dispatch(count: Int(count))
        }
        self.update = update
        self.reset = reset
        self.resetState = resetState
        candidateProduct = resetState.matrixMultiplied(with: candidateWeights.detached())
    }
}

private struct AdamParameters {
    var count: UInt32
    var amsgrad: UInt32
    var learningRate: Float
    var beta1: Float
    var beta2: Float
    var epsilon: Float
    var firstCorrection: Float
    var secondCorrection: Float
}
#endif

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
        parameter: ShapedBuffer<N, GPU>,
        gradient: ShapedBuffer<N, GPU>,
        firstMoment: MutableShapedBuffer<N, GPU>,
        secondMoment: MutableShapedBuffer<N, GPU>,
        secondMomentMax: MutableShapedBuffer<N, GPU>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
        result: MutableShapedBuffer<N, GPU>,
    ) {
        let shape = parameter.shape
        precondition(
            [gradient.shape, firstMoment.shape, secondMoment.shape, result.shape].allSatisfy { $0 == shape } && (secondMomentMax.map { $0.shape == shape } ?? true),
            "The gradient, the moments, and the result must have the shape of the parameter.",
        )
        guard GPUFused.runsKernel(N.self, elements: parameter.count, reading: [parameter.gpuBuffer, gradient.gpuBuffer], writing: [firstMoment.gpuBuffer, secondMoment.gpuBuffer, result.gpuBuffer] + (secondMomentMax.map { [$0.gpuBuffer] } ?? [])) else {
            DefaultFusedOperations<GPU>.adamUpdate(
                parameter: parameter, gradient: gradient, firstMoment: firstMoment, secondMoment: secondMoment, secondMomentMax: secondMomentMax,
                learningRate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon, beta1Power: beta1Power, beta2Power: beta2Power, result: result,
            )
            return
        }
        let (p, g, updated) = (parameter.gpuBuffer, gradient.gpuBuffer, result.gpuBuffer)
        let (m, v, maximum) = (firstMoment.gpuBuffer, secondMoment.gpuBuffer, secondMomentMax?.gpuBuffer)
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
        let kernel = GPUKernels.kernel("adam_update", in: .fused)
        GPUContext.compute(kernel, reading: [p, g] + moments, writing: moments + [updated]) { arguments in
            arguments.buffer(p)
            arguments.buffer(g)
            arguments.buffer(m)
            arguments.buffer(v)
            arguments.buffer(maximum ?? v)
            arguments.buffer(updated)
            arguments.value(parameters)
            arguments.dispatch(count: parameter.count)
        }
    }

    // One step of a gated recurrent unit is three matrix products and two element-wise kernels.
    // The backward pass computes the gates again, then the gradients with three element-wise kernels and the products.

    static func gatedRecurrentUnitStep<N: NumericType>(
        updateInput: ShapedBuffer<N, GPU>,
        resetInput: ShapedBuffer<N, GPU>,
        candidateInput: ShapedBuffer<N, GPU>,
        state: ShapedBuffer<N, GPU>,
        updateWeights: ShapedBuffer<N, GPU>,
        resetWeights: ShapedBuffer<N, GPU>,
        candidateWeights: ShapedBuffer<N, GPU>,
        result: MutableShapedBuffer<N, GPU>,
    ) {
        GPUGatedRecurrentUnitGates<N>.checkShapes(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        guard GPUFused.runsKernel(N.self, elements: state.count, reading: [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights].map { $0.gpuBuffer }, writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.gatedRecurrentUnitStep(
                updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state,
                updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, result: result,
            )
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let gates = GPUGatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, math: math)
        let (z, input, product, h, y) = (gates.update.gpuBuffer, candidateInput.gpuBuffer, gates.candidateProduct.gpuBuffer, state.gpuBuffer, result.gpuBuffer)
        GPUGatedRecurrentUnitKernel.record("gru_state", buffers: [z, input, product, h, y], reading: [z, input, product, h], writing: [y], count: state.count)
    }

    static func gatedRecurrentUnitStepBackward<N: NumericType>(
        updateInput: ShapedBuffer<N, GPU>,
        resetInput: ShapedBuffer<N, GPU>,
        candidateInput: ShapedBuffer<N, GPU>,
        state: ShapedBuffer<N, GPU>,
        updateWeights: ShapedBuffer<N, GPU>,
        resetWeights: ShapedBuffer<N, GPU>,
        candidateWeights: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        gradients: GatedRecurrentUnitGradients<GradientBuffer<N, GPU>?>,
    ) {
        GPUGatedRecurrentUnitGates<N>.checkShapes(updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights)
        precondition(outputGradient.shape == state.shape, "The gradient of the new state must have the shape of the state.")
        guard GPUFused.runsKernel(N.self, elements: state.count, reading: [updateInput, resetInput, candidateInput, state, updateWeights, resetWeights, candidateWeights, outputGradient].map { $0.gpuBuffer }, writing: [gradients.updateInput, gradients.resetInput, gradients.candidateInput, gradients.state, gradients.updateWeights, gradients.resetWeights, gradients.candidateWeights].compactMap { $0?.gpuBuffer }) else {
            DefaultFusedOperations<GPU>.gatedRecurrentUnitStepBackward(
                updateInput: updateInput, resetInput: resetInput, candidateInput: candidateInput, state: state,
                updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights,
                outputGradient: outputGradient, gradients: gradients,
            )
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let gates = GPUGatedRecurrentUnitGates(updateInput: updateInput, resetInput: resetInput, state: state, updateWeights: updateWeights, resetWeights: resetWeights, candidateWeights: candidateWeights, math: math)
        let (updateGradient, candidateGradient, stateGradient) = (math.temporary(state.shape), math.temporary(state.shape), math.temporary(state.shape))
        let (z, input, product, h, g) = (gates.update.gpuBuffer, candidateInput.gpuBuffer, gates.candidateProduct.gpuBuffer, state.gpuBuffer, outputGradient.gpuBuffer)
        let (dz, dc, dh) = (updateGradient.gpuBuffer, candidateGradient.gpuBuffer, stateGradient.gpuBuffer)
        GPUGatedRecurrentUnitKernel.record("gru_backward_gates", buffers: [z, input, product, h, g, dz, dc, dh], reading: [z, input, product, h, g], writing: [dz, dc, dh], count: state.count)
        if let weightGradient = gradients.updateWeights {
            math.multiplyMatrices(state, updateGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        if let weightGradient = gradients.candidateWeights {
            math.multiplyMatrices(gates.resetState, candidateGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        math.writeSum(of: updateGradient, into: gradients.updateInput)
        math.writeSum(of: candidateGradient, into: gradients.candidateInput)
        guard gradients.resetInput != nil || gradients.state != nil || gradients.resetWeights != nil else {
            return
        }
        let (resetStateGradient, resetGradient) = (math.temporary(state.shape), math.temporary(state.shape))
        math.multiplyMatrices(candidateGradient, candidateWeights, rhsTransposed: true, into: resetStateGradient)
        let (r, drs, dr) = (gates.reset.gpuBuffer, resetStateGradient.gpuBuffer, resetGradient.gpuBuffer)
        // The kernel adds the gradient through the reset gate to the gradient of the state.
        GPUGatedRecurrentUnitKernel.record("gru_backward_reset", buffers: [r, h, drs, dr, dh], reading: [r, h, drs, dh], writing: [dr, dh], count: state.count)
        if let weightGradient = gradients.resetWeights {
            math.multiplyMatrices(state, resetGradient, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
        }
        math.writeSum(of: resetGradient, into: gradients.resetInput)
        if let gradient = gradients.state {
            // The products of the gate gradients with the weights are added to the gradient of the state in place.
            math.multiplyMatrices(resetGradient, resetWeights, rhsTransposed: true, into: stateGradient, beta: 1)
            math.multiplyMatrices(updateGradient, updateWeights, rhsTransposed: true, into: stateGradient, beta: 1)
            math.writeSum(of: stateGradient, into: gradient)
        }
    }
}

/// Gates of one step of a gated recurrent unit on the GPU, in intermediate buffers.
private struct GPUGatedRecurrentUnitGates<N: NumericType> {
    /// Update gate
    let update: MutableShapedBuffer<N, GPU>
    /// Reset gate
    let reset: MutableShapedBuffer<N, GPU>
    /// Reset gate times the previous state
    let resetState: MutableShapedBuffer<N, GPU>
    /// Product of the reset state with the candidate weights, without the candidate input
    let candidateProduct: MutableShapedBuffer<N, GPU>

    /// Checks the shapes that ``FusedOperationsType/gatedRecurrentUnitStep(updateInput:resetInput:candidateInput:state:updateWeights:resetWeights:candidateWeights:result:)`` states.
    static func checkShapes(updateInput: ShapedBuffer<N, GPU>, resetInput: ShapedBuffer<N, GPU>, candidateInput: ShapedBuffer<N, GPU>, state: ShapedBuffer<N, GPU>, updateWeights: ShapedBuffer<N, GPU>, resetWeights: ShapedBuffer<N, GPU>, candidateWeights: ShapedBuffer<N, GPU>) {
        precondition(state.dim == 2, "The state must be a matrix.")
        precondition([updateInput, resetInput, candidateInput].allSatisfy { $0.shape == state.shape }, "The inputs of the gates must have the shape of the state.")
        let weightShape = [state.shape[1], state.shape[1]]
        precondition([updateWeights, resetWeights, candidateWeights].allSatisfy { $0.shape == weightShape }, "The weights must have the shape [hiddenSize, hiddenSize].")
    }

    init(updateInput: ShapedBuffer<N, GPU>, resetInput: ShapedBuffer<N, GPU>, state: ShapedBuffer<N, GPU>, updateWeights: ShapedBuffer<N, GPU>, resetWeights: ShapedBuffer<N, GPU>, candidateWeights: ShapedBuffer<N, GPU>, math: BufferMath<N, GPU>) {
        let (update, reset, resetState, candidateProduct) = (math.temporary(state.shape), math.temporary(state.shape), math.temporary(state.shape), math.temporary(state.shape))
        math.multiplyMatrices(state, updateWeights, into: update)
        math.multiplyMatrices(state, resetWeights, into: reset)
        let (zInput, z, rInput, r, h, rs) = (updateInput.gpuBuffer, update.gpuBuffer, resetInput.gpuBuffer, reset.gpuBuffer, state.gpuBuffer, resetState.gpuBuffer)
        GPUGatedRecurrentUnitKernel.record("gru_gates", buffers: [zInput, z, rInput, r, h, rs], reading: [zInput, z, rInput, r, h], writing: [z, r, rs], count: state.count)
        math.multiplyMatrices(resetState, candidateWeights, into: candidateProduct)
        self.update = update
        self.reset = reset
        self.resetState = resetState
        self.candidateProduct = candidateProduct
    }
}

/// The element-wise kernels of a gated recurrent unit, with one thread per element of the state.
private enum GPUGatedRecurrentUnitKernel {
    /// Records a kernel that receives the buffers in the given order and then the number of elements.
    static func record(_ name: String, buffers: [GPUBuffer], reading: [GPUBuffer], writing: [GPUBuffer], count: Int) {
        let elements = UInt32(count)
        GPUContext.compute(GPUKernels.kernel(name, in: .fused), reading: reading, writing: writing) { arguments in
            for buffer in buffers {
                arguments.buffer(buffer)
            }
            arguments.value(elements)
            arguments.dispatch(count: count)
        }
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

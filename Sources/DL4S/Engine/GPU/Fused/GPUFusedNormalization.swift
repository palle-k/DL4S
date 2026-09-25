//
//  GPUFusedNormalization.swift
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
import Metal

// Softmax, log softmax, and layer normalization work on rows: one threadgroup processes one row, so that the row stays
// in the cache between the passes over it. Softmax along another axis than the last one uses the default implementation.

public extension GPUFusedOperations {
    static func softmax<N: NumericType>(input: Tensor<N, GPU>, axis: Int) -> Tensor<N, GPU> {
        guard axis == input.dim - 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.softmax(input: input, axis: axis)
        }
        return rowForward("softmax_forward", input: input)
    }

    static func softmaxBackward<N: NumericType>(output: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, axis: Int, accumulating gradient: inout Tensor<N, GPU>?) {
        guard axis == output.dim - 1, output.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: output.count, reading: [output, outputGradient]) else {
            DefaultFusedOperations<GPU>.softmaxBackward(output: output, outputGradient: outputGradient, axis: axis, accumulating: &gradient)
            return
        }
        rowBackward("softmax_backward", output: output, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func logSoftmax<N: NumericType>(input: Tensor<N, GPU>, axis: Int) -> Tensor<N, GPU> {
        guard axis == input.dim - 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input]) else {
            return DefaultFusedOperations<GPU>.logSoftmax(input: input, axis: axis)
        }
        return rowForward("log_softmax_forward", input: input)
    }

    static func logSoftmaxBackward<N: NumericType>(output: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, axis: Int, accumulating gradient: inout Tensor<N, GPU>?) {
        guard axis == output.dim - 1, output.shape == outputGradient.shape, GPUFused.runsKernel(N.self, elements: output.count, reading: [output, outputGradient]) else {
            DefaultFusedOperations<GPU>.logSoftmaxBackward(output: output, outputGradient: outputGradient, axis: axis, accumulating: &gradient)
            return
        }
        rowBackward("log_softmax_backward", output: output, outputGradient: outputGradient, accumulating: &gradient)
    }

    static func layerNormalization<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, epsilon: N) -> Tensor<N, GPU> {
        guard let length = layerNormalizationRowLength(input: input, scale: scale, shift: shift),
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, shift])
        else {
            return DefaultFusedOperations<GPU>.layerNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon)
        }
        let result: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        let (x, gamma, beta, y) = (input.gpuBuffer, scale.gpuBuffer, shift.gpuBuffer, result.gpuBuffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: 0, epsilon: epsilon.floatValue)
        let pipeline = GPUKernels.pipeline("layer_norm_forward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, gamma, beta], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(gamma)
            arguments.buffer(beta)
            arguments.buffer(y)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: input.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
        return result
    }

    static func layerNormalizationBackward<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, epsilon: N, accumulating gradients: inout (input: Tensor<N, GPU>?, scale: Tensor<N, GPU>?, shift: Tensor<N, GPU>?)) {
        guard let length = layerNormalizationRowLength(input: input, scale: scale, shift: shift), outputGradient.shape == input.shape,
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, outputGradient])
        else {
            DefaultFusedOperations<GPU>.layerNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon, accumulating: &gradients)
            return
        }
        let rows = input.count / length
        let (x, gamma, g) = (input.gpuBuffer, scale.gpuBuffer, outputGradient.gpuBuffer)
        var inputTarget = input.requiresGradient ? GPUGradientTarget(taking: &gradients.input, shape: input.shape) : nil
        let inputBuffer = inputTarget?.buffer
        let accumulatesInput = inputTarget?.accumulates ?? false
        // The kernel writes the terms of the scale gradient of every row, which a column reduction adds up.
        let scaleTerms = scale.requiresGradient ? GPUKernels.temporary(count: input.count) : nil
        let writing = [inputBuffer, scaleTerms].compactMap(\.self)
        if !writing.isEmpty {
            let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: accumulatesInput ? 1 : 0, epsilon: epsilon.floatValue)
            let outputs = SIMD2<UInt32>(inputBuffer != nil ? 1 : 0, scaleTerms != nil ? 1 : 0)
            let pipeline = GPUKernels.pipeline("layer_norm_backward", in: .fused)
            GPUContext.current.compute(pipeline, reading: [x, gamma, g] + writing, writing: writing) { arguments in
                arguments.buffer(x)
                arguments.buffer(gamma)
                arguments.buffer(g)
                arguments.buffer(inputBuffer ?? x)
                arguments.buffer(scaleTerms ?? x)
                arguments.value(parameters)
                arguments.value(outputs)
                arguments.dispatch(threadgroups: MTLSize(width: rows, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
            }
        }
        inputTarget.take()?.finish(into: &gradients.input)
        if let scaleTerms {
            let target = GPUGradientTarget(taking: &gradients.scale, shape: scale.shape)
            GPUKernels.reduce(.sum, .float, values: scaleTerms, result: target.buffer, outer: 1, length: rows, inner: length, accumulate: target.accumulates)
            target.finish(into: &gradients.scale)
        }
        if shift.requiresGradient {
            let target = GPUGradientTarget(taking: &gradients.shift, shape: shift.shape)
            GPUKernels.reduce(.sum, .float, values: g, result: target.buffer, outer: 1, length: rows, inner: length, accumulate: target.accumulates)
            target.finish(into: &gradients.shift)
        }
    }

    static func batchNormalization<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, epsilon: N) -> (output: Tensor<N, GPU>, mean: Tensor<N, GPU>, variance: Tensor<N, GPU>) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, shift]),
              let gamma = broadcastColumns(scale, to: columnShape), let beta = broadcastColumns(shift, to: columnShape)
        else {
            return DefaultFusedOperations<GPU>.batchNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon)
        }
        let output: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        let mean: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let variance: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let parameters = ColumnParameters(input: input, accumulate: false, epsilon: epsilon)
        let (x, g, b, y, m, v) = (input.gpuBuffer, gamma.gpuBuffer, beta.gpuBuffer, output.gpuBuffer, mean.gpuBuffer, variance.gpuBuffer)
        let pipeline = GPUKernels.pipeline("batch_norm_forward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, g, b], writing: [y, m, v]) { arguments in
            for buffer in [x, g, b, y, m, v] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.dispatch(count: Int(parameters.columns))
        }
        return (output, mean, variance)
    }

    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, epsilon: N, accumulating gradients: inout (input: Tensor<N, GPU>?, scale: Tensor<N, GPU>?, shift: Tensor<N, GPU>?)) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, outputGradient.shape == input.shape, shift.dim <= columnShape.count,
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, outputGradient]),
              let gamma = broadcastColumns(scale, to: columnShape)
        else {
            DefaultFusedOperations<GPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon, accumulating: &gradients)
            return
        }
        var inputTarget = input.requiresGradient ? GPUGradientTarget(taking: &gradients.input, shape: input.shape) : nil
        let parameters = ColumnParameters(input: input, accumulate: inputTarget?.accumulates ?? false, epsilon: epsilon)
        let scaleColumns: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let shiftColumns: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let (x, w, g, dx, ds, db) = (input.gpuBuffer, gamma.gpuBuffer, outputGradient.gpuBuffer, inputTarget?.buffer, scaleColumns.gpuBuffer, shiftColumns.gpuBuffer)
        let pipeline = GPUKernels.pipeline("batch_norm_backward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, w, g] + (dx.map { [$0] } ?? []), writing: [ds, db] + (dx.map { [$0] } ?? [])) { arguments in
            for buffer in [x, w, g, dx ?? ds, ds, db] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.value(UInt32(dx != nil ? 1 : 0))
            arguments.dispatch(count: Int(parameters.columns))
        }
        inputTarget.take()?.finish(into: &gradients.input)
        if scale.requiresGradient {
            Tensor.accumulate(scaleColumns.reducingBroadcast(to: scale.shape), into: &gradients.scale)
        }
        if shift.requiresGradient {
            Tensor.accumulate(shiftColumns.reducingBroadcast(to: shift.shape), into: &gradients.shift)
        }
    }

    static func batchNormalization<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, mean: Tensor<N, GPU>, variance: Tensor<N, GPU>, epsilon: N) -> Tensor<N, GPU> {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, shift, mean, variance]),
              let affine = fixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon)
        else {
            return DefaultFusedOperations<GPU>.batchNormalization(input: input, scale: scale, shift: shift, mean: mean, variance: variance, epsilon: epsilon)
        }
        let output: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        let parameters = ColumnParameters(input: input, accumulate: false, epsilon: epsilon)
        let (x, factors, offsets, y) = (input.gpuBuffer, affine.factors.gpuBuffer, affine.offsets.gpuBuffer, output.gpuBuffer)
        let pipeline = GPUKernels.pipeline("affine_columns", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, factors, offsets], writing: [y]) { arguments in
            for buffer in [x, factors, offsets, y] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
        return output
    }

    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, mean: Tensor<N, GPU>, variance: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, epsilon: N, accumulating gradients: inout (input: Tensor<N, GPU>?, scale: Tensor<N, GPU>?, shift: Tensor<N, GPU>?)) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, outputGradient.shape == input.shape, shift.dim <= columnShape.count,
              GPUFused.runsKernel(N.self, elements: input.count, reading: [input, scale, mean, variance, outputGradient]),
              let affine = fixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon),
              let means = broadcastColumns(mean, to: columnShape)
        else {
            DefaultFusedOperations<GPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, mean: mean, variance: variance, outputGradient: outputGradient, epsilon: epsilon, accumulating: &gradients)
            return
        }
        var inputTarget = input.requiresGradient ? GPUGradientTarget(taking: &gradients.input, shape: input.shape) : nil
        let parameters = ColumnParameters(input: input, accumulate: inputTarget?.accumulates ?? false, epsilon: epsilon)
        let scaleColumns: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let shiftColumns: Tensor<N, GPU> = GPUFused.makeTensor(shape: columnShape)
        let (x, g, factors, divisors, mu) = (input.gpuBuffer, outputGradient.gpuBuffer, affine.factors.gpuBuffer, affine.inverseDivisors.gpuBuffer, means.gpuBuffer)
        let (dx, ds, db) = (inputTarget?.buffer, scaleColumns.gpuBuffer, shiftColumns.gpuBuffer)
        let pipeline = GPUKernels.pipeline("batch_norm_fixed_backward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, g, factors, divisors, mu] + (dx.map { [$0] } ?? []), writing: [ds, db] + (dx.map { [$0] } ?? [])) { arguments in
            for buffer in [x, g, factors, divisors, mu, dx ?? ds, ds, db] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.value(UInt32(dx != nil ? 1 : 0))
            arguments.dispatch(count: Int(parameters.columns))
        }
        inputTarget.take()?.finish(into: &gradients.input)
        if scale.requiresGradient {
            Tensor.accumulate(scaleColumns.reducingBroadcast(to: scale.shape), into: &gradients.scale)
        }
        if shift.requiresGradient {
            Tensor.accumulate(shiftColumns.reducingBroadcast(to: shift.shape), into: &gradients.shift)
        }
    }

    /// Broadcasts a tensor to the shape of the columns, or returns nil when its shape does not broadcast to it.
    private static func broadcastColumns<N: NumericType>(_ tensor: Tensor<N, GPU>, to columnShape: [Int]) -> Tensor<N, GPU>? {
        let tensor = tensor.detached()
        if tensor.shape == columnShape {
            return tensor
        }
        guard tensor.dim <= columnShape.count, zip(tensor.shape.reversed(), columnShape.reversed()).allSatisfy({ $0 == $1 || $0 == 1 }) else {
            return nil
        }
        return tensor + Tensor(repeating: 0, shape: columnShape)
    }

    /// The factors `scale / (sqrt(variance) + epsilon)`, their divisors, and the offsets `shift - mean * factor`, with the shape of the columns.
    private static func fixedNormalizationColumns<N: NumericType>(scale: Tensor<N, GPU>, shift: Tensor<N, GPU>, mean: Tensor<N, GPU>, variance: Tensor<N, GPU>, columnShape: [Int], epsilon: N) -> (factors: Tensor<N, GPU>, inverseDivisors: Tensor<N, GPU>, offsets: Tensor<N, GPU>)? {
        guard let gamma = broadcastColumns(scale, to: columnShape),
              let beta = broadcastColumns(shift, to: columnShape),
              let means = broadcastColumns(mean, to: columnShape),
              let variances = broadcastColumns(variance, to: columnShape)
        else {
            return nil
        }
        let inverseDivisors = 1 / (variances.sqrt() + Tensor(epsilon))
        let factors = gamma * inverseDivisors
        return (factors, inverseDivisors, beta - means * factors)
    }

    /// Number of elements of a row of a layer normalization, or nil when the kernels do not support the shapes.
    private static func layerNormalizationRowLength<N>(input: Tensor<N, GPU>, scale: Tensor<N, GPU>, shift: Tensor<N, GPU>) -> Int? {
        guard scale.dim >= 1, scale.shape == shift.shape, scale.dim <= input.dim, Array(input.shape.suffix(scale.dim)) == scale.shape, scale.count > 0 else {
            return nil
        }
        return scale.count
    }

    private static func rowForward<N>(_ name: String, input: Tensor<N, GPU>) -> Tensor<N, GPU> {
        let length = input.shape[input.dim - 1]
        let result: Tensor<N, GPU> = GPUFused.makeTensor(shape: input.shape)
        let (x, y) = (input.gpuBuffer, result.gpuBuffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: 0, epsilon: 0)
        let pipeline = GPUKernels.pipeline(name, in: .fused)
        GPUContext.current.compute(pipeline, reading: [x], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: input.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
        return result
    }

    private static func rowBackward<N>(_ name: String, output: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, accumulating gradient: inout Tensor<N, GPU>?) {
        let length = output.shape[output.dim - 1]
        let target = GPUGradientTarget(taking: &gradient, shape: output.shape)
        let (y, g, dx) = (output.gpuBuffer, outputGradient.gpuBuffer, target.buffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: target.accumulates ? 1 : 0, epsilon: 0)
        let pipeline = GPUKernels.pipeline(name, in: .fused)
        GPUContext.current.compute(pipeline, reading: [y, g, dx], writing: [dx]) { arguments in
            arguments.buffer(y)
            arguments.buffer(g)
            arguments.buffer(dx)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: output.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
        target.finish(into: &gradient)
    }
}

private struct ColumnParameters {
    var rows: UInt32
    var columns: UInt32
    var accumulate: UInt32
    var epsilon: Float

    init(input: Tensor<some NumericType, GPU>, accumulate: Bool, epsilon: some NumericType) {
        rows = UInt32(input.shape[0])
        columns = UInt32(input.count / Swift.max(input.shape[0], 1))
        self.accumulate = accumulate ? 1 : 0
        self.epsilon = epsilon.floatValue
    }
}
#endif

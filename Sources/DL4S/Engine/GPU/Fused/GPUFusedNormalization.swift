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
    static func softmax<N: NumericType>(input: ShapedBuffer<N, GPU>, axis: Int, result: MutableShapedBuffer<N, GPU>) {
        // The kernel supports the last axis.
        guard axis == input.dim - 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer], writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.softmax(input: input, axis: axis, result: result)
            return
        }
        rowForward("softmax_forward", input: input, result: result)
    }

    static func softmaxBackward<N: NumericType>(output: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, axis: Int, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        // The kernel supports the last axis.
        guard axis == output.dim - 1, GPUFused.runsKernel(N.self, elements: output.count, reading: [output.gpuBuffer, outputGradient.gpuBuffer], writing: [inputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.softmaxBackward(output: output, outputGradient: outputGradient, axis: axis, inputGradient: inputGradient)
            return
        }
        rowBackward("softmax_backward", output: output, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func logSoftmax<N: NumericType>(input: ShapedBuffer<N, GPU>, axis: Int, result: MutableShapedBuffer<N, GPU>) {
        // The kernel supports the last axis.
        guard axis == input.dim - 1, GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer], writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.logSoftmax(input: input, axis: axis, result: result)
            return
        }
        rowForward("log_softmax_forward", input: input, result: result)
    }

    static func logSoftmaxBackward<N: NumericType>(output: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, axis: Int, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        // The kernel supports the last axis.
        guard axis == output.dim - 1, GPUFused.runsKernel(N.self, elements: output.count, reading: [output.gpuBuffer, outputGradient.gpuBuffer], writing: [inputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.logSoftmaxBackward(output: output, outputGradient: outputGradient, axis: axis, inputGradient: inputGradient)
            return
        }
        rowBackward("log_softmax_backward", output: output, outputGradient: outputGradient, inputGradient: inputGradient)
    }

    static func layerNormalization<N: NumericType>(input: ShapedBuffer<N, GPU>, scale: ShapedBuffer<N, GPU>, shift: ShapedBuffer<N, GPU>, epsilon: N, result: MutableShapedBuffer<N, GPU>) {
        checkLayerNormalizationShapes(input: input, scale: scale, shift: shift)
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, shift.gpuBuffer], writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.layerNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon, result: result)
            return
        }
        let length = scale.count
        let (x, gamma, beta, y) = (input.gpuBuffer, scale.gpuBuffer, shift.gpuBuffer, result.gpuBuffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: 0, epsilon: epsilon.floatValue)
        let kernel = GPUKernels.kernel("layer_norm_forward", in: .fused)
        GPUContext.compute(kernel, reading: [x, gamma, beta], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(gamma)
            arguments.buffer(beta)
            arguments.buffer(y)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: input.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
    }

    static func layerNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        scale: ShapedBuffer<N, GPU>,
        shift: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, GPU>?,
        scaleGradient: GradientBuffer<N, GPU>?,
        shiftGradient: GradientBuffer<N, GPU>?,
    ) {
        checkLayerNormalizationShapes(input: input, scale: scale, shift: shift)
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, outputGradient.gpuBuffer], writing: [inputGradient, scaleGradient, shiftGradient].compactMap { $0?.gpuBuffer }) else {
            DefaultFusedOperations<GPU>.layerNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon, inputGradient: inputGradient, scaleGradient: scaleGradient, shiftGradient: shiftGradient)
            return
        }
        let length = scale.count
        let rows = input.count / length
        let (x, gamma, g) = (input.gpuBuffer, scale.gpuBuffer, outputGradient.gpuBuffer)
        let inputBuffer = inputGradient?.gpuBuffer
        // The kernel writes the terms of the scale gradient of every row, which a column reduction adds up.
        let scaleTerms = scaleGradient != nil ? GPUKernels.temporary(count: input.count, near: x) : nil
        let writing = [inputBuffer, scaleTerms].compactMap { $0 }
        if !writing.isEmpty {
            let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: inputGradient?.accumulateFlag ?? 0, epsilon: epsilon.floatValue)
            let outputs = SIMD2<UInt32>(inputBuffer != nil ? 1 : 0, scaleTerms != nil ? 1 : 0)
            let kernel = GPUKernels.kernel("layer_norm_backward", in: .fused)
            GPUContext.compute(kernel, reading: [x, gamma, g] + writing, writing: writing) { arguments in
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
        if let scaleTerms, let scaleGradient {
            GPUKernels.reduce(.sum, .float, values: scaleTerms, result: scaleGradient.gpuBuffer, outer: 1, length: rows, inner: length, accumulate: scaleGradient.adds)
        }
        if let shiftGradient {
            GPUKernels.reduce(.sum, .float, values: g, result: shiftGradient.gpuBuffer, outer: 1, length: rows, inner: length, accumulate: shiftGradient.adds)
        }
    }

    static func batchNormalization<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        scale: ShapedBuffer<N, GPU>,
        shift: ShapedBuffer<N, GPU>,
        epsilon: N,
        result: MutableShapedBuffer<N, GPU>,
        mean: MutableShapedBuffer<N, GPU>,
        variance: MutableShapedBuffer<N, GPU>,
    ) {
        precondition(input.dim >= 1, "The input must have a batch axis.")
        let columnShape = Array(input.shape.dropFirst())
        precondition(mean.shape == columnShape && variance.shape == columnShape, "The mean and the variance must have the shape of the input without the batch axis.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, shift.gpuBuffer], writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.batchNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon, result: result, mean: mean, variance: variance)
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let (gamma, beta) = (math.columns(of: scale, shape: columnShape), math.columns(of: shift, shape: columnShape))
        let parameters = ColumnParameters(input: input, accumulate: false, epsilon: epsilon)
        let (x, g, b, y, m, v) = (input.gpuBuffer, gamma.gpuBuffer, beta.gpuBuffer, result.gpuBuffer, mean.gpuBuffer, variance.gpuBuffer)
        let kernel = GPUKernels.kernel("batch_norm_forward", in: .fused)
        GPUContext.compute(kernel, reading: [x, g, b], writing: [y, m, v]) { arguments in
            for buffer in [x, g, b, y, m, v] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.dispatch(count: Int(parameters.columns))
        }
    }

    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        scale: ShapedBuffer<N, GPU>,
        shift: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, GPU>?,
        scaleGradient: GradientBuffer<N, GPU>?,
        shiftGradient: GradientBuffer<N, GPU>?,
    ) {
        precondition(input.dim >= 1, "The input must have a batch axis.")
        let columnShape = Array(input.shape.dropFirst())
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        precondition(ShapeUtil.broadcasts(shift.shape, to: columnShape), "The shift must be broadcastable to the shape of the input without the batch axis.")
        guard inputGradient != nil || scaleGradient != nil || shiftGradient != nil else {
            return
        }
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, outputGradient.gpuBuffer], writing: [inputGradient, scaleGradient, shiftGradient].compactMap { $0?.gpuBuffer }) else {
            DefaultFusedOperations<GPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon, inputGradient: inputGradient, scaleGradient: scaleGradient, shiftGradient: shiftGradient)
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let gamma = math.columns(of: scale, shape: columnShape)
        let parameters = ColumnParameters(input: input, accumulate: inputGradient?.adds ?? false, epsilon: epsilon)
        let (scaleColumns, shiftColumns) = (math.temporary(columnShape), math.temporary(columnShape))
        let (x, w, g, dx, ds, db) = (input.gpuBuffer, gamma.gpuBuffer, outputGradient.gpuBuffer, inputGradient?.gpuBuffer, scaleColumns.gpuBuffer, shiftColumns.gpuBuffer)
        let kernel = GPUKernels.kernel("batch_norm_backward", in: .fused)
        GPUContext.compute(kernel, reading: [x, w, g] + (dx.map { [$0] } ?? []), writing: [ds, db] + (dx.map { [$0] } ?? [])) { arguments in
            for buffer in [x, w, g, dx ?? ds, ds, db] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.value(UInt32(dx != nil ? 1 : 0))
            arguments.dispatch(count: Int(parameters.columns))
        }
        math.writeSum(of: scaleColumns, into: scaleGradient)
        math.writeSum(of: shiftColumns, into: shiftGradient)
    }

    static func batchNormalization<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        scale: ShapedBuffer<N, GPU>,
        shift: ShapedBuffer<N, GPU>,
        mean: ShapedBuffer<N, GPU>,
        variance: ShapedBuffer<N, GPU>,
        epsilon: N,
        result: MutableShapedBuffer<N, GPU>,
    ) {
        precondition(input.dim >= 1, "The input must have a batch axis.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, shift.gpuBuffer, mean.gpuBuffer, variance.gpuBuffer], writing: [result.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.batchNormalization(input: input, scale: scale, shift: shift, mean: mean, variance: variance, epsilon: epsilon, result: result)
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let affine = GPUFixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: Array(input.shape.dropFirst()), epsilon: epsilon, math: math)
        let parameters = ColumnParameters(input: input, accumulate: false, epsilon: epsilon)
        let (x, factors, offsets, y) = (input.gpuBuffer, affine.factors.gpuBuffer, affine.offsets.gpuBuffer, result.gpuBuffer)
        let kernel = GPUKernels.kernel("affine_columns", in: .fused)
        GPUContext.compute(kernel, reading: [x, factors, offsets], writing: [y]) { arguments in
            for buffer in [x, factors, offsets, y] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
    }

    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        scale: ShapedBuffer<N, GPU>,
        shift: ShapedBuffer<N, GPU>,
        mean: ShapedBuffer<N, GPU>,
        variance: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, GPU>?,
        scaleGradient: GradientBuffer<N, GPU>?,
        shiftGradient: GradientBuffer<N, GPU>?,
    ) {
        precondition(input.dim >= 1, "The input must have a batch axis.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        guard inputGradient != nil || scaleGradient != nil || shiftGradient != nil else {
            return
        }
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, scale.gpuBuffer, mean.gpuBuffer, variance.gpuBuffer, outputGradient.gpuBuffer], writing: [inputGradient, scaleGradient, shiftGradient].compactMap { $0?.gpuBuffer }) else {
            DefaultFusedOperations<GPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, mean: mean, variance: variance, outputGradient: outputGradient, epsilon: epsilon, inputGradient: inputGradient, scaleGradient: scaleGradient, shiftGradient: shiftGradient)
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        let columnShape = Array(input.shape.dropFirst())
        let affine = GPUFixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon, math: math)
        let parameters = ColumnParameters(input: input, accumulate: inputGradient?.adds ?? false, epsilon: epsilon)
        let (scaleColumns, shiftColumns) = (math.temporary(columnShape), math.temporary(columnShape))
        let (x, g, factors, divisors, mu) = (input.gpuBuffer, outputGradient.gpuBuffer, affine.factors.gpuBuffer, affine.inverseDivisors.gpuBuffer, affine.means.gpuBuffer)
        let (dx, ds, db) = (inputGradient?.gpuBuffer, scaleColumns.gpuBuffer, shiftColumns.gpuBuffer)
        let kernel = GPUKernels.kernel("batch_norm_fixed_backward", in: .fused)
        GPUContext.compute(kernel, reading: [x, g, factors, divisors, mu] + (dx.map { [$0] } ?? []), writing: [ds, db] + (dx.map { [$0] } ?? [])) { arguments in
            for buffer in [x, g, factors, divisors, mu, dx ?? ds, ds, db] {
                arguments.buffer(buffer)
            }
            arguments.value(parameters)
            arguments.value(UInt32(dx != nil ? 1 : 0))
            arguments.dispatch(count: Int(parameters.columns))
        }
        math.writeSum(of: scaleColumns, into: scaleGradient)
        math.writeSum(of: shiftColumns, into: shiftGradient)
    }

    private static func rowForward<N>(_ name: String, input: ShapedBuffer<N, GPU>, result: MutableShapedBuffer<N, GPU>) {
        let length = input.shape[input.dim - 1]
        let (x, y) = (input.gpuBuffer, result.gpuBuffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: 0, epsilon: 0)
        let kernel = GPUKernels.kernel(name, in: .fused)
        GPUContext.compute(kernel, reading: [x], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: input.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
    }

    private static func rowBackward<N>(_ name: String, output: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, inputGradient: GradientBuffer<N, GPU>) {
        let length = output.shape[output.dim - 1]
        let (y, g, dx) = (output.gpuBuffer, outputGradient.gpuBuffer, inputGradient.gpuBuffer)
        let parameters = GPUFused.RowParameters(length: UInt32(length), accumulate: inputGradient.accumulateFlag, epsilon: 0)
        let kernel = GPUKernels.kernel(name, in: .fused)
        GPUContext.compute(kernel, reading: [y, g, dx], writing: [dx]) { arguments in
            arguments.buffer(y)
            arguments.buffer(g)
            arguments.buffer(dx)
            arguments.value(parameters)
            arguments.dispatch(threadgroups: MTLSize(width: output.count / length, height: 1, depth: 1), threadgroup: MTLSize(width: GPUFused.rowThreadgroupWidth(length: length), height: 1, depth: 1))
        }
    }
}

/// The columns of a normalization with fixed statistics, in intermediate buffers: the factors `scale / (sqrt(variance) + epsilon)`,
/// the inverse divisors `1 / (sqrt(variance) + epsilon)`, the offsets `shift - mean * factor`, and the means.
private struct GPUFixedNormalizationColumns<N: NumericType> {
    let factors: ShapedBuffer<N, GPU>
    let inverseDivisors: ShapedBuffer<N, GPU>
    let offsets: ShapedBuffer<N, GPU>
    let means: ShapedBuffer<N, GPU>

    /// Computes the columns in intermediate buffers of `math`. The parameters must be broadcastable to the columns.
    init(scale: ShapedBuffer<N, GPU>, shift: ShapedBuffer<N, GPU>, mean: ShapedBuffer<N, GPU>, variance: ShapedBuffer<N, GPU>, columnShape: [Int], epsilon: N, math: BufferMath<N, GPU>) {
        let (gamma, beta) = (math.columns(of: scale, shape: columnShape), math.columns(of: shift, shape: columnShape))
        let (means, variances) = (math.columns(of: mean, shape: columnShape), math.columns(of: variance, shape: columnShape))
        let (inverseDivisors, factors, offsets) = (math.temporary(columnShape), math.temporary(columnShape), math.temporary(columnShape))
        math.sqrt(variances, into: inverseDivisors)
        math.add(inverseDivisors, epsilon, into: inverseDivisors)
        math.divide(math.constant(1), inverseDivisors, into: inverseDivisors)
        math.multiply(gamma, inverseDivisors, into: factors)
        math.multiply(means, factors, into: offsets)
        math.subtract(beta, offsets, into: offsets)
        self.factors = ShapedBuffer(factors)
        self.inverseDivisors = ShapedBuffer(inverseDivisors)
        self.offsets = ShapedBuffer(offsets)
        self.means = means
    }
}

private extension BufferMath where Device == GPU {
    /// Values that are broadcastable to the columns, repeated into an intermediate buffer when the shapes differ.
    func columns(of values: ShapedBuffer<N, GPU>, shape columnShape: [Int]) -> ShapedBuffer<N, GPU> {
        if values.shape == columnShape {
            return values
        }
        precondition(ShapeUtil.broadcasts(values.shape, to: columnShape), "The parameters must be broadcastable to the shape of the input without the batch axis.")
        let repeated = temporary(columnShape)
        add(values, constant(0, shape: columnShape), into: repeated)
        return ShapedBuffer(repeated)
    }
}

private struct ColumnParameters {
    var rows: UInt32
    var columns: UInt32
    var accumulate: UInt32
    var epsilon: Float

    init(input: ShapedBuffer<some NumericType, GPU>, accumulate: Bool, epsilon: some NumericType) {
        rows = UInt32(input.shape[0])
        columns = UInt32(input.count / Swift.max(input.shape[0], 1))
        self.accumulate = accumulate ? 1 : 0
        self.epsilon = epsilon.floatValue
    }
}
#endif

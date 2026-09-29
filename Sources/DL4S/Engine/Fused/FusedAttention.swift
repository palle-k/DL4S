//
//  FusedAttention.swift
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
    static func scaledDotProductAttention<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, temperature: N, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let weights = attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature, math: math)
        math.multiplyBatchedMatrices(weights, values, into: result)
    }

    static func scaledDotProductAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        mask: ShapedBuffer<N, Device>?,
        outputGradient: ShapedBuffer<N, Device>,
        temperature: N,
        queryGradient: GradientBuffer<N, Device>?,
        keyGradient: GradientBuffer<N, Device>?,
        valueGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
        let weights = attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature, math: math)
        math.writeBatchedProduct(weights, outputGradient, lhsTransposed: true, into: valueGradient)
        guard queryGradient != nil || keyGradient != nil else {
            return
        }
        // The gradient of the scores is the gradient of the softmax. The products with the keys and the queries divide it by the temperature.
        let weightGradient = Device.Memory.allocateBuffer(withShape: weights.shape, type: N.self)
        math.multiplyBatchedMatrices(outputGradient, values, rhsTransposed: true, into: weightGradient)
        let scoreGradient = math.temporary(weights.shape)
        softmaxBackward(output: ShapedBuffer(weights), outputGradient: ShapedBuffer(weightGradient), axis: 3, inputGradient: GradientBuffer(values: scoreGradient, adds: false))
        Device.Memory.free(weightGradient)
        math.writeBatchedProduct(scoreGradient, keys, alpha: 1 / temperature, into: queryGradient)
        math.writeBatchedProduct(scoreGradient, queries, lhsTransposed: true, alpha: 1 / temperature, into: keyGradient)
    }

    static func multiHeadAttention<N: NumericType>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        mask: ShapedBuffer<N, Device>?,
        queryWeights: ShapedBuffer<N, Device>,
        keyWeights: ShapedBuffer<N, Device>,
        valueWeights: ShapedBuffer<N, Device>,
        outputWeights: ShapedBuffer<N, Device>,
        heads: Int,
        temperature: N,
        result: MutableShapedBuffer<N, Device>,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let queryHeads = projectedHeads(queries, weights: queryWeights, heads: heads, math: math)
        let keyHeads = projectedHeads(keys, weights: keyWeights, heads: heads, math: math)
        let valueHeads = projectedHeads(values, weights: valueWeights, heads: heads, math: math)
        let attended = Device.Memory.allocateBuffer(withShape: [queryHeads.shape[0], heads, queryHeads.shape[2], valueHeads.shape[3]], type: N.self)
        scaledDotProductAttention(queries: ShapedBuffer(queryHeads), keys: ShapedBuffer(keyHeads), values: ShapedBuffer(valueHeads), mask: mask, temperature: temperature, result: attended)
        [queryHeads, keyHeads, valueHeads].forEach(Device.Memory.free)
        let joined = joinedHeads(attended, math: math)
        Device.Memory.free(attended)
        math.multiplyMatrices(joined, outputWeights, into: result.reshaped(to: [joined.shape[0], outputWeights.shape[1]]))
        Device.Memory.free(joined)
    }

    static func multiHeadAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        mask: ShapedBuffer<N, Device>?,
        queryWeights: ShapedBuffer<N, Device>,
        keyWeights: ShapedBuffer<N, Device>,
        valueWeights: ShapedBuffer<N, Device>,
        outputWeights: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        heads: Int,
        temperature: N,
        gradients: MultiHeadAttentionGradients<GradientBuffer<N, Device>?>,
    ) {
        projectedAttentionBackward(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            outputGradient: outputGradient,
            heads: heads,
            gradients: gradients,
        ) { queries, keys, values, outputGradient, output, queryGradient, keyGradient, valueGradient in
            // The attention of the device does not return its result from the backward pass, so the result is computed separately.
            if let output {
                scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: output)
            }
            scaledDotProductAttentionBackward(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                outputGradient: outputGradient,
                temperature: temperature,
                queryGradient: queryGradient,
                keyGradient: keyGradient,
                valueGradient: valueGradient,
            )
        }
    }

    static func positionalEncoding<N: NumericType>(length: Int, hiddenSize: Int, result: MutableShapedBuffer<N, Device>) {
        // The encoding is a constant, so it is computed on the host and copied to the device once.
        var encoding = [N](repeating: 0, count: length * hiddenSize)
        for position in 0 ..< length {
            for index in 0 ..< hiddenSize / 2 {
                let angle = Double(position) / Foundation.pow(10000, Double(index) / Double(hiddenSize / 2))
                encoding[position * hiddenSize + 2 * index] = N(Foundation.sin(angle))
                encoding[position * hiddenSize + 2 * index + 1] = N(Foundation.cos(angle))
            }
        }
        encoding.withUnsafeBufferPointer { elements in
            Device.Memory.assign(from: elements, to: result.values, count: elements.count)
        }
    }
}

/// Computes the gradients of the attention of the heads of multi-head attention, shape [batchSize, heads, count, size].
///
/// The closure receives the queries, keys, and values of the heads, the gradient of the result of the attention, a buffer for
/// the result of the attention, or nil when it is not needed, and the buffers of the requested gradients of the heads.
typealias HeadAttentionBackward<N: NumericType, Device: DeviceType> = (
    _ queries: ShapedBuffer<N, Device>,
    _ keys: ShapedBuffer<N, Device>,
    _ values: ShapedBuffer<N, Device>,
    _ outputGradient: ShapedBuffer<N, Device>,
    _ output: MutableShapedBuffer<N, Device>?,
    _ queryGradient: GradientBuffer<N, Device>?,
    _ keyGradient: GradientBuffer<N, Device>?,
    _ valueGradient: GradientBuffer<N, Device>?,
) -> Void

extension FusedOperationsType {
    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` along the last axis, shape [batchSize, heads, queryCount, keyCount].
    ///
    /// The softmax and its gradient in the attention defaults are the requirements of `Self`, so that a device that does not
    /// override attention runs its softmax kernels.
    static func attentionWeights<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, temperature: N, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let shape = ShapeUtil.batchedProductShape(queries.shape, keys.shape, lhsTransposed: false, rhsTransposed: true)
        let scores = Device.Memory.allocateBuffer(withShape: shape, type: N.self)
        defer {
            Device.Memory.free(scores)
        }
        math.multiplyBatchedMatrices(queries, keys, rhsTransposed: true, into: scores, alpha: 1 / temperature)
        if let mask {
            // The mask contains 1 for every entry that is blocked, so the softmax sets these entries to 0.
            let blocked = math.temporary(mask.shape)
            math.multiply(mask, N(1e9), into: blocked)
            math.subtract(scores, blocked, into: scores)
        }
        let weights = math.temporary(shape)
        softmax(input: ShapedBuffer(scores), axis: shape.count - 1, result: weights)
        return weights
    }

    // The helpers of multi-head attention allocate their results with the memory operators, and the caller frees every
    // intermediate after its last use, so that the next allocations reuse memory that is still in the cache.

    /// Projects a [batchSize, count, inputSize] buffer with weights of the shape [inputSize, heads \* size] and splits the
    /// result into heads, shape [batchSize, heads, count, size]. The caller frees the result.
    static func projectedHeads<N: NumericType>(_ input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, heads: Int, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let (batchSize, count) = (input.shape[0], input.shape[1])
        let size = weights.shape[1] / heads
        let projected = Device.Memory.allocateBuffer(withShape: [batchSize, count, heads, size], type: N.self)
        defer {
            Device.Memory.free(projected)
        }
        math.multiplyMatrices(input.reshaped(to: [batchSize * count, input.shape[2]]), weights, into: projected.reshaped(to: [batchSize * count, weights.shape[1]]))
        let split = Device.Memory.allocateBuffer(withShape: [batchSize, heads, count, size], type: N.self)
        math.permute(projected, to: [0, 2, 1, 3], into: split)
        return split
    }

    /// Joins [batchSize, heads, count, size] into a [batchSize \* count, heads \* size] matrix. The caller frees the result.
    static func joinedHeads<N: NumericType>(_ input: some ReadableBuffer<N, Device>, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let input = input.readable
        let (batchSize, heads, count, size) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        let joined = Device.Memory.allocateBuffer(withShape: [batchSize, count, heads, size], type: N.self)
        math.permute(input, to: [0, 2, 1, 3], into: joined)
        return joined.reshaped(to: [batchSize * count, heads * size])
    }

    /// Computes the gradients of multi-head attention, where `headAttentionBackward` computes the gradients of the attention of
    /// the heads, and its result when the output projection has a requested gradient.
    static func projectedAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        queryWeights: ShapedBuffer<N, Device>,
        keyWeights: ShapedBuffer<N, Device>,
        valueWeights: ShapedBuffer<N, Device>,
        outputWeights: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        heads: Int,
        gradients: MultiHeadAttentionGradients<GradientBuffer<N, Device>?>,
        headAttentionBackward: HeadAttentionBackward<N, Device>,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let queryHeads = projectedHeads(queries, weights: queryWeights, heads: heads, math: math)
        let keyHeads = projectedHeads(keys, weights: keyWeights, heads: heads, math: math)
        let valueHeads = projectedHeads(values, weights: valueWeights, heads: heads, math: math)
        let (batchSize, queryCount) = (queries.shape[0], queries.shape[1])
        let gradientMatrix = outputGradient.reshaped(to: [batchSize * queryCount, outputWeights.shape[1]])

        // The gradient of the attention of the heads, [batchSize, heads, queryCount, valueSize]
        let joinedGradient = Device.Memory.allocateBuffer(withShape: [batchSize, queryCount, heads, valueHeads.shape[3]], type: N.self)
        math.multiplyMatrices(gradientMatrix, outputWeights, rhsTransposed: true, into: joinedGradient.reshaped(to: [batchSize * queryCount, outputWeights.shape[0]]))
        let headOutputGradient = Device.Memory.allocateBuffer(withShape: [batchSize, heads, queryCount, valueHeads.shape[3]], type: N.self)
        math.permute(joinedGradient, to: [0, 2, 1, 3], into: headOutputGradient)
        Device.Memory.free(joinedGradient)

        func headGradient(_ heads: MutableShapedBuffer<N, Device>, _ inputGradient: GradientBuffer<N, Device>?, _ weightGradient: GradientBuffer<N, Device>?) -> GradientBuffer<N, Device>? {
            inputGradient == nil && weightGradient == nil ? nil : GradientBuffer(values: Device.Memory.allocateBuffer(withShape: heads.shape, type: N.self), adds: false)
        }
        let queryHeadGradient = headGradient(queryHeads, gradients.queries, gradients.queryWeights)
        let keyHeadGradient = headGradient(keyHeads, gradients.keys, gradients.keyWeights)
        let valueHeadGradient = headGradient(valueHeads, gradients.values, gradients.valueWeights)
        let attended = gradients.outputWeights == nil ? nil : Device.Memory.allocateBuffer(withShape: headOutputGradient.shape, type: N.self)
        headAttentionBackward(
            ShapedBuffer(queryHeads),
            ShapedBuffer(keyHeads),
            ShapedBuffer(valueHeads),
            ShapedBuffer(headOutputGradient),
            attended,
            queryHeadGradient,
            keyHeadGradient,
            valueHeadGradient,
        )
        [queryHeads, keyHeads, valueHeads, headOutputGradient].forEach(Device.Memory.free)
        if let attended, let outputWeightGradient = gradients.outputWeights {
            let joined = joinedHeads(attended, math: math)
            Device.Memory.free(attended)
            math.multiplyMatrices(joined, gradientMatrix, lhsTransposed: true, into: outputWeightGradient.values, beta: outputWeightGradient.beta)
            Device.Memory.free(joined)
        }
        func addProjectionGradients(input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, headGradient: GradientBuffer<N, Device>?, inputGradient: GradientBuffer<N, Device>?, weightGradient: GradientBuffer<N, Device>?) {
            guard let headGradient else {
                return
            }
            let joined = joinedHeads(headGradient.values, math: math)
            Device.Memory.free(headGradient.values)
            let inputMatrixShape = [input.shape[0] * input.shape[1], input.shape[2]]
            if let inputGradient {
                math.multiplyMatrices(joined, weights, rhsTransposed: true, into: inputGradient.values.reshaped(to: inputMatrixShape), beta: inputGradient.beta)
            }
            if let weightGradient {
                math.multiplyMatrices(input.reshaped(to: inputMatrixShape), joined, lhsTransposed: true, into: weightGradient.values, beta: weightGradient.beta)
            }
            Device.Memory.free(joined)
        }
        addProjectionGradients(input: queries, weights: queryWeights, headGradient: queryHeadGradient, inputGradient: gradients.queries, weightGradient: gradients.queryWeights)
        addProjectionGradients(input: keys, weights: keyWeights, headGradient: keyHeadGradient, inputGradient: gradients.keys, weightGradient: gradients.keyWeights)
        addProjectionGradients(input: values, weights: valueWeights, headGradient: valueHeadGradient, inputGradient: gradients.values, weightGradient: gradients.valueWeights)
    }
}

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
        matrixProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
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
        matrixProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
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
        projectedAttention(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            heads: heads,
            layout: .split,
            result: result,
        ) { queries, keys, values, result in
            scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
        }
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
            layout: .split,
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

/// Layout of the heads that multi-head attention passes to the attention of the heads.
enum HeadsLayout {
    /// [batchSize, heads, count, size]: the heads are split from the projections with a permutation
    case split
    /// [batchSize, count, heads, size]: the layout of the products of the projections, in which the attention reads the
    /// heads with strides
    case interleaved

    func shape(batchSize: Int, count: Int, heads: Int, size: Int) -> [Int] {
        switch self {
        case .split: [batchSize, heads, count, size]
        case .interleaved: [batchSize, count, heads, size]
        }
    }

    /// Arranges projected heads, [batchSize, count, heads, size], in the layout. Split heads are permuted into a new
    /// buffer, and the input is freed. The caller frees the result.
    func arranged<N: NumericType, Device>(consuming projected: MutableShapedBuffer<N, Device>, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        guard self == .split else {
            return projected
        }
        let (batchSize, count, heads, size) = (projected.shape[0], projected.shape[1], projected.shape[2], projected.shape[3])
        let split = Device.Memory.allocateBuffer(withShape: shape(batchSize: batchSize, count: count, heads: heads, size: size), type: N.self)
        math.permute(projected, to: [0, 2, 1, 3], into: split)
        Device.Memory.free(projected)
        return split
    }

    /// Joins heads in the layout into a [batchSize \* count, heads \* size] matrix. Split heads are permuted into a new
    /// buffer, and the input is freed. The caller frees the result.
    func joined<N: NumericType, Device>(consuming input: MutableShapedBuffer<N, Device>, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        switch self {
        case .interleaved:
            let (batchSize, count, heads, size) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
            return input.reshaped(to: [batchSize * count, heads * size])
        case .split:
            let (batchSize, heads, count, size) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
            let joined = Device.Memory.allocateBuffer(withShape: HeadsLayout.interleaved.shape(batchSize: batchSize, count: count, heads: heads, size: size), type: N.self)
            math.permute(input, to: [0, 2, 1, 3], into: joined)
            Device.Memory.free(input)
            return joined.reshaped(to: [batchSize * count, heads * size])
        }
    }
}

/// Computes the attention of the heads of multi-head attention into its result. The heads are in the layout of the caller,
/// and the keys and the values have a number of heads that divides the number of query heads.
typealias HeadAttention<N: NumericType, Device: DeviceType> = (
    _ queries: ShapedBuffer<N, Device>,
    _ keys: ShapedBuffer<N, Device>,
    _ values: ShapedBuffer<N, Device>,
    _ result: MutableShapedBuffer<N, Device>,
) -> Void

/// Computes the gradients of the attention of the heads of multi-head attention. The heads are in the layout of the caller,
/// and the keys and the values have a number of heads that divides the number of query heads.
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
    // The default implementations of attention are these helpers, so that a device calls them for inputs that its kernels
    // do not support. The helpers use the softmax requirements of `Self`, and then run the softmax kernels of the device,
    // which a fallback through `DefaultFusedOperations` does not.

    /// Computes scaled dot product attention with batched matrix products, which write the scores to memory, and the softmax
    /// requirements of `Self`. It is the default implementation of the requirement.
    static func matrixProductAttention<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, temperature: N, result: MutableShapedBuffer<N, Device>) {
        let shape = AttentionShape(queries: queries, keys: keys, values: values)
        precondition(result.shape == shape.resultShape, "The result must have the shape [batchSize, heads, queryCount, valueDim].")
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let weights = attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature, shape: shape, math: math)
        math.multiplyBatchedMatrices(shape.valueGrouped(weights), values, into: shape.valueGrouped(result))
    }

    /// Computes the gradients of scaled dot product attention with batched matrix products and the softmax of `Self`, see
    /// ``matrixProductAttention(queries:keys:values:mask:temperature:result:)``.
    static func matrixProductAttentionBackward<N: NumericType>(
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
        let shape = AttentionShape(queries: queries, keys: keys, values: values)
        precondition(outputGradient.shape == shape.resultShape, "The gradient of the result must have the shape of the result.")
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
        let weights = attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature, shape: shape, math: math)
        let groupedOutputGradient = shape.valueGrouped(outputGradient)
        math.writeBatchedProduct(shape.valueGrouped(weights), groupedOutputGradient, lhsTransposed: true, into: valueGradient)
        guard queryGradient != nil || keyGradient != nil else {
            return
        }
        // The gradient of the scores is the gradient of the softmax. The products with the keys and the queries divide it by the temperature.
        let weightGradient = Device.Memory.allocateBuffer(withShape: weights.shape, type: N.self)
        math.multiplyBatchedMatrices(groupedOutputGradient, values, rhsTransposed: true, into: shape.valueGrouped(weightGradient))
        let scoreGradient = math.temporary(weights.shape)
        softmaxBackward(output: ShapedBuffer(weights), outputGradient: ShapedBuffer(weightGradient), axis: 3, inputGradient: GradientBuffer(values: scoreGradient, adds: false))
        Device.Memory.free(weightGradient)
        let groupedScoreGradient = shape.keyGrouped(scoreGradient)
        math.writeBatchedProduct(groupedScoreGradient, keys, alpha: 1 / temperature, into: queryGradient.map(shape.keyGrouped))
        math.writeBatchedProduct(groupedScoreGradient, shape.keyGrouped(queries), lhsTransposed: true, alpha: 1 / temperature, into: keyGradient)
    }

    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` along the last axis with the softmax requirement of
    /// `Self`, shape [batchSize, heads, queryCount, keyCount].
    static func attentionWeights<N: NumericType>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, mask: ShapedBuffer<N, Device>?, temperature: N, shape attention: AttentionShape, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let shape = attention.scoreShape
        let scores = Device.Memory.allocateBuffer(withShape: shape, type: N.self)
        defer {
            Device.Memory.free(scores)
        }
        math.multiplyBatchedMatrices(attention.keyGrouped(queries), keys, rhsTransposed: true, into: attention.keyGrouped(scores), alpha: 1 / temperature)
        if let mask {
            // The mask contains 1 for every entry that is blocked, so the softmax sets these entries to 0.
            let blocked = Device.Memory.allocateBuffer(withShape: mask.shape, type: N.self)
            math.multiply(mask, N(1e9), into: blocked)
            math.subtract(scores, blocked, into: scores)
            Device.Memory.free(blocked)
        }
        let weights = math.temporary(shape)
        softmax(input: ShapedBuffer(scores), axis: shape.count - 1, result: weights)
        return weights
    }

    // The helpers of multi-head attention allocate their results with the memory operators, and the caller frees every
    // intermediate after its last use, so that the next allocations reuse memory that is still in the cache.

    /// Projects a [batchSize, count, inputSize] buffer with weights of the shape [inputSize, heads \* size] into heads in the
    /// given layout. The caller frees the result.
    static func projectedHeads<N: NumericType>(_ input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, heads: Int, layout: HeadsLayout, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let (batchSize, count) = (input.shape[0], input.shape[1])
        let projected = Device.Memory.allocateBuffer(withShape: [batchSize, count, heads, weights.shape[1] / heads], type: N.self)
        math.multiplyMatrices(input.reshaped(to: [batchSize * count, input.shape[2]]), weights, into: projected.reshaped(to: [batchSize * count, weights.shape[1]]))
        return layout.arranged(consuming: projected, math: math)
    }

    /// Computes multi-head attention, where `headAttention` computes the attention of the heads.
    static func projectedAttention<N: NumericType>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        queryWeights: ShapedBuffer<N, Device>,
        keyWeights: ShapedBuffer<N, Device>,
        valueWeights: ShapedBuffer<N, Device>,
        outputWeights: ShapedBuffer<N, Device>,
        heads: Int,
        layout: HeadsLayout,
        result: MutableShapedBuffer<N, Device>,
        headAttention: HeadAttention<N, Device>,
    ) {
        let shape = MultiHeadAttentionShape(queries: queries, keys: keys, values: values, queryWeights: queryWeights, keyWeights: keyWeights, valueWeights: valueWeights, outputWeights: outputWeights, heads: heads)
        precondition(result.shape == shape.resultShape, "The result must have the shape [batchSize, queryCount, outputDim].")
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let queryHeads = projectedHeads(queries, weights: queryWeights, heads: heads, layout: layout, math: math)
        let keyHeads = projectedHeads(keys, weights: keyWeights, heads: shape.keyHeads, layout: layout, math: math)
        let valueHeads = projectedHeads(values, weights: valueWeights, heads: shape.keyHeads, layout: layout, math: math)
        let attended = Device.Memory.allocateBuffer(withShape: layout.shape(batchSize: shape.batchSize, count: shape.queryCount, heads: heads, size: shape.valueDim), type: N.self)
        headAttention(ShapedBuffer(queryHeads), ShapedBuffer(keyHeads), ShapedBuffer(valueHeads), attended)
        [queryHeads, keyHeads, valueHeads].forEach(Device.Memory.free)
        let joined = layout.joined(consuming: attended, math: math)
        math.multiplyMatrices(joined, outputWeights, into: result.reshaped(to: [joined.shape[0], shape.outputDim]))
        Device.Memory.free(joined)
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
        layout: HeadsLayout,
        gradients: MultiHeadAttentionGradients<GradientBuffer<N, Device>?>,
        headAttentionBackward: HeadAttentionBackward<N, Device>,
    ) {
        let shape = MultiHeadAttentionShape(queries: queries, keys: keys, values: values, queryWeights: queryWeights, keyWeights: keyWeights, valueWeights: valueWeights, outputWeights: outputWeights, heads: heads)
        precondition(outputGradient.shape == shape.resultShape, "The gradient of the result must have the shape of the result.")
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let queryHeads = projectedHeads(queries, weights: queryWeights, heads: heads, layout: layout, math: math)
        let keyHeads = projectedHeads(keys, weights: keyWeights, heads: shape.keyHeads, layout: layout, math: math)
        let valueHeads = projectedHeads(values, weights: valueWeights, heads: shape.keyHeads, layout: layout, math: math)
        let gradientMatrix = outputGradient.reshaped(to: [shape.batchSize * shape.queryCount, shape.outputDim])

        // The gradient of the attention of the heads
        let joinedGradient = Device.Memory.allocateBuffer(withShape: HeadsLayout.interleaved.shape(batchSize: shape.batchSize, count: shape.queryCount, heads: heads, size: shape.valueDim), type: N.self)
        math.multiplyMatrices(gradientMatrix, outputWeights, rhsTransposed: true, into: joinedGradient.reshaped(to: [shape.batchSize * shape.queryCount, heads * shape.valueDim]))
        let headOutputGradient = layout.arranged(consuming: joinedGradient, math: math)

        func headGradient(_ projected: MutableShapedBuffer<N, Device>, _ inputGradient: GradientBuffer<N, Device>?, _ weightGradient: GradientBuffer<N, Device>?) -> GradientBuffer<N, Device>? {
            inputGradient == nil && weightGradient == nil ? nil : GradientBuffer(values: Device.Memory.allocateBuffer(withShape: projected.shape, type: N.self), adds: false)
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
            let joined = layout.joined(consuming: attended, math: math)
            math.multiplyMatrices(joined, gradientMatrix, lhsTransposed: true, into: outputWeightGradient.values, beta: outputWeightGradient.beta)
            Device.Memory.free(joined)
        }
        func addProjectionGradients(input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, headGradient: GradientBuffer<N, Device>?, inputGradient: GradientBuffer<N, Device>?, weightGradient: GradientBuffer<N, Device>?) {
            guard let headGradient else {
                return
            }
            let joined = layout.joined(consuming: headGradient.values, math: math)
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

/// Shapes of scaled dot product attention, see ``FusedOperationsType/scaledDotProductAttention(queries:keys:values:mask:temperature:result:)``.
///
/// The grouped views fold the query heads that share a key head or a value head into the rows of one matrix.
struct AttentionShape {
    // The queries of a group of query heads are consecutive in memory, so one product with the shared head computes the group.

    /// Batch size of the result
    let batchSize: Int
    /// Number of query heads, which is also the number of heads of the result
    let heads: Int
    /// Number of key heads, which divides the number of query heads
    let keyHeads: Int
    /// Number of value heads, which divides the number of query heads
    let valueHeads: Int
    let queryCount: Int
    let keyCount: Int
    let keyDim: Int
    let valueDim: Int
    /// Batch sizes of the queries, the keys, and the values. Each is 1 or the batch size of the result.
    let queryBatchSize: Int
    let keyBatchSize: Int
    let valueBatchSize: Int

    /// Shapes of the attention of the heads of multi-head attention: the queries, keys, and values have the same batch size,
    /// and the values have as many heads as the keys.
    fileprivate init(heads shape: MultiHeadAttentionShape) {
        (batchSize, queryBatchSize, keyBatchSize, valueBatchSize) = (shape.batchSize, shape.batchSize, shape.batchSize, shape.batchSize)
        (heads, keyHeads, valueHeads) = (shape.heads, shape.keyHeads, shape.keyHeads)
        (queryCount, keyCount, keyDim, valueDim) = (shape.queryCount, shape.keyCount, shape.keyDim, shape.valueDim)
    }

    /// Checks the shapes of the operands of scaled dot product attention.
    init<N, Device>(queries: ShapedBuffer<N, Device>, keys: ShapedBuffer<N, Device>, values: ShapedBuffer<N, Device>) {
        precondition(queries.dim == 4 && keys.dim == 4 && values.dim == 4, "The queries, keys, and values must have 4 axes.")
        precondition(queries.shape[3] == keys.shape[3], "The queries and the keys must have the same size.")
        precondition(keys.shape[2] == values.shape[2], "There must be one value for every key.")
        let batchSizes = Set([queries.shape[0], keys.shape[0], values.shape[0]])
        precondition(batchSizes.subtracting([1]).count <= 1, "The batch axes of the queries, keys, and values must be broadcastable.")
        (queryBatchSize, keyBatchSize, valueBatchSize) = (queries.shape[0], keys.shape[0], values.shape[0])
        batchSize = batchSizes.max()!
        (heads, keyHeads, valueHeads) = (queries.shape[1], keys.shape[1], values.shape[1])
        precondition(heads.isMultiple(of: keyHeads) && heads.isMultiple(of: valueHeads), "The numbers of key heads and value heads must divide the number of query heads.")
        (queryCount, keyCount, keyDim, valueDim) = (queries.shape[2], keys.shape[2], keys.shape[3], values.shape[3])
    }

    /// Shape of the result, [batchSize, heads, queryCount, valueDim]
    var resultShape: [Int] {
        [batchSize, heads, queryCount, valueDim]
    }

    /// Shape of the scores and the attention weights, [batchSize, heads, queryCount, keyCount]
    var scoreShape: [Int] {
        [batchSize, heads, queryCount, keyCount]
    }

    /// Folds the query heads of every key head into the rows: [n, heads, rows, columns] becomes [n, keyHeads, heads / keyHeads \* rows, columns].
    func keyGrouped<N, Device>(_ buffer: ShapedBuffer<N, Device>) -> ShapedBuffer<N, Device> {
        buffer.reshaped(to: Self.grouped(buffer.shape, heads: keyHeads))
    }

    func keyGrouped<N, Device>(_ buffer: MutableShapedBuffer<N, Device>) -> MutableShapedBuffer<N, Device> {
        buffer.reshaped(to: Self.grouped(buffer.shape, heads: keyHeads))
    }

    func keyGrouped<N, Device>(_ gradient: GradientBuffer<N, Device>) -> GradientBuffer<N, Device> {
        GradientBuffer(values: keyGrouped(gradient.values), adds: gradient.adds)
    }

    /// Folds the query heads of every value head into the rows: [n, heads, rows, columns] becomes [n, valueHeads, heads / valueHeads \* rows, columns].
    func valueGrouped<N, Device>(_ buffer: ShapedBuffer<N, Device>) -> ShapedBuffer<N, Device> {
        buffer.reshaped(to: Self.grouped(buffer.shape, heads: valueHeads))
    }

    func valueGrouped<N, Device>(_ buffer: MutableShapedBuffer<N, Device>) -> MutableShapedBuffer<N, Device> {
        buffer.reshaped(to: Self.grouped(buffer.shape, heads: valueHeads))
    }

    private static func grouped(_ shape: [Int], heads: Int) -> [Int] {
        [shape[0], heads, shape[1] / heads * shape[2], shape[3]]
    }
}

/// Shapes of multi-head attention, see ``FusedOperationsType/multiHeadAttention(queries:keys:values:mask:queryWeights:keyWeights:valueWeights:outputWeights:heads:temperature:result:)``.
struct MultiHeadAttentionShape {
    let batchSize: Int
    let queryCount: Int
    let keyCount: Int
    /// Number of query heads
    let heads: Int
    /// Number of key heads and value heads, which divides the number of query heads
    let keyHeads: Int
    let keyDim: Int
    let valueDim: Int
    let outputDim: Int

    /// Checks the shapes of the operands of multi-head attention. The number of key heads follows from the shapes of the weights.
    init<N, Device>(
        queries: ShapedBuffer<N, Device>,
        keys: ShapedBuffer<N, Device>,
        values: ShapedBuffer<N, Device>,
        queryWeights: ShapedBuffer<N, Device>,
        keyWeights: ShapedBuffer<N, Device>,
        valueWeights: ShapedBuffer<N, Device>,
        outputWeights: ShapedBuffer<N, Device>,
        heads: Int,
    ) {
        precondition(queries.dim == 3 && keys.dim == 3 && values.dim == 3, "The queries, keys, and values must have 3 axes.")
        precondition(keys.shape[0] == queries.shape[0] && values.shape[0] == queries.shape[0], "The queries, keys, and values must have the same batch size.")
        precondition(keys.shape[1] == values.shape[1], "There must be one value for every key.")
        precondition(
            queryWeights.shape[0] == queries.shape[2] && keyWeights.shape[0] == keys.shape[2] && valueWeights.shape[0] == values.shape[2],
            "Every projection must have one input for every element of its vectors.",
        )
        precondition(heads > 0 && queryWeights.shape[1].isMultiple(of: heads), "The query projection must have a multiple of the number of heads as outputs.")
        let keyDim = queryWeights.shape[1] / heads
        precondition(keyWeights.shape[1].isMultiple(of: keyDim), "The key projection must have a multiple of the key size as outputs.")
        let keyHeads = keyWeights.shape[1] / keyDim
        precondition(keyHeads > 0 && heads.isMultiple(of: keyHeads), "The number of key heads must divide the number of query heads.")
        precondition(valueWeights.shape[1].isMultiple(of: keyHeads), "The value projection must have a multiple of the number of key heads as outputs.")
        let valueDim = valueWeights.shape[1] / keyHeads
        precondition(outputWeights.shape[0] == heads * valueDim, "The output projection must have one input for every element of the joined heads.")
        (batchSize, queryCount, keyCount) = (queries.shape[0], queries.shape[1], keys.shape[1])
        (self.heads, self.keyHeads, self.keyDim, self.valueDim, outputDim) = (heads, keyHeads, keyDim, valueDim, outputWeights.shape[1])
    }

    /// Shape of the result, [batchSize, queryCount, outputDim]
    var resultShape: [Int] {
        [batchSize, queryCount, outputDim]
    }

    /// Shapes of the attention of the heads
    var attention: AttentionShape {
        AttentionShape(heads: self)
    }
}

//
//  Attention.swift
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

// MARK: Attention

/// Computes scaled dot product attention as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The result is `softmax(queries × keysᵀ / temperature - 10⁹ \* mask) × values`, with the softmax along the keys.
///
/// - Parameters:
///   - queries: Queries, shape [batchSize, heads, queryCount, keyDim]
///   - keys: Keys, shape [batchSize, heads, keyCount, keyDim]
///   - values: Values, shape [batchSize, heads, keyCount, valueDim]
///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask. The mask gets no gradient.
///   - temperature: Divisor of the dot products
/// - Returns: Attended values, shape [batchSize, heads, queryCount, valueDim]
public func scaledDotProductAttention<Element, Device>(
    queries: Tensor<Element, Device>,
    keys: Tensor<Element, Device>,
    values: Tensor<Element, Device>,
    mask: Tensor<Element, Device>?,
    temperature: Element,
) -> Tensor<Element, Device> {
    precondition(queries.dim == 4 && keys.dim == 4 && values.dim == 4, "Queries, keys and values must have 4 axes.")
    let mask = mask?.detached()
    var result = Tensor<Element, Device>(uninitializedShape: [queries.shape[0], queries.shape[1], queries.shape[2], values.shape[3]])
    Device.FusedOperations.scaledDotProductAttention(queries: queries.values, keys: keys.values, values: values.values, mask: mask?.values, temperature: temperature, result: result.mutableValues)

    return result.attachingContext(tag: "scaledDotProductAttention", sources: queries, keys, values) { resultGradient, queryGradient, keyGradient, valueGradient in
        Composed.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: resultGradient, temperature: temperature, queryGradient: &queryGradient, keyGradient: &keyGradient, valueGradient: &valueGradient)
    } fused: { resultGradient, queryGradient, keyGradient, valueGradient in
        Device.FusedOperations.scaledDotProductAttentionBackward(queries: queries.values, keys: keys.values, values: values.values, mask: mask?.values, outputGradient: resultGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
    }
}

/// Computes multi-head attention with input and output projections, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The operation projects the queries, keys, and values with their weights, splits the projections into heads,
/// computes ``scaledDotProductAttention(queries:keys:values:mask:temperature:)`` for every head,
/// joins the heads, and multiplies the result with the output weights.
///
/// - Parameters:
///   - queries: Queries, shape [batchSize, queryCount, hiddenDim]
///   - keys: Keys, shape [batchSize, keyCount, hiddenDim]
///   - values: Values, shape [batchSize, keyCount, hiddenDim]
///   - mask: Mask with 1 for every key that a query must not attend to and 0 elsewhere,
///     broadcastable to [batchSize, heads, queryCount, keyCount], or nil for no mask. The mask gets no gradient.
///   - queryWeights: Query projection, shape [hiddenDim, heads \* keyDim]
///   - keyWeights: Key projection, shape [hiddenDim, heads \* keyDim]
///   - valueWeights: Value projection, shape [hiddenDim, heads \* valueDim]
///   - outputWeights: Output projection, shape [heads \* valueDim, outputDim]
///   - heads: Number of attention heads
///   - temperature: Divisor of the dot products
/// - Returns: Attended values, shape [batchSize, queryCount, outputDim]
public func multiHeadAttention<Element, Device>(
    queries: Tensor<Element, Device>,
    keys: Tensor<Element, Device>,
    values: Tensor<Element, Device>,
    mask: Tensor<Element, Device>?,
    queryWeights: Tensor<Element, Device>,
    keyWeights: Tensor<Element, Device>,
    valueWeights: Tensor<Element, Device>,
    outputWeights: Tensor<Element, Device>,
    heads: Int,
    temperature: Element,
) -> Tensor<Element, Device> {
    precondition(queries.dim == 3 && keys.dim == 3 && values.dim == 3, "Queries, keys and values must have 3 axes.")
    precondition(queryWeights.shape[1].isMultiple(of: heads) && valueWeights.shape[1].isMultiple(of: heads), "The projections must have a multiple of the number of heads as outputs.")
    let mask = mask?.detached()
    var result = Tensor<Element, Device>(uninitializedShape: [queries.shape[0], queries.shape[1], outputWeights.shape[1]])
    Device.FusedOperations.multiHeadAttention(
        queries: queries.values,
        keys: keys.values,
        values: values.values,
        mask: mask?.values,
        queryWeights: queryWeights.values,
        keyWeights: keyWeights.values,
        valueWeights: valueWeights.values,
        outputWeights: outputWeights.values,
        heads: heads,
        temperature: temperature,
        result: result.mutableValues,
    )

    return result.attachingContext(tag: "multiHeadAttention", sources: [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights]) { resultGradient, gradients in
        var accumulated = MultiHeadAttentionGradients(inSourceOrder: gradients.indices.map { gradients[$0].take() })
        Composed.multiHeadAttentionBackward(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            outputGradient: resultGradient,
            heads: heads,
            mask: mask,
            temperature: temperature,
            gradients: &accumulated,
        )
        gradients = accumulated.inSourceOrder
    } fused: { resultGradient, gradients in
        Device.FusedOperations.multiHeadAttentionBackward(
            queries: queries.values,
            keys: keys.values,
            values: values.values,
            mask: mask?.values,
            queryWeights: queryWeights.values,
            keyWeights: keyWeights.values,
            valueWeights: valueWeights.values,
            outputWeights: outputWeights.values,
            outputGradient: resultGradient,
            heads: heads,
            temperature: temperature,
            gradients: MultiHeadAttentionGradients(inSourceOrder: gradients),
        )
    }
}

public extension Tensor {
    /// Creates the sinusoidal positional encoding of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    ///
    /// The element at position `p` and index `2i` is `sin(p / 10000^(i / (hiddenSize / 2)))`,
    /// and the element at index `2i + 1` is the cosine of the same value.
    ///
    /// - Parameters:
    ///   - length: Number of positions
    ///   - hiddenSize: Number of elements per position, a multiple of 2
    init(positionalEncodingWithLength length: Int, hiddenSize: Int) {
        precondition(hiddenSize.isMultiple(of: 2), "Hidden size must be multiple of 2")
        self.init(uninitializedShape: [length, hiddenSize])
        Device.FusedOperations.positionalEncoding(length: length, hiddenSize: hiddenSize, result: mutableValues)
    }
}

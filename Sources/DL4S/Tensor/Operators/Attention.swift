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
    let result = Device.FusedOperations.scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature)

    return result.attachingContext(tag: "scaledDotProductAttention", sources: [queries, keys, values]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            let computed = Composed.scaledDotProductAttentionGradients(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                outputGradient: resultGradient,
                temperature: temperature,
                computesQueries: queries.requiresGradient,
                computesKeys: keys.requiresGradient,
                computesValues: values.requiresGradient,
            )
            Tensor.accumulate(computed.queries, into: &gradients[0])
            Tensor.accumulate(computed.keys, into: &gradients[1])
            Tensor.accumulate(computed.values, into: &gradients[2])
        } else {
            var accumulated = (queries: gradients[0].take(), keys: gradients[1].take(), values: gradients[2].take())
            Device.FusedOperations.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: resultGradient, temperature: temperature, accumulating: &accumulated)
            gradients[0] = accumulated.queries
            gradients[1] = accumulated.keys
            gradients[2] = accumulated.values
        }
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
    let result = Device.FusedOperations.multiHeadAttention(
        queries: queries,
        keys: keys,
        values: values,
        mask: mask,
        queryWeights: queryWeights,
        keyWeights: keyWeights,
        valueWeights: valueWeights,
        outputWeights: outputWeights,
        heads: heads,
        temperature: temperature,
    )

    let sources = [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights]
    return result.attachingContext(tag: "multiHeadAttention", sources: sources) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            let computed = Composed.multiHeadAttentionGradients(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                queryWeights: queryWeights,
                keyWeights: keyWeights,
                valueWeights: valueWeights,
                outputWeights: outputWeights,
                outputGradient: resultGradient,
                heads: heads,
                temperature: temperature,
                computes: sources.map(\.requiresGradient),
            )
            for (index, gradient) in computed.inSourceOrder.enumerated() {
                Tensor.accumulate(gradient, into: &gradients[index])
            }
        } else {
            // The struct takes the references out of the array, so the accumulated gradients stay uniquely referenced.
            var accumulated = MultiHeadAttentionGradients(
                queries: gradients[0].take(),
                keys: gradients[1].take(),
                values: gradients[2].take(),
                queryWeights: gradients[3].take(),
                keyWeights: gradients[4].take(),
                valueWeights: gradients[5].take(),
                outputWeights: gradients[6].take(),
            )
            Device.FusedOperations.multiHeadAttentionBackward(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                queryWeights: queryWeights,
                keyWeights: keyWeights,
                valueWeights: valueWeights,
                outputWeights: outputWeights,
                outputGradient: resultGradient,
                heads: heads,
                temperature: temperature,
                accumulating: &accumulated,
            )
            gradients = accumulated.inSourceOrder
        }
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
        let encoding: Self = Device.FusedOperations.positionalEncoding(length: length, hiddenSize: hiddenSize)
        self.init(handle: encoding.handle, shape: encoding.shape, context: nil)
    }
}

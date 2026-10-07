//
//  MultiHeadAttention.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.09.20.
//  Copyright (c) 2019 - 2020 - Palle Klewitz
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

/// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
@Layer
public struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: Codable, Sendable {
    /// Matrix multiplied with queries before dot product attention
    public var qDense: Tensor<Element, Device>
    /// Matrix multiplied with keys before dot product attention
    public var kDense: Tensor<Element, Device>
    /// Matrix multiplied with values before dot product attention
    public var vDense: Tensor<Element, Device>
    /// Matrix multiplied with result from dot product attention layer
    public var fc: Tensor<Element, Device>
    /// Divisor of the dot products of the queries and the keys
    public var temperature: Element
    public var norm: LayerNorm<Element, Device>
    public var dropout: Dropout<Element, Device>

    /// Number of query heads
    public let heads: Int
    /// Dimensionality of query and key vectors
    public let keyDim: Int
    /// Dimensionality of value vectors
    public let valueDim: Int
    /// Last dimension of keys, queries and values before matrix multiplication
    public let hiddenDim: Int

    /// Number of key and value heads. A group of `heads / keyValueHeads` query heads shares one key head and one value head.
    public var keyValueHeads: Int {
        kDense.shape[1] / keyDim
    }

    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of query heads
    ///   - keyValueHeads: Number of key and value heads, which divides `heads`, or nil for one key and value head per query head.
    ///     With fewer key heads than query heads, a group of query heads shares one key head and one value head (grouped-query attention).
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimension of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    public init(heads: Int, keyValueHeads: Int? = nil, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        var generator = WyHash()
        self.init(heads: heads, keyValueHeads: keyValueHeads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout, using: &generator)
    }

    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of query heads
    ///   - keyValueHeads: Number of key and value heads, which divides `heads`, or nil for one key and value head per query head.
    ///     With fewer key heads than query heads, a group of query heads shares one key head and one value head (grouped-query attention).
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimension of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(heads: Int, keyValueHeads: Int? = nil, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1, using generator: inout Generator) {
        let keyValueHeads = keyValueHeads ?? heads
        precondition(keyValueHeads > 0 && heads.isMultiple(of: keyValueHeads), "The number of key and value heads must divide the number of query heads.")
        self.heads = heads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.hiddenDim = hiddenDim

        temperature = Element(keyDim).sqrt()
        qDense = Tensor(heNormalWithShape: [hiddenDim, keyDim * heads], requiresGradient: true, using: &generator)
        kDense = Tensor(heNormalWithShape: [hiddenDim, keyDim * keyValueHeads], requiresGradient: true, using: &generator)
        vDense = Tensor(heNormalWithShape: [hiddenDim, valueDim * keyValueHeads], requiresGradient: true, using: &generator)
        fc = Tensor(heNormalWithShape: [valueDim * heads, hiddenDim], requiresGradient: true, using: &generator)
        self.dropout = Dropout(rate: dropout)
        norm = LayerNorm(inputSize: [hiddenDim])

        #if DEBUG
        qDense.tag = "qDense"
        kDense.tag = "kDense"
        vDense.tag = "vDense"
        fc.tag = "FC"
        #endif
    }

    /// Computes multi-head scaled dot product attention using the provided query, key and value vector as well as the provided mask.
    ///
    /// Additionally applies dropout, a residual connection and layer normalization.
    ///
    /// - Parameter inputs: Tuple containing queries of shape [batchSize, queryCount, hiddenDim], keys of shape [batchSize, keyCount, hiddenDim] and values of shape [batchSize, keyCount, hiddenDim]
    ///       as well as an optional mask that may be used to prevent attention to certain elements outside of the batch or in future timesteps. Mask must be broadcastable to shape [batchSize, heads, queryCount, keyCount] and have 1 entries for all elements that should be blocked.
    /// - Returns: Normalized scaled dot product attended values
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: (q: Tensor<Element, Device>, k: Tensor<Element, Device>, v: Tensor<Element, Device>, mask: Tensor<Element, Device>?)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MultiHeadAttention") {
            let (q, k, v, mask) = inputs // q, k, v: [batchSize, maxLen, hiddenDim]

            let attended = multiHeadAttention(
                queries: q,
                keys: k,
                values: v,
                mask: mask,
                queryWeights: qDense,
                keyWeights: kDense,
                valueWeights: vDense,
                outputWeights: fc,
                heads: heads,
                temperature: temperature,
            ) // [batchSize, queryCount, hiddenDim]
            return norm(dropout(attended) + q)
        }
    }
}

// MARK: Autoregressive decoding

public extension MultiHeadAttention {
    /// Projects keys and values and splits them into heads, for ``callAsFunction(queries:cache:mask:)``.
    /// - Parameters:
    ///   - keys: Keys with the shape [batchSize, count, hiddenDim]
    ///   - values: Values with the shape [batchSize, count, hiddenDim]
    /// - Returns: Cache with `count` positions
    func cache(keys: Tensor<Element, Device>, values: Tensor<Element, Device>) -> AttentionCache<Element, Device> {
        precondition(keys.dim == 3 && values.dim == 3 && keys.shape.dropLast() == values.shape.dropLast(), "The keys and values must have the shape [batchSize, count, hiddenDim].")
        return AttentionCache(
            keys: projectedHeads(keys, weights: kDense, heads: keyValueHeads),
            values: projectedHeads(values, weights: vDense, heads: keyValueHeads),
        )
    }

    /// Computes multi-head attention of queries to cached keys and values.
    ///
    /// The result is the result of ``callAsFunction(_:)`` with keys and values whose projections are in the cache:
    /// the operation projects the queries, attends to the cache, and applies the output projection, dropout,
    /// the residual connection, and layer normalization.
    ///
    /// - Parameters:
    ///   - queries: Queries with the shape [batchSize, queryCount, hiddenDim]
    ///   - cache: Keys and values from ``cache(keys:values:)``, with the batch size of the queries or 1
    ///   - mask: Mask with 1 for every cached position that a query must not attend to and 0 elsewhere,
    ///     broadcastable to [batchSize, heads, queryCount, cache.count], or nil for no mask
    /// - Returns: Attended values with the shape [batchSize, queryCount, hiddenDim]
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    func callAsFunction(queries: Tensor<Element, Device>, cache: AttentionCache<Element, Device>, mask: Tensor<Element, Device>?) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "MultiHeadAttention") {
            precondition(queries.dim == 3, "The queries must have the shape [batchSize, queryCount, hiddenDim].")
            precondition(cache.batchSize == 1 || cache.batchSize == queries.shape[0], "The cache must have the batch size of the queries or 1.")
            let (batchSize, queryCount) = (queries.shape[0], queries.shape[1])
            let projected = projectedHeads(queries, weights: qDense, heads: heads)
            let attended = scaledDotProductAttention(queries: projected, keys: cache.keys, values: cache.values, mask: mask, temperature: temperature) // [batchSize, heads, queryCount, valueDim]
            let joined = Self.swappingHeadsAndPositions(of: attended).view(as: batchSize * queryCount, heads * valueDim)
            let result = joined.matrixMultiplied(with: fc).view(as: batchSize, queryCount, -1)
            return norm(dropout(result) + queries)
        }
    }

    /// Projects inputs with the shape [batchSize, count, hiddenDim] and splits the projection into heads with the shape
    /// [batchSize, heads, count, size], the layout of the heads in
    /// ``multiHeadAttention(queries:keys:values:mask:queryWeights:keyWeights:valueWeights:outputWeights:heads:temperature:)``.
    private func projectedHeads(_ inputs: Tensor<Element, Device>, weights: Tensor<Element, Device>, heads: Int) -> Tensor<Element, Device> {
        let (batchSize, count) = (inputs.shape[0], inputs.shape[1])
        let projected = inputs
            .view(as: batchSize * count, hiddenDim)
            .matrixMultiplied(with: weights)
            .view(as: batchSize, count, heads, -1)
        return Self.swappingHeadsAndPositions(of: projected)
    }

    /// Swaps the second and third axes of a tensor with 4 axes, such as [batchSize, count, heads, size].
    private static func swappingHeadsAndPositions(of tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
        // With one position or one head, both layouts have the same memory order, so a decoding step of one position
        // needs no copies.
        guard tensor.shape[1] > 1, tensor.shape[2] > 1 else {
            return tensor.view(as: tensor.shape[0], tensor.shape[2], tensor.shape[1], tensor.shape[3])
        }
        return tensor.permuted(to: [0, 2, 1, 3])
    }
}

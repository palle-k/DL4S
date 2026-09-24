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
    public var attn: ScaledDotProductAttention<Element, Device>
    public var norm: LayerNorm<Element, Device>
    public var dropout: Dropout<Element, Device>

    /// Number of attention heads
    public let heads: Int
    /// Dimensionality of query and key vectors
    public let keyDim: Int
    /// Dimensionality of value vectors
    public let valueDim: Int
    /// Lat dimension of keys, queries and values before matrix multiplication
    public let hiddenDim: Int

    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of attention heads
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimesion of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    public init(heads: Int, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        var generator = WyHash()
        self.init(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout, using: &generator)
    }

    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of attention heads
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimesion of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(heads: Int, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1, using generator: inout Generator) {
        self.heads = heads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.hiddenDim = hiddenDim

        attn = ScaledDotProductAttention(temperature: Element(keyDim).sqrt())
        qDense = Tensor(xavierNormalWithShape: [hiddenDim, keyDim * heads], requiresGradient: true, using: &generator)
        kDense = Tensor(xavierNormalWithShape: [hiddenDim, keyDim * heads], requiresGradient: true, using: &generator)
        vDense = Tensor(xavierNormalWithShape: [hiddenDim, valueDim * heads], requiresGradient: true, using: &generator)
        fc = Tensor(xavierNormalWithShape: [valueDim * heads, hiddenDim], requiresGradient: true, using: &generator)
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
    /// - Parameter inputs: Tuple containing queries of shape [batchSize, queryCount, hiddenDim], keys of shape [batchSize, keyCount, hiddenDim] and values of shape [batchSize, valueCount, hiddenDim]
    ///       as well as an optional mask that may be used to prevent attention to certain elements outside of the batch or in future timesteps. Mask must be broadcastable to shape [batchSize, heads, queryCount, keyCount] and have 1 entries for all elements that should be blocked.
    /// - Returns: Normalized scaled dot product attended values
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
                temperature: attn.temperature,
            ) // [batchSize, queryCount, hiddenDim]
            return norm(dropout(attended) + q)
        }
    }
}

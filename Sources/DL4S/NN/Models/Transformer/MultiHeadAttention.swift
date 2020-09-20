//
//  File.swift
//  
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
public struct MultiHeadAttention<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
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
    
    public var parameters: [Tensor<Element, Device>] {Array([
        [qDense, kDense, vDense, fc],
        norm.parameters
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        [\.qDense, \.kDense, \.vDense, \.fc],
        norm.parameterPaths.map((\Self.norm).appending(path:))
    ].joined())}
    
    /// Multi-Head Attention Layer following [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - heads: Number of attention heads
    ///   - hiddenDim: Last dimension of keys, queries and values
    ///   - keyDim: Last dimesion of keys
    ///   - valueDim: Intermediate last dimension of values
    ///   - dropout: Dropout rate
    public init(heads: Int, hiddenDim: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        self.heads = heads
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.hiddenDim = hiddenDim
        
        attn = ScaledDotProductAttention(temperature: Element(keyDim).sqrt())
        qDense = Tensor(xavierNormalWithShape: hiddenDim, keyDim * heads, requiresGradient: true)
        kDense = Tensor(xavierNormalWithShape: hiddenDim, keyDim * heads, requiresGradient: true)
        vDense = Tensor(xavierNormalWithShape: hiddenDim, valueDim * heads, requiresGradient: true)
        fc = Tensor(xavierNormalWithShape: valueDim * heads, hiddenDim, requiresGradient: true)
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
            
            let batchSize = q.shape[0]
            let queryCount = q.shape[1] // == maxLen
            let keyCount = k.shape[1]
            let valueCount = v.shape[1]
            
            let res = q
            
            // [batchSize, queryCount, hiddenDim] x [hiddenDim, keyDim * heads] --> [batchSize, maxLen, keyDim * heads]
            let q_prep = q.broadcastMatrixMultiplied(with: qDense).view(as: batchSize, queryCount, heads, keyDim) // [batchSize, queryCount
            let k_prep = k.broadcastMatrixMultiplied(with: kDense).view(as: batchSize, keyCount, heads, keyDim)
            let v_prep = v.broadcastMatrixMultiplied(with: vDense).view(as: batchSize, valueCount, heads, valueDim)
            
            let q_trans = q_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, queryCount, keyDim]
            let k_trans = k_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, keyCount, keyDim]
            let v_trans = v_prep.permuted(to: 0, 2, 1, 3) // [batchSize, heads, valueCount, valueDim]
            
            let q_attn = self.attn((q: q_trans, k: k_trans, v: v_trans, mask: mask)) // [batchSize, heads, queryCount, valueDim]
            
            let out = q_attn.permuted(to: 0, 2, 1, 3) // [batchSize, queryCount, heads, valueDim]
                .view(as: batchSize, queryCount, -1) // [batchSize, queryCount, heads * valueDim]
                .broadcastMatrixMultiplied(with: fc) // [batchSize, queryCount, heads * valueDim] x [valueDim * heads, hiddenDim] --> [batchSize, queryCount, hiddenDim]
            let out_drop = dropout(out)
            let q_res = out_drop + res
            let normalized = norm(q_res)
            return normalized // [batchSize, queryCount, hiddenDim]
        }
    }
}

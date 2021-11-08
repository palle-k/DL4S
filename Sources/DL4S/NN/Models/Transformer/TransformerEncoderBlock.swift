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

/// Transformer encoder layer consisting of a self-attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
public struct TransformerEncoderBlock<Element: RandomizableType, Device: DeviceType>: LayerType, Codable {
    public var selfAttention: MultiHeadAttention<Element, Device>
    public var pointwiseFeedForward: PointwiseFeedForward<Element, Device>
    
    public var parameters: [Tensor<Element, Device>] {Array([
        selfAttention.parameters, pointwiseFeedForward.parameters
    ].joined())}
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {Array([
        selfAttention.parameterPaths.map((\Self.selfAttention).appending(path:)),
        pointwiseFeedForward.parameterPaths.map((\Self.pointwiseFeedForward).appending(path:))
    ].joined())}
    
    /// Creates Transformer encoder layer consisting of a self-attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - hiddenDim: Last dimension of inputs and outputs
    ///   - forwardDim: Size of value vectors within pointwise feed forward layer
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors within multi-head attention layer
    ///   - valueDim: Size of value vectors within multi-head attention layer
    ///   - dropout: Dropout rate for dropout applied within self-attention and pointwise feed forward layer
    public init(hiddenDim: Int, forwardDim: Int, heads: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1) {
        selfAttention = MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout)
        pointwiseFeedForward = PointwiseFeedForward(size: hiddenDim, hiddenSize: forwardDim, dropoutRate: dropout)
    }
    
    /// Applies multi-head self attention and a pointwise feed forward layer to the inputs
    /// - Parameter inputs: Layer input with shape [batchSize, maxLen, hiddenSize] and padding mask broadcastable to [batchSize, heads, queryCount, keyCount] with 1 entries for all elements that should be blocked.
    /// - Returns: Result of layer operations with shape [batchSize, maxLen, hiddenSize]
    public func callAsFunction(_ inputs: (inputs: Tensor<Element, Device>, mask: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "EncoderLayer") {
            // inputs: [batchSize, maxLen == queryCount, hiddenDim]
            let (inputs, mask) = inputs
            let attn = selfAttention((q: inputs, k: inputs, v: inputs, mask: mask))
            return pointwiseFeedForward(attn)
        }
    }
}

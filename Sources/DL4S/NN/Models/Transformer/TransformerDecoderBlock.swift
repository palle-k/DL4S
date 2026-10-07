//
//  TransformerDecoderBlock.swift
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

/// Transformer decoder layer consisting of a self attention, an optional encoder attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// A block without encoder attention is a block of a decoder-only model.
@Layer
public struct TransformerDecoderBlock<Element: RandomizableType, Device: DeviceType>: Codable, Sendable {
    public var selfAttention: MultiHeadAttention<Element, Device>
    /// Attention to the encoder states, or nil in a decoder-only model
    public var encoderAttention: MultiHeadAttention<Element, Device>?
    public var pointwiseFeedForward: PointwiseFeedForward<Element, Device>

    /// Whether the block attends to the states of an encoder
    public var attendsToEncoder: Bool {
        encoderAttention != nil
    }

    /// Creates Transformer decoder layer consisting of a self attention, an optional encoder attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - hiddenDim: Last dimension of inputs and outputs
    ///   - forwardDim: Size of value vectors within pointwise feed forward layer
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors within multi-head attention layer
    ///   - valueDim: Size of value vectors within multi-head attention layer
    ///   - dropout: Dropout rate for dropout applied within self-attention and pointwise feed forward layer
    ///   - attendsToEncoder: Whether the block has an encoder attention. A block of a decoder-only model has none.
    public init(hiddenDim: Int, forwardDim: Int, heads: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1, attendsToEncoder: Bool = true) {
        var generator = WyHash()
        self.init(hiddenDim: hiddenDim, forwardDim: forwardDim, heads: heads, keyDim: keyDim, valueDim: valueDim, dropout: dropout, attendsToEncoder: attendsToEncoder, using: &generator)
    }

    /// Creates Transformer decoder layer consisting of a self attention, an optional encoder attention and a pointwise feed forward layer as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - hiddenDim: Last dimension of inputs and outputs
    ///   - forwardDim: Size of value vectors within pointwise feed forward layer
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors within multi-head attention layer
    ///   - valueDim: Size of value vectors within multi-head attention layer
    ///   - dropout: Dropout rate for dropout applied within self-attention and pointwise feed forward layer
    ///   - attendsToEncoder: Whether the block has an encoder attention. A block of a decoder-only model has none.
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(hiddenDim: Int, forwardDim: Int, heads: Int, keyDim: Int, valueDim: Int, dropout: Float = 0.1, attendsToEncoder: Bool = true, using generator: inout Generator) {
        selfAttention = MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout, using: &generator)
        encoderAttention = attendsToEncoder ? MultiHeadAttention(heads: heads, hiddenDim: hiddenDim, keyDim: keyDim, valueDim: valueDim, dropout: dropout, using: &generator) : nil
        pointwiseFeedForward = PointwiseFeedForward(size: hiddenDim, hiddenSize: forwardDim, dropoutRate: dropout, using: &generator)
    }

    /// Applies multi-head self attention, the encoder attention, and a pointwise feed forward layer to the inputs.
    /// - Parameter inputs: Layer input with shape [batchSize, maxLen, hiddenSize], encoder outputs with shape [batchSize, maxLen, hiddenSize], and masks broadcastable to [batchSize, heads, queryCount, keyCount] with 1 entries for all elements that should be blocked for encoder and decoder states.
    ///   The encoder outputs and the encoder mask must be nil when the block has no encoder attention, and the encoder outputs must not be nil when it has one.
    /// - Returns: Result of layer operations with shape [batchSize, maxLen, hiddenSize]
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: (decoderInput: Tensor<Element, Device>, encoderOutput: Tensor<Element, Device>?, encoderMask: Tensor<Element, Device>?, decoderMask: Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "DecoderLayer") {
            let (decoderInput, encoderOutput, encoderMask, decoderMask) = inputs
            precondition((encoderOutput != nil) == attendsToEncoder, attendsToEncoder ? "The block attends to an encoder, so it needs the encoder outputs." : "The block has no encoder attention, so it takes no encoder outputs.")
            let attended = selfAttention((q: decoderInput, k: decoderInput, v: decoderInput, mask: decoderMask))
            let encoderAttended = if let encoderAttention, let encoderOutput {
                encoderAttention((q: attended, k: encoderOutput, v: encoderOutput, mask: encoderMask))
            } else {
                attended
            }
            return pointwiseFeedForward(encoderAttended)
        }
    }

    /// Applies the block to new positions and appends their keys and values to the cache of the self attention.
    /// - Parameters:
    ///   - input: Inputs of the new positions with the shape [batchSize, count, hiddenSize]
    ///   - cache: Keys and values of the self attention at the previous positions, or nil before the first step
    ///   - mask: Mask of the self attention, broadcastable to [batchSize, heads, count, previousCount + count]
    ///   - encoder: Keys and values of the encoder attention, which must be nil exactly when the block has no encoder attention
    ///   - encoderMask: Mask of the encoder attention
    /// - Returns: Result of the block with the shape of the input
    func decode(
        _ input: Tensor<Element, Device>,
        cache: inout AttentionCache<Element, Device>?,
        mask: Tensor<Element, Device>?,
        encoder: AttentionCache<Element, Device>?,
        encoderMask: Tensor<Element, Device>?,
    ) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "DecoderLayer") {
            precondition((encoder != nil) == attendsToEncoder, attendsToEncoder ? "The block attends to an encoder, so it needs the encoder cache." : "The block has no encoder attention, so it takes no encoder cache.")
            let added = selfAttention.cache(keys: input, values: input)
            let extended = cache.map { $0.appending(added) } ?? added
            cache = extended
            let attended = selfAttention(queries: input, cache: extended, mask: mask)
            let encoderAttended = if let encoderAttention, let encoder {
                encoderAttention(queries: attended, cache: encoder, mask: encoderMask)
            } else {
                attended
            }
            return pointwiseFeedForward(encoderAttended)
        }
    }
}

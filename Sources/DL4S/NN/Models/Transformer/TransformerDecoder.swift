//
//  TransformerDecoder.swift
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

/// The output of an encoder that a decoder attends to.
public struct EncodedSequence<Element: NumericType, Device: DeviceType>: Sendable {
    /// Encoder states with the shape [batchSize, sourceLength, hiddenDim]
    public var states: Tensor<Element, Device>
    /// Length of each source sequence without padding
    public var lengths: [Int]

    /// Creates the output of an encoder.
    /// - Parameters:
    ///   - states: Encoder states with the shape [batchSize, sourceLength, hiddenDim]
    ///   - lengths: Length of each source sequence without padding, one per sequence of the batch
    public init(states: Tensor<Element, Device>, lengths: [Int]) {
        precondition(states.dim == 3 && states.shape[0] == lengths.count, "The encoder states must have the shape [batchSize, sourceLength, hiddenDim], with one length per sequence.")
        precondition(lengths.allSatisfy { 0 ... states.shape[1] ~= $0 }, "Every length must be in 0 ... \(states.shape[1]).")
        self.states = states
        self.lengths = lengths
    }
}

/// The inputs of a ``TransformerDecoder`` that decodes whole sequences.
public struct TransformerDecoderInputs<Element: NumericType, Device: DeviceType>: Sendable {
    /// Embedded decoder inputs with positional encoding, with the shape [batchSize, length, hiddenDim]
    public var input: Tensor<Element, Device>
    /// Length of each decoder input sequence without padding
    public var lengths: [Int]
    /// Output of the encoder, or nil for a decoder that does not attend to an encoder
    public var encoded: EncodedSequence<Element, Device>?

    /// Creates the inputs of a decoder.
    /// - Parameters:
    ///   - input: Embedded decoder inputs with positional encoding, with the shape [batchSize, length, hiddenDim]
    ///   - lengths: Length of each decoder input sequence without padding
    ///   - encoded: Output of the encoder, or nil for a decoder that does not attend to an encoder
    public init(input: Tensor<Element, Device>, lengths: [Int], encoded: EncodedSequence<Element, Device>? = nil) {
        precondition(input.dim == 3 && input.shape[0] == lengths.count, "The decoder input must have the shape [batchSize, length, hiddenDim], with one length per sequence.")
        self.input = input
        self.lengths = lengths
        self.encoded = encoded
    }
}

/// Transformer decoder sequencing multiple transformer decoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
///
/// The decoder computes whole sequences with ``callAsFunction(_:)``, which is the form for training, and decodes
/// autoregressively with ``makeState(batchSize:encoded:)`` and ``decode(_:lengths:state:)``, which compute only the new
/// positions of each step. A decoder whose blocks do not attend to an encoder is the decoder of a decoder-only model.
@Layer
public struct TransformerDecoder<Element: RandomizableType, Device: DeviceType>: Codable, Sendable {
    public var decoderLayers: [TransformerDecoderBlock<Element, Device>]

    /// Whether a block of the decoder attends to the states of an encoder
    public var attendsToEncoder: Bool {
        decoderLayers.contains(where: \.attendsToEncoder)
    }

    /// Creates a transformer decoder sequencing multiple transformer decoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameter attendsToEncoder: Whether the blocks attend to the states of an encoder. The blocks of a decoder-only model do not.
    public init(layerCount: Int, heads: Int, keyDim: Int, valueDim: Int, modelDim: Int, forwardDim: Int, dropout: Float, attendsToEncoder: Bool = true) {
        var generator = WyHash()
        self.init(layerCount: layerCount, heads: heads, keyDim: keyDim, valueDim: valueDim, modelDim: modelDim, forwardDim: forwardDim, dropout: dropout, attendsToEncoder: attendsToEncoder, using: &generator)
    }

    /// Creates a transformer decoder sequencing multiple transformer decoder layers, as introduced by [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf).
    /// - Parameters:
    ///   - attendsToEncoder: Whether the blocks attend to the states of an encoder. The blocks of a decoder-only model do not.
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(layerCount: Int, heads: Int, keyDim: Int, valueDim: Int, modelDim: Int, forwardDim: Int, dropout: Float, attendsToEncoder: Bool = true, using generator: inout Generator) {
        decoderLayers = (0 ..< layerCount).map { _ in
            TransformerDecoderBlock(hiddenDim: modelDim, forwardDim: forwardDim, heads: heads, keyDim: keyDim, valueDim: valueDim, dropout: dropout, attendsToEncoder: attendsToEncoder, using: &generator)
        }
    }

    /// Decodes whole sequences.
    ///
    /// Every position attends to itself and to the positions before it.
    /// - Parameter inputs: Decoder inputs, with the encoder output exactly when the decoder attends to an encoder
    /// - Returns: Decoded sequences with the shape [batchSize, length, hiddenDim]
    public func callAsFunction(_ inputs: TransformerDecoderInputs<Element, Device>) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Decoder") {
            precondition((inputs.encoded != nil) == attendsToEncoder, attendsToEncoder ? "The decoder attends to an encoder, so it needs the encoder output." : "The decoder does not attend to an encoder, so it takes no encoder output.")
            let encoderMask: Tensor<Element, Device>? = inputs.encoded.map { makeEncoderMasks(sequenceLengths: $0.lengths, length: $0.states.shape[1]) }
            let decoderMask: Tensor<Element, Device> = makeDecoderMasks(sequenceLengths: inputs.lengths)

            return decoderLayers.reduce(inputs.input) { acc, layer in
                layer.attendsToEncoder
                    ? layer((acc, inputs.encoded?.states, encoderMask, decoderMask))
                    : layer((acc, nil, nil, decoderMask))
            } // [batchSize, maxLen, hiddenDim]
        }
    }

    /// Creates the state of an autoregressive decoding, without decoded positions.
    /// - Parameters:
    ///   - batchSize: Number of sequences to decode
    ///   - encoded: Output of the encoder with the batch size `batchSize` or 1, or nil for a decoder that does not attend
    ///     to an encoder. With the batch size 1, every sequence attends to the same encoder output, as the hypotheses of a beam search do.
    /// - Returns: State for ``decode(_:lengths:state:)``
    public func makeState(batchSize: Int, encoded: EncodedSequence<Element, Device>? = nil) -> TransformerDecoderState<Element, Device> {
        precondition((encoded != nil) == attendsToEncoder, attendsToEncoder ? "The decoder attends to an encoder, so it needs the encoder output." : "The decoder does not attend to an encoder, so it takes no encoder output.")
        precondition(batchSize > 0, "The batch size must be positive.")
        guard let encoded else {
            return TransformerDecoderState(layerCount: decoderLayers.count, batchSize: batchSize, encoder: nil)
        }
        precondition(encoded.lengths.count == 1 || encoded.lengths.count == batchSize, "The encoder output must have the batch size \(batchSize) or 1.")
        let sourceLength = encoded.states.shape[1]
        let caches = decoderLayers.map { layer in
            layer.encoderAttention.map { $0.cache(keys: encoded.states, values: encoded.states) }
        }
        // A source sequence without padding needs no mask.
        let mask: Tensor<Element, Device>? = encoded.lengths.allSatisfy { $0 == sourceLength } ? nil : makeEncoderMasks(sequenceLengths: encoded.lengths, length: sourceLength)
        return TransformerDecoderState(layerCount: decoderLayers.count, batchSize: batchSize, encoder: TransformerDecoderState.Encoder(caches: caches, mask: mask))
    }

    /// Decodes new positions of every sequence and appends them to the state.
    ///
    /// The new positions attend to every decoded position of their sequence, to themselves, and to the new positions
    /// before them. The result at a position is the result of ``callAsFunction(_:)`` at that position for the sequence
    /// of all positions so far.
    ///
    /// A step can decode several positions, for example the prompts of a decoder-only model, whose lengths can differ.
    /// The positions after the length of a sequence are padding: their results are undefined, and later steps do not
    /// attend to them. Add positional encodings at ``TransformerDecoderState/positions(count:)`` to the inputs.
    ///
    /// - Parameters:
    ///   - next: Embedded inputs of the new positions with positional encoding, with the shape [batchSize, count, hiddenDim]
    ///   - lengths: Number of new positions of each sequence that are not padding, in `0 ... count`, or nil when every new position is a position of its sequence
    ///   - state: State of the decoding, which this function updates
    /// - Returns: Results at the new positions with the shape [batchSize, count, hiddenDim]
    public func decode(_ next: Tensor<Element, Device>, lengths: [Int]? = nil, state: inout TransformerDecoderState<Element, Device>) -> Tensor<Element, Device> {
        precondition(next.dim == 3 && next.shape[0] == state.batchSize, "The inputs must have the shape [\(state.batchSize), count, hiddenDim].")
        precondition(state.selfAttention.count == decoderLayers.count, "The state belongs to a decoder with \(state.selfAttention.count) blocks.")
        let count = next.shape[1]
        let lengths = lengths ?? Array(repeating: count, count: state.batchSize)
        precondition(lengths.count == state.batchSize && lengths.allSatisfy { 0 ... count ~= $0 }, "There must be one length in 0 ... \(count) per sequence.")

        return OperationGroup.capture(named: "Decoder") {
            let mask = state.selfAttentionMask(count: count)
            var result = next
            for index in decoderLayers.indices {
                result = decoderLayers[index].decode(result, cache: &state.selfAttention[index], mask: mask, encoder: state.encoder?.caches[index], encoderMask: state.encoder?.mask)
            }
            state.advance(count: count, lengths: lengths)
            return result
        }
    }
}

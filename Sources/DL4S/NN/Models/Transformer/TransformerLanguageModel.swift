//
//  TransformerLanguageModel.swift
//  DL4S
//
//  Created by Palle Klewitz on 01.10.26.
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

/// Token sequences of a batch, padded at the end to the longest sequence.
public struct TokenSequences<Device: DeviceType>: Sendable {
    /// Token indices with the shape [batchSize, length]
    public var tokens: Tensor<Int32, Device>
    /// Length of each sequence without padding
    public var lengths: [Int]

    /// Creates a batch from padded token indices.
    /// - Parameters:
    ///   - tokens: Token indices with the shape [batchSize, length]
    ///   - lengths: Length of each sequence without padding
    public init(tokens: Tensor<Int32, Device>, lengths: [Int]) {
        precondition(tokens.dim == 2 && tokens.shape[0] == lengths.count, "The tokens must have the shape [batchSize, length], with one length per sequence.")
        precondition(lengths.allSatisfy { 0 ... tokens.shape[1] ~= $0 }, "Every length must be in 0 ... \(tokens.shape[1]).")
        self.tokens = tokens
        self.lengths = lengths
    }

    /// Creates a batch from sequences and pads the shorter ones.
    /// - Parameters:
    ///   - sequences: Token indices of every sequence. There must be at least one sequence with at least one token.
    ///   - padding: Index of padding positions
    public init(_ sequences: [[Int32]], padding: Int32 = -1) {
        let length = sequences.map(\.count).max() ?? 0
        precondition(length > 0, "The batch must contain at least one token.")
        self.init(
            tokens: Tensor(sequences.map { $0 + Array(repeating: padding, count: length - $0.count) }),
            lengths: sequences.map(\.count),
        )
    }
}

/// Decoder-only transformer language model, which predicts the next token of a sequence.
///
/// The model consists of a token embedding with positional encoding, a ``TransformerDecoder`` whose blocks do not attend
/// to an encoder, and an output projection that shares its weights with the embedding.
/// It trains on whole sequences with ``callAsFunction(_:)`` and generates with ``generate(prompts:maxLength:endToken:)``,
/// which decodes one position per step with a ``TransformerDecoderState``.
@Layer
public struct TransformerLanguageModel<Element: RandomizableType, Device: DeviceType>: Codable, Sendable {
    public typealias Outputs = Tensor<Element, Device> // disambiguates callAsFunction protocol requirement

    public var embedding: Embedding<Element, Device>
    public var positionalEncoding: PositionalEncoding<Element, Device>
    public var dropout: Dropout<Element, Device>
    public var decoder: TransformerDecoder<Element, Device>
    public var outputBias: Tensor<Element, Device>

    /// Creates a decoder-only transformer language model.
    /// - Parameters:
    ///   - layers: Number of decoder blocks
    ///   - vocabSize: Number of tokens
    ///   - hiddenDim: Size of the embeddings and of the states of the blocks
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors of a head
    ///   - valueDim: Size of value vectors of a head
    ///   - forwardDim: Size of the hidden layer of the pointwise feed forward layers
    ///   - dropout: Dropout rate
    public init(layers: Int, vocabSize: Int, hiddenDim: Int, heads: Int, keyDim: Int, valueDim: Int, forwardDim: Int, dropout: Float = 0.1) {
        var generator = WyHash()
        self.init(layers: layers, vocabSize: vocabSize, hiddenDim: hiddenDim, heads: heads, keyDim: keyDim, valueDim: valueDim, forwardDim: forwardDim, dropout: dropout, using: &generator)
    }

    /// Creates a decoder-only transformer language model.
    /// - Parameters:
    ///   - layers: Number of decoder blocks
    ///   - vocabSize: Number of tokens
    ///   - hiddenDim: Size of the embeddings and of the states of the blocks
    ///   - heads: Number of attention heads
    ///   - keyDim: Size of key and query vectors of a head
    ///   - valueDim: Size of value vectors of a head
    ///   - forwardDim: Size of the hidden layer of the pointwise feed forward layers
    ///   - dropout: Dropout rate
    ///   - generator: Random number generator that provides the initial weights.
    public init<Generator: RandomNumberGenerator>(layers: Int, vocabSize: Int, hiddenDim: Int, heads: Int, keyDim: Int, valueDim: Int, forwardDim: Int, dropout: Float = 0.1, using generator: inout Generator) {
        embedding = Embedding(inputFeatures: vocabSize, outputSize: hiddenDim, ignoreIndex: -1, using: &generator)
        positionalEncoding = PositionalEncoding(hiddenSize: hiddenDim)
        self.dropout = Dropout(rate: dropout)
        decoder = TransformerDecoder(layerCount: layers, heads: heads, keyDim: keyDim, valueDim: valueDim, modelDim: hiddenDim, forwardDim: forwardDim, dropout: dropout, attendsToEncoder: false, using: &generator)
        outputBias = Tensor(repeating: 0, shape: [vocabSize], requiresGradient: true)
    }

    /// Predicts the next token at every position of the sequences.
    /// - Parameter inputs: Token sequences with padding index -1
    /// - Returns: Log probabilities of the next token with the shape [batchSize, length, vocabSize]
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: TokenSequences<Device>) -> Tensor<Element, Device> {
        let length = inputs.tokens.shape[1]
        let decoded = decoder(TransformerDecoderInputs(input: embedded(inputs.tokens, positionalEncodings: positionalEncoding(length)), lengths: inputs.lengths))
        return logSoftmax(logits(decoded), axis: 2)
    }

    /// Generates continuations of prompts by greedy decoding.
    ///
    /// The model decodes the prompts in one step and then one token of every unfinished sequence per step.
    /// The model decodes without dropout, also when its ``Dropout`` layers are active.
    /// - Parameters:
    ///   - prompts: Token indices of the prompts. Every prompt must contain at least one token.
    ///   - maxLength: Maximum number of tokens to generate for each prompt
    ///   - endToken: Token that ends a sequence, which is part of the result, or nil to generate `maxLength` tokens
    /// - Returns: Generated tokens of each prompt, without the prompt
    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func generate(prompts: [[Int32]], maxLength: Int, endToken: Int32? = nil) -> [[Int32]] {
        var model = self
        model.modifyLayers(of: Dropout<Element, Device>.self) { $0.isActive = false }
        return model.greedilyDecoded(prompts: prompts, maxLength: maxLength, endToken: endToken)
    }

    /// Generates continuations of prompts by greedy decoding with the dropout layers of the model.
    private func greedilyDecoded(prompts: [[Int32]], maxLength: Int, endToken: Int32?) -> [[Int32]] {
        precondition(!prompts.isEmpty && prompts.allSatisfy { !$0.isEmpty }, "There must be at least one prompt, and every prompt must contain a token.")
        var generated = [[Int32]](repeating: [], count: prompts.count)
        guard maxLength > 0 else {
            return generated
        }
        let prompt = TokenSequences<Device>(prompts)
        let promptLength = prompt.tokens.shape[1]
        var state = decoder.makeState(batchSize: prompts.count)
        let decoded = decoder.decode(embedded(prompt.tokens, positionalEncodings: positionalEncodings(of: state, count: promptLength)), lengths: prompt.lengths, state: &state)
        // The prediction of a prompt comes from the position of its last token.
        let lastPositions = Tensor<Int32, Device>(prompt.lengths.enumerated().map { Int32($0.offset * promptLength + $0.element - 1) })
        var next = mostProbableTokens(decoded.view(as: prompts.count * promptLength, -1).gatheringRows(at: lastPositions))

        // Index of the prompt of each sequence of the state
        var sequences = Array(prompts.indices)
        for step in 0 ..< maxLength {
            for (sequence, token) in zip(sequences, next) {
                generated[sequence].append(token)
            }
            let unfinished = next.indices.filter { next[$0] != endToken }
            guard step + 1 < maxLength, !unfinished.isEmpty else {
                break
            }
            if unfinished.count < sequences.count {
                state = state.selecting(sequences: unfinished)
                sequences = unfinished.map { sequences[$0] }
                next = unfinished.map { next[$0] }
            }
            let input = embedded(Tensor(next).view(as: next.count, 1), positionalEncodings: positionalEncodings(of: state, count: 1))
            next = mostProbableTokens(decoder.decode(input, state: &state).view(as: next.count, -1))
        }
        return generated
    }

    /// Embeds tokens with the shape [batchSize, length] and adds positional encodings that broadcast to [batchSize, length, hiddenDim].
    private func embedded(_ tokens: Tensor<Int32, Device>, positionalEncodings: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let embedded = embedding(tokens.flattened()).view(as: tokens.shape[0], tokens.shape[1], -1)
        return dropout(embedded * Tensor(Element(embedded.shape[2]).sqrt()) + positionalEncodings)
    }

    /// Positional encodings of the positions of the next step, with the shape [batchSize, count, hiddenDim].
    private func positionalEncodings(of state: TransformerDecoderState<Element, Device>, count: Int) -> Tensor<Element, Device> {
        // Every position of the step is less than state.count + count.
        positionalEncoding(state.count + count).gatheringRows(at: state.positions(count: count))
    }

    /// Projects decoder results onto the vocabulary.
    private func logits(_ decoded: Tensor<Element, Device>) -> Tensor<Element, Device> {
        decoded.broadcastMatrixMultiplied(with: embedding.embeddingMatrix, transposeOther: true) + outputBias
    }

    /// Returns the most probable token for each row of decoder results with the shape [batchSize, hiddenDim].
    private func mostProbableTokens(_ decoded: Tensor<Element, Device>) -> [Int32] {
        let vocabSize = embedding.inputFeatures
        // One read of all logits waits for the device once.
        let elements = logits(decoded).elements
        return (0 ..< decoded.shape[0]).map { row in
            let scores = elements[(row * vocabSize) ..< ((row + 1) * vocabSize)]
            return Int32(scores.indices.max { scores[$0] < scores[$1] }! - row * vocabSize)
        }
    }
}

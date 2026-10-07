//
//  TransformerDecoderState.swift
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

/// The state of an autoregressive decoding with a ``TransformerDecoder``.
///
/// The state holds the keys and values of the self attention of every block at the decoded positions, and the keys and
/// values of the encoder attention, which the decoder projects once. Create it with
/// ``TransformerDecoder/makeState(batchSize:encoded:)`` and pass it to ``TransformerDecoder/decode(_:lengths:state:)``.
///
/// The state is a value: a copy, for example of a decoded prompt, continues independently of the original.
public struct TransformerDecoderState<Element: NumericType, Device: DeviceType>: Sendable {
    /// The encoder attention of the blocks.
    struct Encoder: Sendable {
        /// Keys and values of the encoder attention of every block, or nil for a block without encoder attention
        var caches: [AttentionCache<Element, Device>?]
        /// Mask of the padding of the source sequences with the shape [batchSize, 1, 1, sourceLength], or nil without padding
        var mask: Tensor<Element, Device>?
    }

    /// Number of decoded positions of each sequence, without padding
    public private(set) var lengths: [Int]

    /// Number of cached positions of every sequence, with padding
    public private(set) var count = 0

    /// Number of sequences
    public var batchSize: Int {
        lengths.count
    }

    /// Keys and values of the self attention of every block, nil before the first step
    var selfAttention: [AttentionCache<Element, Device>?]

    /// Encoder attention of the blocks, or nil in a decoder-only model
    private(set) var encoder: Encoder?

    // padding[sequence][position] is true for a cached position after the length of the sequence in a step. The host
    // builds the masks from it, and a state without padding needs no masks for steps of one position.
    private var padding: [[Bool]]
    private var hasPadding = false

    init(layerCount: Int, batchSize: Int, encoder: Encoder?) {
        lengths = Array(repeating: 0, count: batchSize)
        selfAttention = Array(repeating: nil, count: layerCount)
        padding = Array(repeating: [], count: batchSize)
        self.encoder = encoder
    }

    /// Returns the positions of the next positions of every sequence, for the positional encoding of the inputs of a step.
    /// - Parameter count: Number of positions of the step
    /// - Returns: Positions with the shape [batchSize, count]: `lengths[sequence] + index`
    public func positions(count: Int) -> Tensor<Int32, Device> {
        Tensor(Array(lengths.map { length in (0 ..< count).map { Int32(length + $0) } }.joined()), shape: [batchSize, count])
    }

    /// Selects sequences, for example the hypotheses that a step of a beam search continues.
    /// - Parameter indices: Index of the sequence of this state for each sequence of the result. Indices can repeat.
    /// - Returns: State with one sequence per index
    public func selecting(sequences indices: [Int]) -> Self {
        precondition(!indices.isEmpty && indices.allSatisfy { 0 ..< batchSize ~= $0 }, "The indices must select at least one of the \(batchSize) sequences.")
        let rows = Tensor<Int32, Device>(indices.map { Int32($0) })
        var selected = self
        selected.lengths = indices.map { lengths[$0] }
        selected.padding = indices.map { padding[$0] }
        selected.selfAttention = selfAttention.map { $0?.selecting(batches: rows) }
        // An encoder output with the batch size 1 belongs to every sequence.
        if var encoder, encoder.caches.contains(where: { ($0?.batchSize ?? 1) > 1 }) {
            encoder.caches = encoder.caches.map { $0?.selecting(batches: rows) }
            encoder.mask = encoder.mask?.gatheringRows(at: rows)
            selected.encoder = encoder
        }
        return selected
    }

    /// Returns the mask of the self attention of a step, broadcastable to [batchSize, heads, count, self.count + count],
    /// or nil when no position is masked.
    ///
    /// The padding of the step needs no mask: it follows the positions of its sequence, which do not attend to later positions.
    func selfAttentionMask(count newCount: Int) -> Tensor<Element, Device>? {
        let total = count + newCount
        // A new position does not attend to the new positions after it.
        func blocksNew(query: Int, key: Int) -> Bool {
            key >= count && key - count > query
        }
        guard hasPadding else {
            guard newCount > 1 else {
                return nil
            }
            let causal = (0 ..< newCount).flatMap { query in (0 ..< total).map { blocksNew(query: query, key: $0) ? Element(1) : Element(0) } }
            return Tensor(causal, shape: [1, 1, newCount, total])
        }
        let masked = padding.flatMap { sequence in
            (0 ..< newCount).flatMap { query in
                (0 ..< total).map { key in (key < count ? sequence[key] : blocksNew(query: query, key: key)) ? Element(1) : Element(0) }
            }
        }
        return Tensor(masked, shape: [batchSize, 1, newCount, total])
    }

    /// Records the positions of a step.
    mutating func advance(count newCount: Int, lengths newLengths: [Int]) {
        count += newCount
        for sequence in 0 ..< batchSize {
            lengths[sequence] += newLengths[sequence]
            padding[sequence] += (0 ..< newCount).map { $0 >= newLengths[sequence] }
        }
        hasPadding = hasPadding || newLengths.contains { $0 < newCount }
    }
}

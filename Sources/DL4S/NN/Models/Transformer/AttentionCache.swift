//
//  AttentionCache.swift
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

/// Keys and values of one attention layer, projected and split into heads, for autoregressive decoding.
///
/// A cache holds the keys and values of a batch of sequences at a number of positions. Every sequence of the batch has
/// the same number of positions, so positions that do not belong to a sequence are padding, which the caller masks.
/// ``MultiHeadAttention/cache(keys:values:)`` creates a cache, and
/// ``MultiHeadAttention/callAsFunction(queries:cache:mask:)`` attends to it.
public struct AttentionCache<Element: NumericType, Device: DeviceType>: Sendable {
    // Keys with the shape [batchSize, keyHeads, count, keyDim] and values with the shape
    // [batchSize, valueHeads, count, valueDim], the layout that scaledDotProductAttention reads.
    // The storage is not public, so that it can change without a change of the operations.
    let keys: Tensor<Element, Device>
    let values: Tensor<Element, Device>

    /// Number of sequences
    public var batchSize: Int {
        keys.shape[0]
    }

    /// Number of positions of every sequence
    public var count: Int {
        keys.shape[2]
    }

    init(keys: Tensor<Element, Device>, values: Tensor<Element, Device>) {
        precondition(keys.dim == 4 && values.dim == 4, "The keys and values of a cache must have 4 axes.")
        precondition(keys.shape[0] == values.shape[0] && keys.shape[2] == values.shape[2], "The keys and values of a cache must have the same batch size and count.")
        self.keys = keys
        self.values = values
    }

    /// Appends the positions of another cache to the positions of this cache.
    /// - Parameter other: Cache of the same layer with the same batch size
    /// - Returns: Cache with `count + other.count` positions
    public func appending(_ other: Self) -> Self {
        precondition(other.batchSize == batchSize, "Only caches with the same batch size can be appended.")
        return Self(keys: Tensor(stacking: [keys, other.keys], along: 2), values: Tensor(stacking: [values, other.values], along: 2))
    }

    /// Selects sequences of the batch, for example after a step of a beam search.
    /// - Parameter indices: Index of the sequence of this cache for each sequence of the result. Indices can repeat.
    /// - Returns: Cache with one sequence per index
    public func selecting(batches indices: Tensor<Int32, Device>) -> Self {
        Self(keys: keys.gatheringRows(at: indices), values: values.gatheringRows(at: indices))
    }

    /// Returns the first positions of every sequence.
    /// - Parameter count: Number of positions to keep, in `1 ... self.count`
    /// - Returns: Cache with `count` positions
    public func prefix(count: Int) -> Self {
        precondition(1 ... self.count ~= count, "The prefix must contain between 1 and \(self.count) positions.")
        guard count < self.count else {
            return self
        }
        return Self(keys: keys[nil, nil, 0 ..< count], values: values[nil, nil, 0 ..< count])
    }

    /// Joins the sequences of two caches into one batch.
    ///
    /// The cache with fewer positions is padded with zeros at the end. The caller must mask these positions.
    /// - Parameter other: Cache of the same layer
    /// - Returns: Cache with `batchSize + other.batchSize` sequences and `max(count, other.count)` positions
    public func merging(_ other: Self) -> Self {
        let count = Swift.max(count, other.count)
        let first = padded(to: count)
        let second = other.padded(to: count)
        return Self(keys: Tensor(stacking: [first.keys, second.keys], along: 0), values: Tensor(stacking: [first.values, second.values], along: 0))
    }

    private func padded(to count: Int) -> Self {
        guard count > self.count else {
            return self
        }
        func padding(of tensor: Tensor<Element, Device>) -> Tensor<Element, Device> {
            Tensor(repeating: 0, shape: [tensor.shape[0], tensor.shape[1], count - self.count, tensor.shape[3]])
        }
        return Self(keys: Tensor(stacking: [keys, padding(of: keys)], along: 2), values: Tensor(stacking: [values, padding(of: values)], along: 2))
    }
}

//
//  CPUFusedAttention.swift
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

// Attention runs one (batch, head) slice at a time. The attention weights of a slice live in scratch buffers of
// [queryCount, keyCount] elements, which are reused for every slice, and the matrix products write directly into
// the slices of the results. Multi-head attention uses `projectedAttention` and `projectedAttentionBackward` of the
// default implementation with the attention kernels of this file, whose backward pass also returns the result of the attention.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func scaledDotProductAttention<N: NumericType>(queries: ShapedBuffer<N, CPU>, keys: ShapedBuffer<N, CPU>, values: ShapedBuffer<N, CPU>, mask: ShapedBuffer<N, CPU>?, temperature: N, result: MutableShapedBuffer<N, CPU>) {
        let shape = AttentionShape(queries: queries, keys: keys, values: values)
        guard shape.hasOneBatchSize else {
            matrixProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
            return
        }
        precondition(result.shape == shape.resultShape, "The result must have the shape [batchSize, heads, queryCount, valueDim].")
        let attentionMask = AttentionMask(mask, shape: shape)
        let (q, k, v, y) = (queries.elementPointer, keys.elementPointer, values.elementPointer, result.elementPointer)
        let weights = UnsafeMutablePointer<N>.allocate(capacity: shape.queryCount * shape.keyCount)
        defer {
            weights.deallocate()
        }
        for slice in 0 ..< shape.slices {
            let operands = shape.slice(slice, queries: q, keys: k, values: v)
            shape.attentionWeights(queries: operands.queries, keys: operands.keys, mask: attentionMask.slice(slice, shape: shape), temperature: temperature, into: weights)
            CPUKernels.gemm(weights, shape: (shape.queryCount, shape.keyCount), operands.values, shape: (shape.keyCount, shape.valueDim), into: y + slice * shape.queryCount * shape.valueDim)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func scaledDotProductAttentionBackward<N: NumericType>(
        queries: ShapedBuffer<N, CPU>,
        keys: ShapedBuffer<N, CPU>,
        values: ShapedBuffer<N, CPU>,
        mask: ShapedBuffer<N, CPU>?,
        outputGradient: ShapedBuffer<N, CPU>,
        temperature: N,
        queryGradient: GradientBuffer<N, CPU>?,
        keyGradient: GradientBuffer<N, CPU>?,
        valueGradient: GradientBuffer<N, CPU>?,
    ) {
        let shape = AttentionShape(queries: queries, keys: keys, values: values)
        guard shape.hasOneBatchSize else {
            matrixProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
            return
        }
        attentionBackward(
            shape: shape,
            queries: queries,
            keys: keys,
            values: values,
            mask: mask,
            outputGradient: outputGradient,
            temperature: temperature,
            output: nil,
            queryGradient: queryGradient,
            keyGradient: keyGradient,
            valueGradient: valueGradient,
        )
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func multiHeadAttentionBackward<N: NumericType>(queries: ShapedBuffer<N, CPU>, keys: ShapedBuffer<N, CPU>, values: ShapedBuffer<N, CPU>, mask: ShapedBuffer<N, CPU>?, queryWeights: ShapedBuffer<N, CPU>, keyWeights: ShapedBuffer<N, CPU>, valueWeights: ShapedBuffer<N, CPU>, outputWeights: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, heads: Int, temperature: N, gradients: MultiHeadAttentionGradients<GradientBuffer<N, CPU>?>) {
        projectedAttentionBackward(
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            outputWeights: outputWeights,
            outputGradient: outputGradient,
            heads: heads,
            layout: .split,
            gradients: gradients,
        ) { queries, keys, values, outputGradient, output, queryGradient, keyGradient, valueGradient in
            // The heads of multi-head attention have one batch size. The backward kernel computes the attention weights of
            // every slice, so it writes the result of the attention as well.
            attentionBackward(
                shape: AttentionShape(queries: queries, keys: keys, values: values),
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                outputGradient: outputGradient,
                temperature: temperature,
                output: output?.elementPointer,
                queryGradient: queryGradient,
                keyGradient: keyGradient,
                valueGradient: valueGradient,
            )
        }
    }
}

extension CPUFusedOperations {
    /// Computes the gradients of scaled dot product attention, and its result when `output` is not nil. The queries, keys,
    /// and values must have the batch size of the result.
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func attentionBackward<N: NumericType>(
        shape: AttentionShape,
        queries: ShapedBuffer<N, CPU>,
        keys: ShapedBuffer<N, CPU>,
        values: ShapedBuffer<N, CPU>,
        mask: ShapedBuffer<N, CPU>?,
        outputGradient: ShapedBuffer<N, CPU>,
        temperature: N,
        output: UnsafeMutablePointer<N>?,
        queryGradient: GradientBuffer<N, CPU>?,
        keyGradient: GradientBuffer<N, CPU>?,
        valueGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(shape.hasOneBatchSize, "The attention kernels support queries, keys, and values with the batch size of the result.")
        precondition(outputGradient.shape == shape.resultShape, "The gradient of the result must have the shape of the result.")
        let attentionMask = AttentionMask(mask, shape: shape)
        let (q, k, v, g) = (queries.elementPointer, keys.elementPointer, values.elementPointer, outputGradient.elementPointer)
        // The products are added to the accumulated gradients directly: every slice of the query gradient is written once, and
        // the first query head of a group writes the slice of the key and value gradients that the other heads add to.
        let dq = queryGradient?.elementsToWrite()
        let dk = keyGradient?.elementsToWrite()
        let dv = valueGradient?.elementsToWrite()
        let (queryCount, keyCount, keyDim, valueDim) = (shape.queryCount, shape.keyCount, shape.keyDim, shape.valueDim)
        let matrixSize = queryCount * keyCount
        let weights = UnsafeMutablePointer<N>.allocate(capacity: matrixSize)
        let scoreGradient = UnsafeMutablePointer<N>.allocate(capacity: matrixSize)
        let products = UnsafeMutablePointer<N>.allocate(capacity: keyCount)
        defer {
            weights.deallocate()
            scoreGradient.deallocate()
            products.deallocate()
        }

        for slice in 0 ..< shape.slices {
            let operands = shape.slice(slice, queries: q, keys: k, values: v)
            let gradientSlice = g + slice * queryCount * valueDim
            // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
            shape.attentionWeights(queries: operands.queries, keys: operands.keys, mask: attentionMask.slice(slice, shape: shape), temperature: temperature, into: weights)
            if let output {
                CPUKernels.gemm(weights, shape: (queryCount, keyCount), operands.values, shape: (keyCount, valueDim), into: output + slice * queryCount * valueDim)
            }
            // The query heads of a group add to the gradients of their shared key head and value head.
            if let (dv, beta) = dv {
                let beta = shape.isFirstOfValueGroup(slice) ? beta : 1
                CPUKernels.gemm(weights, shape: (queryCount, keyCount), lhsTransposed: true, gradientSlice, shape: (queryCount, valueDim), into: dv + shape.valueSlice(of: slice) * keyCount * valueDim, beta: beta)
            }
            guard dq != nil || dk != nil else {
                continue
            }
            // The gradient of the scores is the gradient of the softmax, divided by the temperature.
            CPUKernels.gemm(gradientSlice, shape: (queryCount, valueDim), operands.values, shape: (keyCount, valueDim), rhsTransposed: true, into: scoreGradient)
            CPUKernels.softmaxRowsBackward(output: weights, outputGradient: scoreGradient, scale: 1 / temperature, into: scoreGradient, scratch: products, rows: queryCount, rowLength: keyCount)
            if let (dq, beta) = dq {
                CPUKernels.gemm(scoreGradient, shape: (queryCount, keyCount), operands.keys, shape: (keyCount, keyDim), into: dq + slice * queryCount * keyDim, beta: beta)
            }
            if let (dk, beta) = dk {
                let beta = shape.isFirstOfKeyGroup(slice) ? beta : 1
                CPUKernels.gemm(scoreGradient, shape: (queryCount, keyCount), lhsTransposed: true, operands.queries, shape: (queryCount, keyDim), into: dk + shape.keySlice(of: slice) * keyCount * keyDim, beta: beta)
            }
        }
    }
}

/// The queries, keys, and values of one (batch, query head) slice of scaled dot product attention.
struct AttentionSlice<N> {
    let queries: UnsafePointer<N>
    let keys: UnsafePointer<N>
    let values: UnsafePointer<N>
}

// The kernels process one (batch, query head) slice at a time.
extension AttentionShape {
    /// Whether the queries, keys, and values have the batch size of the result. The kernels support only these shapes.
    var hasOneBatchSize: Bool {
        [queryBatchSize, keyBatchSize, valueBatchSize].allSatisfy { $0 == batchSize }
    }

    /// Number of (batch, query head) slices.
    var slices: Int {
        batchSize * heads
    }

    /// Index of the (batch, key head) slice that a (batch, query head) slice uses.
    @inline(__always)
    func keySlice(of slice: Int) -> Int {
        let (batch, head) = slice.quotientAndRemainder(dividingBy: heads)
        return batch * keyHeads + head / (heads / keyHeads)
    }

    /// Index of the (batch, value head) slice that a (batch, query head) slice uses.
    @inline(__always)
    func valueSlice(of slice: Int) -> Int {
        let (batch, head) = slice.quotientAndRemainder(dividingBy: heads)
        return batch * valueHeads + head / (heads / valueHeads)
    }

    /// Whether the slice is the first query head of the group that shares its key head.
    @inline(__always)
    func isFirstOfKeyGroup(_ slice: Int) -> Bool {
        (slice % heads).isMultiple(of: heads / keyHeads)
    }

    /// Whether the slice is the first query head of the group that shares its value head.
    @inline(__always)
    func isFirstOfValueGroup(_ slice: Int) -> Bool {
        (slice % heads).isMultiple(of: heads / valueHeads)
    }

    /// The queries, keys, and values of one slice.
    @inline(__always)
    func slice<N>(_ slice: Int, queries: UnsafePointer<N>, keys: UnsafePointer<N>, values: UnsafePointer<N>) -> AttentionSlice<N> {
        AttentionSlice(queries: queries + slice * queryCount * keyDim, keys: keys + keySlice(of: slice) * keyCount * keyDim, values: values + valueSlice(of: slice) * keyCount * valueDim)
    }

    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` of one slice into a [queryCount, keyCount] matrix.
    @inline(__always)
    func attentionWeights<N: NumericType>(queries: UnsafePointer<N>, keys: UnsafePointer<N>, mask: AttentionMask<N>.Slice?, temperature: N, into weights: UnsafeMutablePointer<N>) {
        CPUKernels.gemm(queries, shape: (queryCount, keyDim), keys, shape: (keyCount, keyDim), rhsTransposed: true, into: weights, alpha: 1 / temperature)
        mask?.apply(to: weights, queryCount: queryCount, keyCount: keyCount)
        CPUKernels.softmaxRows(weights, into: weights, scratch: weights, rows: queryCount, rowLength: keyCount)
    }
}

/// A mask of scaled dot product attention that broadcasts to [batchSize, heads, queryCount, keyCount].
struct AttentionMask<N: NumericType> {
    /// The mask of one (batch, head) slice.
    struct Slice {
        let values: UnsafePointer<N>
        let queryStride: Int
        let keyStride: Int

        /// Subtracts 10⁹ times the mask from the scores, so that the softmax sets the blocked entries to 0.
        @inline(__always)
        func apply(to scores: UnsafeMutablePointer<N>, queryCount: Int, keyCount: Int) {
            let blocked = N(FusedConstants.maskScale)
            for row in 0 ..< queryCount {
                let (maskRow, rowScores) = (values + row * queryStride, scores + row * keyCount)
                if keyStride == 1 {
                    for column in 0 ..< keyCount {
                        rowScores[column] -= maskRow[column] * blocked
                    }
                } else {
                    // The mask has one value for all keys.
                    let value = maskRow[0] * blocked
                    for column in 0 ..< keyCount {
                        rowScores[column] -= value
                    }
                }
            }
        }
    }

    /// Elements of the mask, or nil without a mask
    let values: UnsafePointer<N>?
    /// Strides along the batch, head, query, and key axes. A broadcast axis has the stride 0.
    let strides: [Int]

    /// The mask with its strides. Without a mask, no score is blocked.
    ///
    /// The mask must be broadcastable to the scores, [batchSize, heads, queryCount, keyCount]. The pointer is valid while the mask is alive.
    init(_ mask: ShapedBuffer<N, CPU>?, shape: AttentionShape) {
        guard let mask else {
            values = nil
            strides = [0, 0, 0, 0]
            return
        }
        precondition(ShapeUtil.broadcasts(mask.shape, to: shape.scoreShape), "The mask must be broadcastable to the shape of the scores.")
        values = mask.elementPointer
        strides = ShapeUtil.broadcastStrides(Array(repeating: 1, count: 4 - mask.dim) + mask.shape)
    }

    /// The mask of a (batch, head) slice, or nil without a mask.
    @inline(__always)
    func slice(_ slice: Int, shape: AttentionShape) -> Slice? {
        guard let values else {
            return nil
        }
        let offset = (slice / shape.heads) * strides[0] + (slice % shape.heads) * strides[1]
        return Slice(values: values + offset, queryStride: strides[2], keyStride: strides[3])
    }
}

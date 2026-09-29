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
// the slices of the results. Multi-head attention uses the composed projections of the default implementation with
// the attention kernels of this file, whose backward pass also returns the result of the attention.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func scaledDotProductAttention<N: NumericType>(queries: ShapedBuffer<N, CPU>, keys: ShapedBuffer<N, CPU>, values: ShapedBuffer<N, CPU>, mask: ShapedBuffer<N, CPU>?, temperature: N, result: MutableShapedBuffer<N, CPU>) {
        guard let geometry = AttentionGeometry(queries: queries, keys: keys, values: values) else {
            DefaultFusedOperations<CPU>.scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: result)
            return
        }
        let attentionMask = AttentionMask(mask, geometry: geometry)
        let (q, k, v, y) = (queries.elementPointer, keys.elementPointer, values.elementPointer, result.elementPointer)
        let matrixSize = geometry.queryCount * geometry.keyCount
        let weights = UnsafeMutablePointer<N>.allocate(capacity: matrixSize)
        defer {
            weights.deallocate()
        }
        for slice in 0 ..< geometry.slices {
            let slices = geometry.slice(slice, queries: q, keys: k, values: v)
            geometry.attentionWeights(queries: slices.queries, keys: slices.keys, mask: attentionMask.slice(slice, geometry: geometry), temperature: temperature, into: weights)
            CPUKernels.gemm(weights, shape: (geometry.queryCount, geometry.keyCount), slices.values, shape: (geometry.keyCount, geometry.valueDim), into: y + slice * geometry.queryCount * geometry.valueDim)
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
        guard attentionBackward(
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
        ) else {
            DefaultFusedOperations<CPU>.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
            return
        }
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
            gradients: gradients,
        ) { queries, keys, values, outputGradient, output, queryGradient, keyGradient, valueGradient in
            // The backward kernel computes the attention weights of every slice, so it writes the result of the attention as well.
            let supported = attentionBackward(
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
            guard supported else {
                if let output {
                    scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature, result: output)
                }
                DefaultFusedOperations<CPU>.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, queryGradient: queryGradient, keyGradient: keyGradient, valueGradient: valueGradient)
                return
            }
        }
    }
}

extension CPUFusedOperations {
    /// Computes the gradients of scaled dot product attention, and its result when `output` is not nil.
    ///
    /// - Returns: False for shapes that the kernel does not support. It then does not change the gradients.
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func attentionBackward<N: NumericType>(
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
    ) -> Bool {
        guard let geometry = AttentionGeometry(queries: queries, keys: keys, values: values) else {
            return false
        }
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        let attentionMask = AttentionMask(mask, geometry: geometry)
        let (q, k, v, g) = (queries.elementPointer, keys.elementPointer, values.elementPointer, outputGradient.elementPointer)
        // Every slice of a gradient is written once, so the products are added to the accumulated gradients directly.
        let dq = queryGradient?.elementsToWrite()
        let dk = keyGradient?.elementsToWrite()
        let dv = valueGradient?.elementsToWrite()
        let (queryCount, keyCount, keyDim, valueDim) = (geometry.queryCount, geometry.keyCount, geometry.keyDim, geometry.valueDim)
        let matrixSize = queryCount * keyCount
        let weights = UnsafeMutablePointer<N>.allocate(capacity: matrixSize)
        let scoreGradient = UnsafeMutablePointer<N>.allocate(capacity: matrixSize)
        let products = UnsafeMutablePointer<N>.allocate(capacity: keyCount)
        defer {
            weights.deallocate()
            scoreGradient.deallocate()
            products.deallocate()
        }

        for slice in 0 ..< geometry.slices {
            let slices = geometry.slice(slice, queries: q, keys: k, values: v)
            let gradientSlice = g + slice * queryCount * valueDim
            // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
            geometry.attentionWeights(queries: slices.queries, keys: slices.keys, mask: attentionMask.slice(slice, geometry: geometry), temperature: temperature, into: weights)
            if let output {
                CPUKernels.gemm(weights, shape: (queryCount, keyCount), slices.values, shape: (keyCount, valueDim), into: output + slice * queryCount * valueDim)
            }
            if let (dv, beta) = dv {
                CPUKernels.gemm(weights, shape: (queryCount, keyCount), lhsTransposed: true, gradientSlice, shape: (queryCount, valueDim), into: dv + slice * keyCount * valueDim, beta: beta)
            }
            guard dq != nil || dk != nil else {
                continue
            }
            // The gradient of the scores is the gradient of the softmax, divided by the temperature.
            CPUKernels.gemm(gradientSlice, shape: (queryCount, valueDim), slices.values, shape: (keyCount, valueDim), rhsTransposed: true, into: scoreGradient)
            CPUKernels.softmaxRowsBackward(output: weights, outputGradient: scoreGradient, scale: 1 / temperature, into: scoreGradient, scratch: products, rows: queryCount, rowLength: keyCount)
            if let (dq, beta) = dq {
                CPUKernels.gemm(scoreGradient, shape: (queryCount, keyCount), slices.keys, shape: (keyCount, keyDim), into: dq + slice * queryCount * keyDim, beta: beta)
            }
            if let (dk, beta) = dk {
                CPUKernels.gemm(scoreGradient, shape: (queryCount, keyCount), lhsTransposed: true, slices.queries, shape: (queryCount, keyDim), into: dk + slice * keyCount * keyDim, beta: beta)
            }
        }
        return true
    }
}

/// Shapes of scaled dot product attention.
struct AttentionGeometry {
    let batchSize: Int
    let heads: Int
    let queryCount: Int
    let keyCount: Int
    let keyDim: Int
    let valueDim: Int

    /// Number of (batch, head) slices.
    var slices: Int {
        batchSize * heads
    }

    var outputShape: [Int] {
        [batchSize, heads, queryCount, valueDim]
    }

    /// The shapes of scaled dot product attention, or nil when the batch or head axes broadcast between the queries, keys,
    /// and values, which the kernels do not support.
    ///
    /// The arguments must have the shapes that ``FusedOperationsType/scaledDotProductAttention(queries:keys:values:mask:temperature:result:)`` states.
    init?<N>(queries: ShapedBuffer<N, CPU>, keys: ShapedBuffer<N, CPU>, values: ShapedBuffer<N, CPU>) {
        precondition(queries.dim == 4 && keys.dim == 4 && values.dim == 4, "The queries, keys, and values must have 4 axes.")
        precondition(queries.shape[3] == keys.shape[3], "The queries and the keys must have the same size.")
        precondition(keys.shape[2] == values.shape[2], "There must be one value for every key.")
        precondition(
            (0 ..< 2).allSatisfy { axis in Set([queries.shape[axis], keys.shape[axis], values.shape[axis]]).subtracting([1]).count <= 1 },
            "The batch and head axes of the queries, keys, and values must be broadcastable.",
        )
        guard queries.shape.prefix(2) == keys.shape.prefix(2), keys.shape.prefix(2) == values.shape.prefix(2) else {
            return nil
        }
        batchSize = queries.shape[0]
        heads = queries.shape[1]
        queryCount = queries.shape[2]
        keyCount = keys.shape[2]
        keyDim = queries.shape[3]
        valueDim = values.shape[3]
    }

    /// The queries, keys, and values of one slice.
    @inline(__always)
    func slice<N>(_ slice: Int, queries: UnsafePointer<N>, keys: UnsafePointer<N>, values: UnsafePointer<N>) -> (queries: UnsafePointer<N>, keys: UnsafePointer<N>, values: UnsafePointer<N>) {
        (queries + slice * queryCount * keyDim, keys + slice * keyCount * keyDim, values + slice * keyCount * valueDim)
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
            let blocked = N(1e9)
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
    let strides: (batch: Int, head: Int, query: Int, key: Int)

    /// The mask with its strides. Without a mask, no score is blocked.
    ///
    /// The mask must be broadcastable to the scores, [batchSize, heads, queryCount, keyCount]. The pointer is valid while the mask is alive.
    init(_ mask: ShapedBuffer<N, CPU>?, geometry: AttentionGeometry) {
        guard let mask else {
            values = nil
            strides = (0, 0, 0, 0)
            return
        }
        let target = [geometry.batchSize, geometry.heads, geometry.queryCount, geometry.keyCount]
        precondition(ShapeUtil.broadcasts(mask.shape, to: target), "The mask must be broadcastable to the shape of the scores.")
        let shape = Array(repeating: 1, count: 4 - mask.dim) + mask.shape
        var strides = [0, 0, 0, 0]
        var stride = 1
        for axis in (0 ..< 4).reversed() {
            strides[axis] = shape[axis] == 1 ? 0 : stride
            stride *= shape[axis]
        }
        values = mask.elementPointer
        self.strides = (strides[0], strides[1], strides[2], strides[3])
    }

    /// The mask of a (batch, head) slice, or nil without a mask.
    @inline(__always)
    func slice(_ slice: Int, geometry: AttentionGeometry) -> Slice? {
        guard let values else {
            return nil
        }
        let offset = (slice / geometry.heads) * strides.batch + (slice % geometry.heads) * strides.head
        return Slice(values: values + offset, queryStride: strides.query, keyStride: strides.key)
    }
}

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
// the slices of the results. Multi-head attention uses its default implementation, which calls these kernels.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func scaledDotProductAttention<N: NumericType>(queries: Tensor<N, CPU>, keys: Tensor<N, CPU>, values: Tensor<N, CPU>, mask: Tensor<N, CPU>?, temperature: N) -> Tensor<N, CPU> {
        guard let geometry = AttentionGeometry(queries: queries, keys: keys, values: values, mask: mask) else {
            return DefaultFusedOperations<CPU>.scaledDotProductAttention(queries: queries, keys: keys, values: values, mask: mask, temperature: temperature)
        }
        let (result, y) = CPUKernels.makeTensor(shape: [geometry.batchSize, geometry.heads, geometry.queryCount, geometry.valueDim]) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (q, k, v) = (queries.elementPointer, keys.elementPointer, values.elementPointer)
        let m = mask?.elementPointer
        let matrixSize = geometry.queryCount * geometry.keyCount

        CPUKernels.withScratch(N.self, count: 2 * matrixSize) { scratch in
            let (scores, weights) = (scratch, scratch + matrixSize)
            for slice in 0 ..< geometry.slices {
                geometry.attentionWeights(slice: slice, queries: q, keys: k, mask: m, temperature: temperature, scores: scores, into: weights)
                CPUKernels.gemm(
                    weights, shape: (geometry.queryCount, geometry.keyCount),
                    v + slice * geometry.keyCount * geometry.valueDim, shape: (geometry.keyCount, geometry.valueDim),
                    into: y + slice * geometry.queryCount * geometry.valueDim,
                )
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func scaledDotProductAttentionBackward<N: NumericType>(queries: Tensor<N, CPU>, keys: Tensor<N, CPU>, values: Tensor<N, CPU>, mask: Tensor<N, CPU>?, outputGradient: Tensor<N, CPU>, temperature: N, accumulating gradients: inout (queries: Tensor<N, CPU>?, keys: Tensor<N, CPU>?, values: Tensor<N, CPU>?)) {
        guard let geometry = AttentionGeometry(queries: queries, keys: keys, values: values, mask: mask),
              outputGradient.shape == [geometry.batchSize, geometry.heads, geometry.queryCount, geometry.valueDim]
        else {
            DefaultFusedOperations<CPU>.scaledDotProductAttentionBackward(queries: queries, keys: keys, values: values, mask: mask, outputGradient: outputGradient, temperature: temperature, accumulating: &gradients)
            return
        }
        let (q, k, v, g) = (queries.elementPointer, keys.elementPointer, values.elementPointer, outputGradient.elementPointer)
        let m = mask?.elementPointer
        // Every slice of a gradient is written once, so the products are added to the accumulated gradients directly.
        let queryGradient = queries.requiresGradient ? GradientTarget(taking: &gradients.queries, shape: queries.shape) : nil
        let keyGradient = keys.requiresGradient ? GradientTarget(taking: &gradients.keys, shape: keys.shape) : nil
        let valueGradient = values.requiresGradient ? GradientTarget(taking: &gradients.values, shape: values.shape) : nil
        let (queryCount, keyCount, keyDim, valueDim) = (geometry.queryCount, geometry.keyCount, geometry.keyDim, geometry.valueDim)
        let matrixSize = queryCount * keyCount
        let inverseTemperature = 1 / temperature

        CPUKernels.withScratch(N.self, count: 3 * matrixSize) { scratch in
            let (scores, weights, weightGradient) = (scratch, scratch + matrixSize, scratch + 2 * matrixSize)
            for slice in 0 ..< geometry.slices {
                let (querySlice, keySlice, valueSlice) = (q + slice * queryCount * keyDim, k + slice * keyCount * keyDim, v + slice * keyCount * valueDim)
                let gradientSlice = g + slice * queryCount * valueDim
                // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
                geometry.attentionWeights(slice: slice, queries: q, keys: k, mask: m, temperature: temperature, scores: scores, into: weights)

                if let valueGradient {
                    CPUKernels.gemm(weights, shape: (queryCount, keyCount), lhsTransposed: true, gradientSlice, shape: (queryCount, valueDim), into: valueGradient.pointer + slice * keyCount * valueDim, beta: valueGradient.beta)
                }
                guard queryGradient != nil || keyGradient != nil else {
                    continue
                }
                CPUKernels.gemm(gradientSlice, shape: (queryCount, valueDim), valueSlice, shape: (keyCount, valueDim), rhsTransposed: true, into: weightGradient)
                // The gradient of the softmax, divided by the temperature: dS = P * (dP - sum(dP * P)) / temperature
                for i in 0 ..< matrixSize {
                    scores[i] = weightGradient[i] * weights[i]
                }
                for row in 0 ..< queryCount {
                    let start = row * keyCount
                    let product = CPUKernels.sum(scores + start, count: keyCount)
                    for j in start ..< start + keyCount {
                        weightGradient[j] = weights[j] * (weightGradient[j] - product) * inverseTemperature
                    }
                }
                if let queryGradient {
                    CPUKernels.gemm(weightGradient, shape: (queryCount, keyCount), keySlice, shape: (keyCount, keyDim), into: queryGradient.pointer + slice * queryCount * keyDim, beta: queryGradient.beta)
                }
                if let keyGradient {
                    CPUKernels.gemm(weightGradient, shape: (queryCount, keyCount), lhsTransposed: true, querySlice, shape: (queryCount, keyDim), into: keyGradient.pointer + slice * keyCount * keyDim, beta: keyGradient.beta)
                }
            }
        }
        queryGradient?.finish(into: &gradients.queries)
        keyGradient?.finish(into: &gradients.keys)
        valueGradient?.finish(into: &gradients.values)
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
    /// Strides of the mask along the batch, head, query, and key axes. A broadcast axis has the stride 0.
    let maskStrides: (batch: Int, head: Int, query: Int, key: Int)?

    /// Number of (batch, head) slices.
    var slices: Int {
        batchSize * heads
    }

    /// Returns nil for shapes that the kernels do not support, such as broadcasting between the queries, keys, and values.
    init?<N>(queries: Tensor<N, CPU>, keys: Tensor<N, CPU>, values: Tensor<N, CPU>, mask: Tensor<N, CPU>?) {
        guard queries.dim == 4, keys.dim == 4, values.dim == 4, queries.count > 0, keys.count > 0, values.count > 0,
              queries.shape.prefix(2) == keys.shape.prefix(2), keys.shape.prefix(2) == values.shape.prefix(2),
              queries.shape[3] == keys.shape[3], keys.shape[2] == values.shape[2]
        else {
            return nil
        }
        batchSize = queries.shape[0]
        heads = queries.shape[1]
        queryCount = queries.shape[2]
        keyCount = keys.shape[2]
        keyDim = queries.shape[3]
        valueDim = values.shape[3]

        guard let mask else {
            maskStrides = nil
            return
        }
        let target = [batchSize, heads, queryCount, keyCount]
        guard mask.dim <= 4, mask.count > 0 else {
            return nil
        }
        let shape = Array(repeating: 1, count: 4 - mask.dim) + mask.shape
        guard zip(shape, target).allSatisfy({ $0 == $1 || $0 == 1 }) else {
            return nil
        }
        var strides = [0, 0, 0, 0]
        var stride = 1
        for axis in (0 ..< 4).reversed() {
            strides[axis] = shape[axis] == 1 ? 0 : stride
            stride *= shape[axis]
        }
        maskStrides = (strides[0], strides[1], strides[2], strides[3])
    }

    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` of one slice. `scores` and `weights` hold queryCount \* keyCount elements.
    @inline(__always)
    func attentionWeights<N: NumericType>(
        slice: Int,
        queries: UnsafePointer<N>,
        keys: UnsafePointer<N>,
        mask: UnsafePointer<N>?,
        temperature: N,
        scores: UnsafeMutablePointer<N>,
        into weights: UnsafeMutablePointer<N>,
    ) {
        CPUKernels.gemm(
            queries + slice * queryCount * keyDim, shape: (queryCount, keyDim),
            keys + slice * keyCount * keyDim, shape: (keyCount, keyDim), rhsTransposed: true,
            into: scores,
            alpha: 1 / temperature,
        )
        if let mask, let strides = maskStrides {
            // The mask contains 1 for every entry that is blocked, so the softmax sets these entries to 0.
            let blocked = N(1e9)
            let sliceOffset = (slice / heads) * strides.batch + (slice % heads) * strides.head
            for row in 0 ..< queryCount {
                let maskRow = mask + (sliceOffset + row * strides.query)
                let rowScores = scores + row * keyCount
                if strides.key == 1 {
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
        CPUFusedOperations.subtractRowMaxima(scores, into: scores, rows: queryCount, rowLength: keyCount)
        CPUKernels.exp(scores, into: weights, count: queryCount * keyCount)
        for row in 0 ..< queryCount {
            let rowWeights = weights + row * keyCount
            let inverseSum = 1 / CPUKernels.sum(rowWeights, count: keyCount)
            for column in 0 ..< keyCount {
                rowWeights[column] *= inverseSum
            }
        }
    }
}

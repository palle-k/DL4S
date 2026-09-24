//
//  FusedAttention.swift
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

// MARK: Default implementations

public extension FusedOperationsType {
    static func scaledDotProductAttention<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, temperature: N) -> Tensor<N, Device> {
        Composed.attentionWeights(queries: queries.detached(), keys: keys.detached(), mask: mask?.detached(), temperature: temperature)
            .broadcastMatrixMultiplied(with: values.detached())
    }

    static func scaledDotProductAttentionBackward<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, temperature: N, accumulating gradients: inout (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?)) {
        let computed = Composed.scaledDotProductAttentionGradients(
            queries: queries.detached(),
            keys: keys.detached(),
            values: values.detached(),
            mask: mask?.detached(),
            outputGradient: outputGradient.detached(),
            temperature: temperature,
            computesQueries: queries.requiresGradient,
            computesKeys: keys.requiresGradient,
            computesValues: values.requiresGradient,
        )
        Tensor.accumulate(computed.queries, into: &gradients.queries)
        Tensor.accumulate(computed.keys, into: &gradients.keys)
        Tensor.accumulate(computed.values, into: &gradients.values)
    }

    static func multiHeadAttention<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, queryWeights: Tensor<N, Device>, keyWeights: Tensor<N, Device>, valueWeights: Tensor<N, Device>, outputWeights: Tensor<N, Device>, heads: Int, temperature: N) -> Tensor<N, Device> {
        let projectedQueries = Composed.splitHeads(Composed.project(queries.detached(), with: queryWeights.detached()), heads: heads)
        let projectedKeys = Composed.splitHeads(Composed.project(keys.detached(), with: keyWeights.detached()), heads: heads)
        let projectedValues = Composed.splitHeads(Composed.project(values.detached(), with: valueWeights.detached()), heads: heads)

        let attended = Device.FusedOperations.scaledDotProductAttention(queries: projectedQueries, keys: projectedKeys, values: projectedValues, mask: mask?.detached(), temperature: temperature)
        return Composed.project(Composed.joinHeads(attended), with: outputWeights.detached())
    }

    static func multiHeadAttentionBackward<N: NumericType>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, values: Tensor<N, Device>, mask: Tensor<N, Device>?, queryWeights: Tensor<N, Device>, keyWeights: Tensor<N, Device>, valueWeights: Tensor<N, Device>, outputWeights: Tensor<N, Device>, outputGradient: Tensor<N, Device>, heads: Int, temperature: N, accumulating gradients: inout MultiHeadAttentionGradients<N, Device>) {
        let computes = [queries, keys, values, queryWeights, keyWeights, valueWeights, outputWeights].map(\.requiresGradient)
        let (queries, keys, values) = (queries.detached(), keys.detached(), values.detached())
        let outputGradient = outputGradient.detached()
        var queryHeads = Composed.splitHeads(Composed.project(queries, with: queryWeights.detached()), heads: heads)
        var keyHeads = Composed.splitHeads(Composed.project(keys, with: keyWeights.detached()), heads: heads)
        var valueHeads = Composed.splitHeads(Composed.project(values, with: valueWeights.detached()), heads: heads)

        var computed = MultiHeadAttentionGradients<N, Device>(queries: nil, keys: nil, values: nil, queryWeights: nil, keyWeights: nil, valueWeights: nil, outputWeights: nil)
        if computes[6] {
            let attended = Device.FusedOperations.scaledDotProductAttention(queries: queryHeads, keys: keyHeads, values: valueHeads, mask: mask?.detached(), temperature: temperature)
            computed.outputWeights = Composed.projectionWeightGradient(input: Composed.joinHeads(attended), outputGradient: outputGradient)
        }

        // The flags select the gradients that the fused attention of the device computes.
        queryHeads.requiresGradient = computes[0] || computes[3]
        keyHeads.requiresGradient = computes[1] || computes[4]
        valueHeads.requiresGradient = computes[2] || computes[5]
        var headGradients: (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?) = (nil, nil, nil)
        Device.FusedOperations.scaledDotProductAttentionBackward(
            queries: queryHeads,
            keys: keyHeads,
            values: valueHeads,
            mask: mask?.detached(),
            outputGradient: Composed.splitHeads(Composed.projectionInputGradient(outputGradient, weights: outputWeights.detached()), heads: heads),
            temperature: temperature,
            accumulating: &headGradients,
        )
        Composed.addProjectionGradients(
            to: &computed,
            headGradients: headGradients,
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights.detached(),
            keyWeights: keyWeights.detached(),
            valueWeights: valueWeights.detached(),
            computes: computes,
        )
        gradients.accumulate(computed)
    }

    static func positionalEncoding<N: NumericType>(length: Int, hiddenSize: Int) -> Tensor<N, Device> {
        let positions = Tensor<N, Device>((0 ..< length).map(N.init))
        let frequencyIndices = Tensor<N, Device>((0 ..< hiddenSize / 2).map(N.init))
        let frequencies = Tensor<N, Device>(10000).raised(toPowerOf: frequencyIndices / Tensor(N(hiddenSize / 2)))
        let samplePoints = positions.unsqueezed(at: 1) / frequencies.unsqueezed(at: 0) // [length, hiddenSize / 2]

        return Tensor(
            stacking: [
                samplePoints.sine().unsqueezed(at: 2),
                samplePoints.cosine().unsqueezed(at: 2),
            ],
            along: 2,
        ).view(as: [-1, hiddenSize])
    }
}

// MARK: Composed gradients

extension Composed {
    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` along the last axis, shape [batchSize, heads, queryCount, keyCount].
    static func attentionWeights<N, Device>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, mask: Tensor<N, Device>?, temperature: N) -> Tensor<N, Device> {
        var scores = (queries / Tensor(temperature)).broadcastMatrixMultiplied(with: keys, transposeOther: true)
        if let mask {
            // The mask contains 1 for every entry that is blocked, so the softmax sets these entries to 0.
            scores -= mask * 1e9
        }
        return scores.softmax(axis: 3)
    }

    static func scaledDotProductAttentionGradients<N, Device>(
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        mask: Tensor<N, Device>?,
        outputGradient: Tensor<N, Device>,
        temperature: N,
        computesQueries: Bool,
        computesKeys: Bool,
        computesValues: Bool,
    ) -> (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?) {
        // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
        scaledDotProductAttentionGradients(
            weights: attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature),
            queries: queries,
            keys: keys,
            values: values,
            outputGradient: outputGradient,
            temperature: temperature,
            computesQueries: computesQueries,
            computesKeys: computesKeys,
            computesValues: computesValues,
        )
    }

    /// Computes the gradients of scaled dot product attention from the attention weights of the forward pass.
    static func scaledDotProductAttentionGradients<N, Device>(
        weights: Tensor<N, Device>,
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        temperature: N,
        computesQueries: Bool,
        computesKeys: Bool,
        computesValues: Bool,
    ) -> (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?) {
        let valueGradient = computesValues ? weights
            .broadcastMatrixMultiplied(with: outputGradient, transposeSelf: true)
            .reducingBroadcast(to: values.shape) : nil

        guard computesQueries || computesKeys else {
            return (nil, nil, valueGradient)
        }
        let weightGradient = outputGradient.broadcastMatrixMultiplied(with: values, transposeOther: true)
        let scaledScoreGradient = softmaxGradient(output: weights, outputGradient: weightGradient, axis: 3) / Tensor(temperature)

        let queryGradient = computesQueries ? scaledScoreGradient
            .broadcastMatrixMultiplied(with: keys)
            .reducingBroadcast(to: queries.shape) : nil
        let keyGradient = computesKeys ? scaledScoreGradient
            .broadcastMatrixMultiplied(with: queries, transposeSelf: true)
            .reducingBroadcast(to: keys.shape) : nil
        return (queryGradient, keyGradient, valueGradient)
    }

    /// Multiplies every vector of a [batchSize, count, inputSize] tensor with weights of the shape [inputSize, outputSize].
    static func project<N, Device>(_ input: Tensor<N, Device>, with weights: Tensor<N, Device>) -> Tensor<N, Device> {
        input
            .view(as: [-1, input.shape[2]])
            .matrixMultiplied(with: weights)
            .view(as: [input.shape[0], input.shape[1], weights.shape[1]])
    }

    /// Computes the gradient of the input of ``project(_:with:)``.
    static func projectionInputGradient<N, Device>(_ outputGradient: Tensor<N, Device>, weights: Tensor<N, Device>) -> Tensor<N, Device> {
        project(outputGradient, with: weights.transposed())
    }

    /// Computes the gradient of the weights of ``project(_:with:)``.
    static func projectionWeightGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>) -> Tensor<N, Device> {
        input
            .view(as: [-1, input.shape[2]])
            .matrixMultiplied(with: outputGradient.view(as: [-1, outputGradient.shape[2]]), transposeSelf: true)
    }

    /// Splits [batchSize, count, heads \* size] into [batchSize, heads, count, size].
    static func splitHeads<N, Device>(_ input: Tensor<N, Device>, heads: Int) -> Tensor<N, Device> {
        input
            .view(as: [input.shape[0], input.shape[1], heads, -1])
            .permuted(to: [0, 2, 1, 3])
    }

    /// Joins [batchSize, heads, count, size] into [batchSize, count, heads \* size].
    static func joinHeads<N, Device>(_ input: Tensor<N, Device>) -> Tensor<N, Device> {
        input
            .permuted(to: [0, 2, 1, 3])
            .view(as: [input.shape[0], input.shape[2], -1])
    }

    /// Computes the gradients of multi-head attention.
    ///
    /// The flags in `computes` select the gradients in the order queries, keys, values, query weights, key weights, value weights, and output weights.
    static func multiHeadAttentionGradients<N, Device>(
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        mask: Tensor<N, Device>?,
        queryWeights: Tensor<N, Device>,
        keyWeights: Tensor<N, Device>,
        valueWeights: Tensor<N, Device>,
        outputWeights: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        heads: Int,
        temperature: N,
        computes: [Bool],
    ) -> MultiHeadAttentionGradients<N, Device> {
        let queryHeads = splitHeads(project(queries, with: queryWeights), heads: heads)
        let keyHeads = splitHeads(project(keys, with: keyWeights), heads: heads)
        let valueHeads = splitHeads(project(values, with: valueWeights), heads: heads)
        let weights = attentionWeights(queries: queryHeads, keys: keyHeads, mask: mask, temperature: temperature)

        var gradients = MultiHeadAttentionGradients<N, Device>(queries: nil, keys: nil, values: nil, queryWeights: nil, keyWeights: nil, valueWeights: nil, outputWeights: nil)

        if computes[6] {
            let joined = joinHeads(weights.broadcastMatrixMultiplied(with: valueHeads))
            gradients.outputWeights = projectionWeightGradient(input: joined, outputGradient: outputGradient)
        }

        let (queryHeadGradient, keyHeadGradient, valueHeadGradient) = scaledDotProductAttentionGradients(
            weights: weights,
            queries: queryHeads,
            keys: keyHeads,
            values: valueHeads,
            outputGradient: splitHeads(projectionInputGradient(outputGradient, weights: outputWeights), heads: heads),
            temperature: temperature,
            computesQueries: computes[0] || computes[3],
            computesKeys: computes[1] || computes[4],
            computesValues: computes[2] || computes[5],
        )

        addProjectionGradients(
            to: &gradients,
            headGradients: (queryHeadGradient, keyHeadGradient, valueHeadGradient),
            queries: queries,
            keys: keys,
            values: values,
            queryWeights: queryWeights,
            keyWeights: keyWeights,
            valueWeights: valueWeights,
            computes: computes,
        )
        return gradients
    }

    /// Adds the gradients of the queries, keys, values, and their projections, given the gradients of the heads.
    static func addProjectionGradients<N, Device>(
        to gradients: inout MultiHeadAttentionGradients<N, Device>,
        headGradients: (queries: Tensor<N, Device>?, keys: Tensor<N, Device>?, values: Tensor<N, Device>?),
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        queryWeights: Tensor<N, Device>,
        keyWeights: Tensor<N, Device>,
        valueWeights: Tensor<N, Device>,
        computes: [Bool],
    ) {
        if let queryHeadGradient = headGradients.queries {
            let projectedGradient = joinHeads(queryHeadGradient)
            gradients.queries = computes[0] ? projectionInputGradient(projectedGradient, weights: queryWeights) : nil
            gradients.queryWeights = computes[3] ? projectionWeightGradient(input: queries, outputGradient: projectedGradient) : nil
        }
        if let keyHeadGradient = headGradients.keys {
            let projectedGradient = joinHeads(keyHeadGradient)
            gradients.keys = computes[1] ? projectionInputGradient(projectedGradient, weights: keyWeights) : nil
            gradients.keyWeights = computes[4] ? projectionWeightGradient(input: keys, outputGradient: projectedGradient) : nil
        }
        if let valueHeadGradient = headGradients.values {
            let projectedGradient = joinHeads(valueHeadGradient)
            gradients.values = computes[2] ? projectionInputGradient(projectedGradient, weights: valueWeights) : nil
            gradients.valueWeights = computes[5] ? projectionWeightGradient(input: values, outputGradient: projectedGradient) : nil
        }
    }
}

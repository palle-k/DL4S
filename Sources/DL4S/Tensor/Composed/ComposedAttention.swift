//
//  ComposedAttention.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.09.26.
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

extension Composed {
    /// Computes `softmax(queries × keysᵀ / temperature - 10⁹ * mask)` along the last axis, shape [batchSize, heads, queryCount, keyCount].
    static func attentionWeights<N, Device>(queries: Tensor<N, Device>, keys: Tensor<N, Device>, mask: Tensor<N, Device>?, temperature: N) -> Tensor<N, Device> {
        var scores = grouped(queries / Tensor(temperature), heads: keys.shape[1])
            .broadcastMatrixMultiplied(with: keys, transposeOther: true)
            .view(as: [-1, queries.shape[1], queries.shape[2], keys.shape[2]])
        if let mask {
            // The mask contains 1 for every entry that is blocked, so the softmax sets these entries to 0.
            scores -= mask * Tensor(N(FusedConstants.maskScale))
        }
        return scores.softmax(axis: 3)
    }

    static func scaledDotProductAttentionBackward<N, Device>(
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        mask: Tensor<N, Device>?,
        outputGradient: Tensor<N, Device>,
        temperature: N,
        queryGradient: inout GradientAccumulator<N, Device>,
        keyGradient: inout GradientAccumulator<N, Device>,
        valueGradient: inout GradientAccumulator<N, Device>,
    ) {
        // The attention weights are computed again instead of being kept alive between the forward and the backward pass.
        attentionBackward(
            weights: attentionWeights(queries: queries, keys: keys, mask: mask, temperature: temperature),
            queries: queries,
            keys: keys,
            values: values,
            outputGradient: outputGradient,
            temperature: temperature,
            queryGradient: &queryGradient,
            keyGradient: &keyGradient,
            valueGradient: &valueGradient,
        )
    }

    /// Computes the gradients of scaled dot product attention from the attention weights of the forward pass.
    private static func attentionBackward<N, Device>(
        weights: Tensor<N, Device>,
        queries: Tensor<N, Device>,
        keys: Tensor<N, Device>,
        values: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        temperature: N,
        queryGradient: inout GradientAccumulator<N, Device>,
        keyGradient: inout GradientAccumulator<N, Device>,
        valueGradient: inout GradientAccumulator<N, Device>,
    ) {
        let (keyHeads, valueHeads) = (keys.shape[1], values.shape[1])
        let groupedOutputGradient = grouped(outputGradient, heads: valueHeads)
        if valueGradient.isRequested {
            valueGradient.add(grouped(weights, heads: valueHeads).broadcastMatrixMultiplied(with: groupedOutputGradient, transposeSelf: true).reducingBroadcast(to: values.shape))
        }
        guard queryGradient.isRequested || keyGradient.isRequested else {
            return
        }
        let weightGradient = groupedOutputGradient.broadcastMatrixMultiplied(with: values, transposeOther: true).view(as: weights.shape)
        let scaledScoreGradient = grouped(softmaxGradient(output: weights, outputGradient: weightGradient, axis: 3) / Tensor(temperature), heads: keyHeads)
        if queryGradient.isRequested {
            let gradient = scaledScoreGradient.broadcastMatrixMultiplied(with: keys).view(as: [-1, queries.shape[1], queries.shape[2], queries.shape[3]])
            queryGradient.add(gradient.reducingBroadcast(to: queries.shape))
        }
        if keyGradient.isRequested {
            keyGradient.add(scaledScoreGradient.broadcastMatrixMultiplied(with: grouped(queries, heads: keyHeads), transposeSelf: true).reducingBroadcast(to: keys.shape))
        }
    }

    /// Folds the query heads that share one of `heads` key or value heads into the rows:
    /// [n, queryHeads, rows, columns] becomes [n, heads, queryHeads / heads \* rows, columns].
    private static func grouped<N, Device>(_ input: Tensor<N, Device>, heads: Int) -> Tensor<N, Device> {
        input.view(as: [input.shape[0], heads, -1, input.shape[3]])
    }

    /// Multiplies every vector of a [batchSize, count, inputSize] tensor with weights of the shape [inputSize, outputSize].
    static func project<N, Device>(_ input: Tensor<N, Device>, with weights: Tensor<N, Device>) -> Tensor<N, Device> {
        input
            .view(as: [-1, input.shape[2]])
            .matrixMultiplied(with: weights)
            .view(as: [input.shape[0], input.shape[1], weights.shape[1]])
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

    /// Computes the gradients of multi-head attention. The result of the attention of the heads and its gradients share the attention weights.
    static func multiHeadAttentionBackward<N, Device>(
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
        gradients: inout MultiHeadAttentionGradients<GradientAccumulator<N, Device>>,
    ) {
        let shape = MultiHeadAttentionShape(
            queries: queries.values,
            keys: keys.values,
            values: values.values,
            queryWeights: queryWeights.values,
            keyWeights: keyWeights.values,
            valueWeights: valueWeights.values,
            outputWeights: outputWeights.values,
            heads: heads,
        )
        let queryHeads = splitHeads(project(queries, with: queryWeights), heads: heads)
        let keyHeads = splitHeads(project(keys, with: keyWeights), heads: shape.keyHeads)
        let valueHeads = splitHeads(project(values, with: valueWeights), heads: shape.keyHeads)

        var queryHeadGradient = GradientAccumulator<N, Device>(isRequested: gradients.queries.isRequested || gradients.queryWeights.isRequested, shape: queryHeads.shape)
        var keyHeadGradient = GradientAccumulator<N, Device>(isRequested: gradients.keys.isRequested || gradients.keyWeights.isRequested, shape: keyHeads.shape)
        var valueHeadGradient = GradientAccumulator<N, Device>(isRequested: gradients.values.isRequested || gradients.valueWeights.isRequested, shape: valueHeads.shape)
        let weights = attentionWeights(queries: queryHeads, keys: keyHeads, mask: mask, temperature: temperature)
        attentionBackward(
            weights: weights,
            queries: queryHeads,
            keys: keyHeads,
            values: valueHeads,
            outputGradient: splitHeads(outputGradient.view(as: [-1, outputGradient.shape[2]]).matrixMultiplied(with: outputWeights, transposeOther: true).view(as: [outputGradient.shape[0], outputGradient.shape[1], -1]), heads: heads),
            temperature: temperature,
            queryGradient: &queryHeadGradient,
            keyGradient: &keyHeadGradient,
            valueGradient: &valueHeadGradient,
        )
        if gradients.outputWeights.isRequested {
            let attended = grouped(weights, heads: shape.keyHeads).broadcastMatrixMultiplied(with: valueHeads).view(as: shape.attention.resultShape)
            addProjectionWeightGradient(input: joinHeads(attended), outputGradient: outputGradient, to: &gradients.outputWeights)
        }
        if let headGradient = queryHeadGradient.value.map(joinHeads) {
            addProjectionInputGradient(outputGradient: headGradient, weights: queryWeights, to: &gradients.queries)
            addProjectionWeightGradient(input: queries, outputGradient: headGradient, to: &gradients.queryWeights)
        }
        if let headGradient = keyHeadGradient.value.map(joinHeads) {
            addProjectionInputGradient(outputGradient: headGradient, weights: keyWeights, to: &gradients.keys)
            addProjectionWeightGradient(input: keys, outputGradient: headGradient, to: &gradients.keyWeights)
        }
        if let headGradient = valueHeadGradient.value.map(joinHeads) {
            addProjectionInputGradient(outputGradient: headGradient, weights: valueWeights, to: &gradients.values)
            addProjectionWeightGradient(input: values, outputGradient: headGradient, to: &gradients.valueWeights)
        }
    }

    /// Adds the gradient of the input of ``project(_:with:)`` when it is requested.
    private static func addProjectionInputGradient<N, Device>(outputGradient: Tensor<N, Device>, weights: Tensor<N, Device>, to inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(
            outputGradient
                .view(as: [-1, outputGradient.shape[2]])
                .matrixMultiplied(with: weights, transposeOther: true)
                .view(as: [outputGradient.shape[0], outputGradient.shape[1], weights.shape[0]]),
        )
    }

    /// Adds the gradient of the weights of ``project(_:with:)`` when it is requested.
    private static func addProjectionWeightGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, to weightGradient: inout GradientAccumulator<N, Device>) {
        guard weightGradient.isRequested else {
            return
        }
        weightGradient.add(input.view(as: [-1, input.shape[2]]).matrixMultiplied(with: outputGradient.view(as: [-1, outputGradient.shape[2]]), transposeSelf: true))
    }
}

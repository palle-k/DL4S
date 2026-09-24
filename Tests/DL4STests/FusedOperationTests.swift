//
//  FusedOperationTests.swift
//  DL4STests
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

import DL4S
import Foundation
import Testing

private typealias DoubleTensor = Tensor<Double, CPU>

/// A fused operation, a reference implementation of the same operation, and the values of its sources.
///
/// The reference implementations are composed from basic tensor operations, as the operations were before they had fused requirements.
/// The test points are away from the kinks of piecewise differentiable operations, so every operation is twice differentiable there.
struct FusedOperationCase: CustomTestStringConvertible, Sendable {
    let name: String
    /// Values of the sources, in the order that the operations take them.
    fileprivate let sources: [DoubleTensor]
    /// Positions of the sources that the gradients are checked for. The other sources are constants.
    let differentiable: [Int]
    fileprivate let fused: @Sendable ([DoubleTensor]) -> DoubleTensor
    fileprivate let reference: @Sendable ([DoubleTensor]) -> DoubleTensor

    var testDescription: String {
        name
    }

    fileprivate init(
        _ name: String,
        sources: [DoubleTensor],
        differentiable: [Int]? = nil,
        fused: @escaping @Sendable ([DoubleTensor]) -> DoubleTensor,
        reference: @escaping @Sendable ([DoubleTensor]) -> DoubleTensor,
    ) {
        self.name = name
        self.sources = sources
        self.differentiable = differentiable ?? Array(sources.indices)
        self.fused = fused
        self.reference = reference
    }

    /// Copies of the sources. The differentiable sources require a gradient.
    fileprivate func makeSources() -> [DoubleTensor] {
        sources.enumerated().map { index, source in
            var copy = source.detached()
            copy.requiresGradient = differentiable.contains(index)
            return copy
        }
    }
}

// MARK: Test values

private func uniform(_ shape: [Int], min: Double = -1, max: Double = 1, seed: UInt64) -> DoubleTensor {
    var generator = WyHash(seed: seed)
    return DoubleTensor(uniformlyDistributedWithShape: shape, min: min, max: max, using: &generator)
}

/// Uniform values whose magnitudes are at least 0.1, so that no value is close to the kink of a piecewise function.
private func awayFromZero(_ shape: [Int], seed: UInt64) -> DoubleTensor {
    let values = uniform(shape, min: 0.1, max: 1, seed: seed)
    let signs = uniform(shape, seed: seed + 1).elements.map { $0 < 0 ? -1.0 : 1.0 }
    return values * DoubleTensor(signs, shape: shape)
}

// MARK: Reference implementations

/// Mean along the axes, composed from a sum and a division.
private func referenceMean(_ tensor: DoubleTensor, along axes: [Int]) -> DoubleTensor {
    tensor.reduceSum(along: axes) / DoubleTensor(Double(axes.map { tensor.shape[$0] }.reduce(1, *)))
}

private func referenceVariance(_ tensor: DoubleTensor, along axes: [Int]) -> DoubleTensor {
    let mean = referenceMean(tensor, along: axes)
    return referenceMean(tensor * tensor, along: axes) - mean * mean
}

private func referenceSigmoid(_ tensor: DoubleTensor) -> DoubleTensor {
    0.5 * (tensor * 0.5).tanh() + 0.5
}

private func referenceSoftmax(_ tensor: DoubleTensor, axis: Int) -> DoubleTensor {
    let normalizer = tensor.detached().reduceMax(along: [axis]).unsqueezed(at: axis)
    let exponentiated = (tensor - normalizer).exp()
    return exponentiated / exponentiated.reduceSum(along: [axis]).unsqueezed(at: axis)
}

private func referenceLogSoftmax(_ tensor: DoubleTensor, axis: Int) -> DoubleTensor {
    let normalized = tensor - tensor.detached().reduceMax(along: [axis]).unsqueezed(at: axis)
    return normalized - normalized.exp().reduceSum(along: [axis]).log().unsqueezed(at: axis)
}

private func referenceConvolution(_ input: DoubleTensor, filters: DoubleTensor, padding: Int, stride: Int) -> DoubleTensor {
    let outputShape = [
        input.shape[0],
        filters.shape[0],
        (input.shape[2] + 2 * padding - filters.shape[2]) / stride + 1,
        (input.shape[3] + 2 * padding - filters.shape[3]) / stride + 1,
    ]
    let columns = input.img2col(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
    return filters
        .view(as: [filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3]])
        .matrixMultiplied(with: columns)
        .view(as: [outputShape[1], outputShape[0], outputShape[2], outputShape[3]])
        .permuted(to: [1, 0, 2, 3])
}

private func referenceTransposedConvolution(_ input: DoubleTensor, filters: DoubleTensor, inset: Int, stride: Int) -> DoubleTensor {
    let outputShape = [
        input.shape[0],
        filters.shape[0],
        (input.shape[2] - 1) * stride - 2 * inset + filters.shape[2],
        (input.shape[3] - 1) * stride - 2 * inset + filters.shape[3],
    ]
    let inputMatrix = input.permuted(to: [1, 0, 2, 3]).view(as: [input.shape[1], input.shape[0] * input.shape[2] * input.shape[3]])
    return filters
        .view(as: [filters.shape[1], filters.shape[0] * filters.shape[2] * filters.shape[3]])
        .transposed()
        .matrixMultiplied(with: inputMatrix)
        .col2img(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: inset, stride: stride, resultShape: outputShape)
}

private func referencePooling(_ input: DoubleTensor, windowSize: Int, padding: Int, stride: Int, reduce: (DoubleTensor) -> DoubleTensor) -> DoubleTensor {
    let outputShape = [
        input.shape[0],
        input.shape[1],
        (input.shape[2] + 2 * padding - windowSize) / stride + 1,
        (input.shape[3] + 2 * padding - windowSize) / stride + 1,
    ]
    let columns = input
        .view(as: [input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3]])
        .img2col(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride)
    return reduce(columns).view(as: outputShape)
}

private func referenceNormalization(_ input: DoubleTensor, along axes: [Int], scale: DoubleTensor, shift: DoubleTensor) -> DoubleTensor {
    var statisticsShape = input.shape
    for axis in axes {
        statisticsShape[axis] = 1
    }
    let mean = referenceMean(input, along: axes).view(as: statisticsShape)
    let variance = referenceVariance(input, along: axes).view(as: statisticsShape)
    return (input - mean) / (variance.sqrt() + 1e-5) * scale + shift
}

private func referenceAttention(queries: DoubleTensor, keys: DoubleTensor, values: DoubleTensor, mask: DoubleTensor?, temperature: Double) -> DoubleTensor {
    var scores = (queries / DoubleTensor(temperature)).broadcastMatrixMultiplied(with: keys, transposeSelf: false, transposeOther: true)
    if let mask {
        scores -= mask * 1e9
    }
    return referenceSoftmax(scores, axis: 3).broadcastMatrixMultiplied(with: values)
}

private func referenceMultiHeadAttention(_ sources: [DoubleTensor], mask: DoubleTensor?, heads: Int) -> DoubleTensor {
    let (q, k, v) = (sources[0], sources[1], sources[2])
    let (queryWeights, keyWeights, valueWeights, outputWeights) = (sources[3], sources[4], sources[5], sources[6])
    let keyDim = queryWeights.shape[1] / heads
    let valueDim = valueWeights.shape[1] / heads

    let queryHeads = q.broadcastMatrixMultiplied(with: queryWeights).view(as: q.shape[0], q.shape[1], heads, keyDim).permuted(to: 0, 2, 1, 3)
    let keyHeads = k.broadcastMatrixMultiplied(with: keyWeights).view(as: k.shape[0], k.shape[1], heads, keyDim).permuted(to: 0, 2, 1, 3)
    let valueHeads = v.broadcastMatrixMultiplied(with: valueWeights).view(as: v.shape[0], v.shape[1], heads, valueDim).permuted(to: 0, 2, 1, 3)
    let attended = referenceAttention(queries: queryHeads, keys: keyHeads, values: valueHeads, mask: mask, temperature: Double(keyDim).squareRoot())
    return attended
        .permuted(to: 0, 2, 1, 3)
        .view(as: q.shape[0], q.shape[1], -1)
        .broadcastMatrixMultiplied(with: outputWeights)
}

// MARK: Cases

extension FusedOperationCase {
    static let convolutions: [FusedOperationCase] = [
        FusedOperationCase(
            "convolved2d, padding 1, stride 2",
            sources: [uniform([2, 2, 5, 5], seed: 1), uniform([3, 2, 3, 3], seed: 2), uniform([1, 3, 1, 1], seed: 3)],
            fused: { $0[0].convolved2d(filters: $0[1], bias: $0[2], padding: 1, stride: 2) },
            reference: { referenceConvolution($0[0], filters: $0[1], padding: 1, stride: 2) + $0[2] },
        ),
        FusedOperationCase(
            "convolved2d without bias, rectangular kernel",
            sources: [uniform([1, 2, 5, 4], seed: 4), uniform([2, 2, 3, 2], seed: 5)],
            fused: { $0[0].convolved2d(filters: $0[1], padding: 0) },
            reference: { referenceConvolution($0[0], filters: $0[1], padding: 0, stride: 1) },
        ),
        FusedOperationCase(
            "transposedConvolved2d, inset 1, stride 2",
            sources: [uniform([2, 3, 3, 3], seed: 6), uniform([2, 3, 3, 3], seed: 7), uniform([1, 2, 1, 1], seed: 8)],
            fused: { $0[0].transposedConvolved2d(filters: $0[1], bias: $0[2], inset: 1, stride: 2) },
            reference: { referenceTransposedConvolution($0[0], filters: $0[1], inset: 1, stride: 2) + $0[2] },
        ),
        FusedOperationCase(
            "transposedConvolved2d without bias",
            sources: [uniform([1, 2, 3, 3], seed: 9), uniform([3, 2, 2, 2], seed: 10)],
            fused: { $0[0].transposedConvolved2d(filters: $0[1], inset: 0) },
            reference: { referenceTransposedConvolution($0[0], filters: $0[1], inset: 0, stride: 1) },
        ),
        // Distinct values, so that the maximum of every window is unique.
        FusedOperationCase(
            "maxPooled2d",
            sources: [DoubleTensor((0 ..< 32).map { Double(($0 * 7) % 32) / 32 + 0.01 }, shape: [2, 1, 4, 4])],
            fused: { $0[0].maxPooled2d(windowSize: 2, padding: 0, stride: 2) },
            reference: { referencePooling($0[0], windowSize: 2, padding: 0, stride: 2) { $0.reduceMax(along: [0]) } },
        ),
        FusedOperationCase(
            "maxPooled2d with padding and overlapping windows",
            sources: [DoubleTensor((0 ..< 50).map { Double(($0 * 11) % 50) / 50 + 0.01 }, shape: [1, 2, 5, 5])],
            fused: { $0[0].maxPooled2d(windowSize: 3, padding: 1, stride: 2) },
            reference: { referencePooling($0[0], windowSize: 3, padding: 1, stride: 2) { $0.reduceMax(along: [0]) } },
        ),
        FusedOperationCase(
            "averagePooled2d",
            sources: [uniform([2, 2, 5, 5], seed: 11)],
            fused: { $0[0].averagePooled2d(windowSize: 3, padding: 1, stride: 2) },
            reference: { referencePooling($0[0], windowSize: 3, padding: 1, stride: 2) { referenceMean($0, along: [0]) } },
        ),
    ]

    static let activations: [FusedOperationCase] = [
        FusedOperationCase("tanh", sources: [uniform([3, 4], seed: 20)], fused: { $0[0].tanh() }, reference: { $0[0].tanh() }),
        FusedOperationCase(
            "rectifiedLinear",
            sources: [awayFromZero([3, 4], seed: 21)],
            fused: { $0[0].rectifiedLinear() },
            reference: { $0[0].rectifiedLinear() },
        ),
        FusedOperationCase("sigmoid", sources: [uniform([3, 4], seed: 22)], fused: { $0[0].sigmoid() }, reference: { referenceSigmoid($0[0]) }),
        FusedOperationCase("softmax", sources: [uniform([3, 4], seed: 23)], fused: { $0[0].softmax(axis: 1) }, reference: { referenceSoftmax($0[0], axis: 1) }),
        FusedOperationCase(
            "softmax along axis 0 of 3 axes",
            sources: [uniform([3, 2, 2], seed: 24)],
            fused: { $0[0].softmax(axis: 0) },
            reference: { referenceSoftmax($0[0], axis: 0) },
        ),
        FusedOperationCase("logSoftmax", sources: [uniform([3, 4], seed: 25)], fused: { $0[0].logSoftmax(axis: 1) }, reference: { referenceLogSoftmax($0[0], axis: 1) }),
        FusedOperationCase(
            "leakyRectifiedLinear",
            sources: [awayFromZero([3, 4], seed: 26), uniform([4], min: 0.01, max: 0.3, seed: 27)],
            fused: { $0[0].leakyRectifiedLinear(leakage: $0[1]) },
            reference: { $0[0].rectifiedLinear() - $0[1] * (-$0[0]).rectifiedLinear() },
        ),
        FusedOperationCase(
            "gaussianErrorLinear",
            sources: [uniform([3, 4], min: -3, max: 3, seed: 28)],
            fused: { $0[0].gaussianErrorLinear() },
            reference: { $0[0] * referenceSigmoid($0[0] * 1.702) },
        ),
        FusedOperationCase(
            "swishActivated",
            sources: [uniform([3, 4], min: -3, max: 3, seed: 29), uniform([4], min: 0.5, max: 2, seed: 30)],
            fused: { $0[0].swishActivated(beta: $0[1]) },
            reference: { $0[0] * referenceSigmoid($0[1] * $0[0]) },
        ),
        FusedOperationCase(
            "mishActivated",
            sources: [uniform([3, 4], min: -3, max: 3, seed: 31)],
            fused: { $0[0].mishActivated() },
            reference: { $0[0] * (1 + $0[0].exp()).log().tanh() },
        ),
        FusedOperationCase(
            "lishtActivated",
            sources: [uniform([3, 4], min: -3, max: 3, seed: 32)],
            fused: { $0[0].lishtActivated() },
            reference: { $0[0] * $0[0].tanh() },
        ),
        FusedOperationCase(
            "exponentialLinearActivated",
            sources: [awayFromZero([3, 4], seed: 33), DoubleTensor(0.7)],
            fused: { $0[0].exponentialLinearActivated(alpha: $0[1]) },
            reference: {
                let positive = $0[0].heaviside()
                return positive * $0[0] + (1 - positive) * $0[1] * ($0[0].exp() - 1)
            },
        ),
        FusedOperationCase("softplus", sources: [uniform([3, 4], min: -3, max: 3, seed: 34)], fused: { $0[0].softplus() }, reference: { ($0[0].exp() + 1).log() }),
        FusedOperationCase(
            "squareplus",
            sources: [uniform([3, 4], min: -3, max: 3, seed: 35)],
            fused: { $0[0].squareplus() },
            reference: { ($0[0] + ($0[0] * $0[0] + 4).sqrt()) / 2 },
        ),
    ]

    static let reductions: [FusedOperationCase] = [
        FusedOperationCase("reduceMean along 1", sources: [uniform([3, 4], seed: 40)], fused: { $0[0].reduceMean(along: [1]) }, reference: { referenceMean($0[0], along: [1]) }),
        FusedOperationCase("reduceMean along 0 and 2", sources: [uniform([2, 3, 4], seed: 41)], fused: { $0[0].reduceMean(along: [0, 2]) }, reference: { referenceMean($0[0], along: [0, 2]) }),
        FusedOperationCase("reduceMean of all elements", sources: [uniform([2, 3], seed: 42)], fused: { $0[0].reduceMean() }, reference: { referenceMean($0[0], along: [0, 1]) }),
        FusedOperationCase("variance along 1", sources: [uniform([3, 4], seed: 43)], fused: { $0[0].variance(along: [1]) }, reference: { referenceVariance($0[0], along: [1]) }),
        FusedOperationCase("variance along 0 and 2", sources: [uniform([2, 3, 4], seed: 44)], fused: { $0[0].variance(along: [0, 2]) }, reference: { referenceVariance($0[0], along: [0, 2]) }),
    ]

    static let normalizations: [FusedOperationCase] = [
        FusedOperationCase(
            "layerNormalized along the last axis",
            sources: [uniform([2, 3, 4], seed: 50), uniform([4], min: 0.5, max: 1.5, seed: 51), uniform([4], seed: 52)],
            fused: { $0[0].layerNormalized(scale: $0[1], shift: $0[2]) },
            reference: { referenceNormalization($0[0], along: [2], scale: $0[1], shift: $0[2]) },
        ),
        FusedOperationCase(
            "layerNormalized along 2 axes",
            sources: [uniform([2, 3, 4], seed: 53), uniform([3, 4], min: 0.5, max: 1.5, seed: 54), uniform([3, 4], seed: 55)],
            fused: { $0[0].layerNormalized(scale: $0[1], shift: $0[2]) },
            reference: { referenceNormalization($0[0], along: [1, 2], scale: $0[1], shift: $0[2]) },
        ),
        FusedOperationCase(
            "batchNormalized",
            sources: [uniform([4, 3], seed: 56), uniform([3], min: 0.5, max: 1.5, seed: 57), uniform([3], seed: 58)],
            fused: { $0[0].batchNormalized(scale: $0[1], shift: $0[2]).output },
            reference: { referenceNormalization($0[0], along: [0], scale: $0[1], shift: $0[2]) },
        ),
        FusedOperationCase(
            "batchNormalized with broadcast scale",
            sources: [uniform([3, 2, 2, 2], seed: 59), uniform([2, 1, 1], min: 0.5, max: 1.5, seed: 60), uniform([2, 1, 1], seed: 61)],
            fused: { $0[0].batchNormalized(scale: $0[1], shift: $0[2]).output },
            reference: { referenceNormalization($0[0], along: [0], scale: $0[1], shift: $0[2]) },
        ),
        FusedOperationCase(
            "batchNormalized with fixed statistics",
            sources: [uniform([4, 3], seed: 62), uniform([3], min: 0.5, max: 1.5, seed: 63), uniform([3], seed: 64)],
            fused: { $0[0].batchNormalized(scale: $0[1], shift: $0[2], mean: uniform([3], seed: 65), variance: uniform([3], min: 0.5, max: 2, seed: 66)) },
            reference: { ($0[0] - uniform([3], seed: 65)) / (uniform([3], min: 0.5, max: 2, seed: 66).sqrt() + 1e-5) * $0[1] + $0[2] },
        ),
    ]

    static let layers: [FusedOperationCase] = [
        FusedOperationCase(
            "linearlyTransformed",
            sources: [uniform([3, 4], seed: 70), uniform([4, 2], seed: 71), uniform([2], seed: 72)],
            fused: { $0[0].linearlyTransformed(weights: $0[1], bias: $0[2]) },
            reference: { $0[0].matrixMultiplied(with: $0[1]) + $0[2] },
        ),
        FusedOperationCase(
            "linearlyTransformed vector without bias",
            sources: [uniform([4], seed: 73), uniform([4, 3], seed: 74)],
            fused: { $0[0].linearlyTransformed(weights: $0[1]) },
            reference: { $0[0].matrixMultiplied(with: $0[1]) },
        ),
    ]

    static let losses: [FusedOperationCase] = [
        FusedOperationCase(
            "binaryCrossEntropy",
            sources: [uniform([2, 3], seed: 80), uniform([2, 3], min: 0.1, max: 0.9, seed: 81)],
            fused: { binaryCrossEntropy(expected: $0[0], actual: $0[1]) },
            reference: {
                let (e, a) = ($0[0].view(as: [-1]), $0[1].view(as: [-1]))
                return referenceMean(-(e * a.log() + (1 - e) * (1 - a).log()), along: [0])
            },
        ),
        FusedOperationCase(
            "categoricalCrossEntropy",
            sources: [uniform([4, 3], min: 0.1, max: 0.9, seed: 82)],
            fused: { categoricalCrossEntropy(expected: Tensor<Int32, CPU>([2, 0, 1, 1]), actual: $0[0]) },
            reference: { -referenceMean($0[0].gather(using: Tensor<Int32, CPU>([2, 0, 1, 1]), alongAxis: 1).log(), along: [0]) },
        ),
        FusedOperationCase(
            "categoricalNegativeLogLikelihood with an ignored label",
            sources: [uniform([2, 2, 3], min: -3, max: -0.1, seed: 83)],
            fused: { categoricalNegativeLogLikelihood(expected: Tensor<Int32, CPU>([2, -1, 1, 0], shape: [2, 2]), actual: $0[0]) },
            reference: { -referenceMean($0[0].view(as: [4, 3]).gather(using: Tensor<Int32, CPU>([2, -1, 1, 0]), alongAxis: 1), along: [0]) },
        ),
        FusedOperationCase(
            "meanSquaredError",
            sources: [uniform([3, 2], seed: 84), uniform([3, 2], seed: 85)],
            fused: { meanSquaredError(expected: $0[0], actual: $0[1]) },
            reference: { (($0[0] - $0[1]) * ($0[0] - $0[1])).reduceSum() / 3 },
        ),
        FusedOperationCase(
            "meanSquaredError with a broadcast expected value",
            sources: [DoubleTensor(0.25), uniform([3, 2], seed: 86)],
            fused: { meanSquaredError(expected: $0[0], actual: $0[1]) },
            reference: { (($0[0] - $0[1]) * ($0[0] - $0[1])).reduceSum() },
        ),
        FusedOperationCase(
            "l1loss",
            sources: [awayFromZero([3, 2], seed: 87)],
            fused: { l1loss($0[0], loss: 0.3) },
            reference: { referenceMean($0[0].rectifiedLinear() + (-$0[0]).rectifiedLinear(), along: [0, 1]) * 0.3 },
        ),
        FusedOperationCase("l2loss", sources: [uniform([3, 2], seed: 88)], fused: { l2loss($0[0], loss: 0.3) }, reference: { referenceMean($0[0] * $0[0], along: [0, 1]) * 0.3 }),
    ]

    static let attention: [FusedOperationCase] = [
        FusedOperationCase(
            "scaledDotProductAttention with mask",
            sources: [uniform([1, 2, 3, 4], seed: 90), uniform([1, 2, 5, 4], seed: 91), uniform([1, 2, 5, 3], seed: 92)],
            fused: { scaledDotProductAttention(queries: $0[0], keys: $0[1], values: $0[2], mask: DoubleTensor([0, 0, 1, 0, 1], shape: [1, 1, 1, 5]), temperature: 2) },
            reference: { referenceAttention(queries: $0[0], keys: $0[1], values: $0[2], mask: DoubleTensor([0, 0, 1, 0, 1], shape: [1, 1, 1, 5]), temperature: 2) },
        ),
        FusedOperationCase(
            "scaledDotProductAttention without mask",
            sources: [uniform([2, 1, 3, 2], seed: 93), uniform([2, 1, 3, 2], seed: 94), uniform([2, 1, 3, 2], seed: 95)],
            fused: { scaledDotProductAttention(queries: $0[0], keys: $0[1], values: $0[2], mask: nil, temperature: 1.5) },
            reference: { referenceAttention(queries: $0[0], keys: $0[1], values: $0[2], mask: nil, temperature: 1.5) },
        ),
        FusedOperationCase(
            "multiHeadAttention",
            sources: [
                uniform([2, 3, 4], seed: 96), uniform([2, 5, 4], seed: 97), uniform([2, 5, 4], seed: 98),
                uniform([4, 4], seed: 99), uniform([4, 4], seed: 100), uniform([4, 6], seed: 101), uniform([6, 4], seed: 102),
            ],
            fused: {
                multiHeadAttention(
                    queries: $0[0], keys: $0[1], values: $0[2], mask: DoubleTensor([0, 0, 0, 1, 1], shape: [1, 1, 1, 5]),
                    queryWeights: $0[3], keyWeights: $0[4], valueWeights: $0[5], outputWeights: $0[6], heads: 2, temperature: 2.0.squareRoot(),
                )
            },
            reference: { referenceMultiHeadAttention($0, mask: DoubleTensor([0, 0, 0, 1, 1], shape: [1, 1, 1, 5]), heads: 2) },
        ),
    ]

    static let recurrent: [FusedOperationCase] = [
        FusedOperationCase(
            "gatedRecurrentUnitStep",
            sources: [
                uniform([3, 4], seed: 110), uniform([3, 4], seed: 111), uniform([3, 4], seed: 112), uniform([3, 4], seed: 113),
                uniform([4, 4], seed: 114), uniform([4, 4], seed: 115), uniform([4, 4], seed: 116),
            ],
            fused: {
                gatedRecurrentUnitStep(updateInput: $0[0], resetInput: $0[1], candidateInput: $0[2], state: $0[3], updateWeights: $0[4], resetWeights: $0[5], candidateWeights: $0[6])
            },
            reference: {
                let update = referenceSigmoid($0[0] + $0[3].matrixMultiplied(with: $0[4]))
                let reset = referenceSigmoid($0[1] + $0[3].matrixMultiplied(with: $0[5]))
                let candidate = ($0[2] + (reset * $0[3]).matrixMultiplied(with: $0[6])).tanh()
                return (1 - update) * $0[3] + update * candidate
            },
        ),
    ]

    static let all = convolutions + activations + reductions + normalizations + layers + losses + attention + recurrent
}

// MARK: Checks

/// Returns the gradients of `sum(function(sources) * weights)` with respect to the differentiable sources, estimated with central differences.
private func numericalGradients(of function: ([DoubleTensor]) -> DoubleTensor, at sources: [DoubleTensor], differentiable: [Int], step: Double = 1e-5) -> [DoubleTensor] {
    let plainSources = sources.map { $0.detached() }
    return differentiable.map { position in
        let elements = plainSources[position].elements
        let gradient = elements.indices.map { index in
            func value(at offset: Double) -> Double {
                var shifted = elements
                shifted[index] += offset
                var arguments = plainSources
                arguments[position] = DoubleTensor(shifted, shape: plainSources[position].shape)
                return function(arguments).reduceSum().item
            }
            return (value(at: step) - value(at: -step)) / (2 * step)
        }
        return DoubleTensor(gradient, shape: plainSources[position].shape)
    }
}

/// Records an issue when the largest difference is larger than `tolerance` times the largest magnitude of `expected`, plus `tolerance`.
private func expectApproximatelyEqual(_ actual: DoubleTensor, _ expected: DoubleTensor, tolerance: Double = 1e-6, _ comment: String, sourceLocation: SourceLocation = #_sourceLocation) {
    guard actual.shape == expected.shape else {
        Issue.record("\(comment): shape \(actual.shape) differs from \(expected.shape)", sourceLocation: sourceLocation)
        return
    }
    let difference = zip(actual.elements, expected.elements).map { abs($0 - $1) }.max() ?? 0
    let magnitude = expected.elements.map(abs).max() ?? 0
    #expect(difference <= tolerance * (1 + magnitude), "\(comment): difference \(difference): \(actual) vs \(expected)", sourceLocation: sourceLocation)
}

struct FusedOperationTests {
    /// Weights of the elements of a result, so that every element contributes differently to the checked sum.
    private func outputWeights(for result: DoubleTensor) -> DoubleTensor {
        uniform(result.shape, min: 0.5, max: 1.5, seed: 1000)
    }

    @Test(arguments: FusedOperationCase.all)
    func defaultMatchesComposedImplementation(_ operation: FusedOperationCase) {
        let sources = operation.sources.map { $0.detached() }
        let result = operation.fused(sources)
        #expect(!result.requiresGradient)
        expectApproximatelyEqual(result, operation.reference(sources), tolerance: 1e-9, "result")
    }

    @Test(arguments: FusedOperationCase.all)
    func firstDerivativeMatchesNumericalGradient(_ operation: FusedOperationCase) {
        let sources = operation.makeSources()
        let weights = outputWeights(for: operation.fused(sources))
        let differentiableSources = operation.differentiable.map { sources[$0] }

        let gradients = (operation.fused(sources) * weights).reduceSum().gradients(of: differentiableSources)
        let referenceSources = operation.makeSources()
        let referenceGradients = (operation.reference(referenceSources) * weights).reduceSum().gradients(of: operation.differentiable.map { referenceSources[$0] })
        let numerical = numericalGradients(of: { operation.fused($0) * weights }, at: sources, differentiable: operation.differentiable)

        for (index, position) in operation.differentiable.enumerated() {
            #expect(!gradients[index].requiresGradient)
            expectApproximatelyEqual(gradients[index], referenceGradients[index], tolerance: 1e-9, "gradient of source \(position) vs. composed implementation")
            expectApproximatelyEqual(gradients[index], numerical[index], "gradient of source \(position) vs. central differences")
        }
    }

    @Test(arguments: FusedOperationCase.all)
    func retainedFirstDerivativeMatchesNumericalGradient(_ operation: FusedOperationCase) {
        let sources = operation.makeSources()
        let weights = outputWeights(for: operation.fused(sources))

        let gradients = (operation.fused(sources) * weights).reduceSum().gradients(of: operation.differentiable.map { sources[$0] }, retainBackwardsGraph: true)
        let numerical = numericalGradients(of: { operation.fused($0) * weights }, at: sources, differentiable: operation.differentiable)

        for (index, position) in operation.differentiable.enumerated() {
            #expect(gradients[index].requiresGradient)
            expectApproximatelyEqual(gradients[index], numerical[index], "gradient of source \(position)")
        }
    }

    /// Checks all second derivatives at once: the gradient of `sum(gradient(i) * directions(i))` over the differentiable sources `i`,
    /// a Hessian-vector product, must match its central difference estimate.
    @Test(arguments: FusedOperationCase.all)
    func secondDerivativeMatchesNumericalGradient(_ operation: FusedOperationCase) {
        let sources = operation.makeSources()
        let weights = outputWeights(for: operation.fused(sources))
        let directions = operation.differentiable.map { uniform(sources[$0].shape, seed: 2000 + UInt64($0)) }

        let directionalDerivative: ([DoubleTensor]) -> DoubleTensor = { arguments in
            var arguments = arguments
            for position in operation.differentiable {
                arguments[position].requiresGradient = true
            }
            let gradients = (operation.fused(arguments) * weights).reduceSum().gradients(of: operation.differentiable.map { arguments[$0] })
            return zip(gradients, directions).map { ($0 * $1).reduceSum() }.reduce(DoubleTensor(0), +)
        }

        let differentiableSources = operation.differentiable.map { sources[$0] }
        let gradients = (operation.fused(sources) * weights).reduceSum().gradients(of: differentiableSources, retainBackwardsGraph: true)
        let hessianProduct = zip(gradients, directions)
            .map { ($0 * $1).reduceSum() }
            .reduce(DoubleTensor(0), +)
            .gradients(of: differentiableSources)
        let numerical = numericalGradients(of: directionalDerivative, at: sources, differentiable: operation.differentiable)

        for (index, position) in operation.differentiable.enumerated() {
            expectApproximatelyEqual(hessianProduct[index], numerical[index], tolerance: 1e-5, "second derivative of source \(position)")
        }
    }

    @Test func backwardComputesOnlyRequiredGradients() {
        var input = uniform([1, 2, 4, 4], seed: 3000)
        let filters = uniform([3, 2, 3, 3], seed: 3001)
        let bias = uniform([3], seed: 3002)
        input.requiresGradient = true
        let outputGradient = DoubleTensor(repeating: 1, shape: [1, 3, 4, 4])

        var gradients: (input: DoubleTensor?, filters: DoubleTensor?, bias: DoubleTensor?) = (nil, nil, nil)
        CPU.FusedOperations.convolution2dBackward(input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: 1, stride: 1, accumulating: &gradients)
        #expect(gradients.input?.shape == input.shape)
        #expect(gradients.input?.requiresGradient == false)
        #expect(gradients.filters == nil)
        #expect(gradients.bias == nil)
    }

    @Test func backwardAddsToAccumulatedGradients() throws {
        var (input, weights) = (uniform([3, 4], seed: 3006), uniform([4, 5], seed: 3007))
        input.requiresGradient = true
        weights.requiresGradient = true
        let outputGradient = uniform([3, 5], seed: 3008)
        let (inputStart, weightStart) = (uniform([3, 4], seed: 3009), uniform([4, 5], seed: 3010))

        var empty: (input: DoubleTensor?, weights: DoubleTensor?, bias: DoubleTensor?) = (nil, nil, nil)
        CPU.FusedOperations.linearBackward(input: input, weights: weights, bias: nil, outputGradient: outputGradient, accumulating: &empty)
        var accumulated: (input: DoubleTensor?, weights: DoubleTensor?, bias: DoubleTensor?) = (inputStart, weightStart, nil)
        CPU.FusedOperations.linearBackward(input: input, weights: weights, bias: nil, outputGradient: outputGradient, accumulating: &accumulated)

        try expectApproximatelyEqual(#require(accumulated.input), inputStart + empty.input!, tolerance: 1e-12, "input gradient")
        try expectApproximatelyEqual(#require(accumulated.weights), weightStart + empty.weights!, tolerance: 1e-12, "weight gradient")
        #expect(accumulated.bias == nil)
        // The accumulators share their storage with the start values, so the kernels must copy them before they write.
        #expect(inputStart == uniform([3, 4], seed: 3009))
        #expect(weightStart == uniform([4, 5], seed: 3010))
    }

    @Test func weightsUsedInSeveralStepsGetSumOfGradients() {
        let sources = [uniform([2, 6], seed: 3060), uniform([6, 6], seed: 3061), uniform([6, 6], seed: 3062), uniform([6, 6], seed: 3063)]
        // Three steps of a recurrent unit with batch size 2 and shared weights, as in a sequence model.
        func unroll(_ sources: [DoubleTensor]) -> DoubleTensor {
            var state = sources[0]
            for step in 0 ..< 3 {
                let input = DoubleTensor(repeating: Double(step) / 4, shape: [2, 6])
                state = gatedRecurrentUnitStep(
                    updateInput: input, resetInput: -input, candidateInput: input * 2, state: state,
                    updateWeights: sources[1], resetWeights: sources[2], candidateWeights: sources[3],
                )
            }
            return state
        }
        var trainable = sources
        for index in trainable.indices {
            trainable[index].requiresGradient = true
        }
        let weights = uniform([2, 6], min: 0.5, max: 1.5, seed: 3064)
        let gradients = (unroll(trainable) * weights).reduceSum().gradients(of: trainable)
        let numerical = numericalGradients(of: { unroll($0) * weights }, at: sources, differentiable: Array(sources.indices))
        for index in sources.indices {
            expectApproximatelyEqual(gradients[index], numerical[index], "gradient of source \(index)")
        }
    }

    @Test func frozenSourcesGetNoGradient() {
        var input = uniform([1, 2, 4, 4], seed: 3003)
        input.requiresGradient = true
        let filters = uniform([3, 2, 3, 3], seed: 3004)
        let result = input.convolved2d(filters: filters, bias: uniform([1, 3, 1, 1], seed: 3005), padding: 1)

        let gradients = result.reduceSum().gradients(of: [input, filters])
        expectApproximatelyEqual(gradients[0], referenceConvolution(input, filters: filters, padding: 1, stride: 1).reduceSum().gradients(of: [input])[0], "input gradient")
        #expect(gradients[1] == DoubleTensor(repeating: 0, shape: filters.shape))
    }

    @Test func sameSourceInSeveralPositionsGetsSumOfGradients() {
        let sources = [
            uniform([2, 3, 4], seed: 3010),
            uniform([4, 4], seed: 3011), uniform([4, 4], seed: 3012), uniform([4, 4], seed: 3013), uniform([4, 4], seed: 3014),
        ]
        func selfAttention(_ sources: [DoubleTensor]) -> DoubleTensor {
            multiHeadAttention(
                queries: sources[0], keys: sources[0], values: sources[0], mask: nil,
                queryWeights: sources[1], keyWeights: sources[2], valueWeights: sources[3], outputWeights: sources[4], heads: 2, temperature: 2.0.squareRoot(),
            )
        }
        var input = sources[0]
        input.requiresGradient = true
        let gradient = selfAttention([input] + sources.dropFirst()).reduceSum().gradients(of: [input])[0]
        let numerical = numericalGradients(of: selfAttention, at: sources, differentiable: [0])[0]
        expectApproximatelyEqual(gradient, numerical, "input gradient")
    }

    @Test func batchNormalizationReturnsBatchStatistics() {
        let input = uniform([5, 3], seed: 3020)
        let (_, mean, variance) = input.batchNormalized(scale: DoubleTensor(repeating: 1, shape: [3]), shift: DoubleTensor(repeating: 0, shape: [3]))
        expectApproximatelyEqual(mean, referenceMean(input, along: [0]), "mean")
        expectApproximatelyEqual(variance, referenceVariance(input, along: [0]), "variance")
    }

    @Test func dropoutGradientUsesTheMaskOfTheForwardPass() {
        var input = uniform([200], min: 1, max: 2, seed: 3030)
        input.requiresGradient = true
        let result = input.droppedOut(rate: 0.5)
        // Every element is either kept or zero.
        let ratios = (result.detached() / input.detached()).elements
        #expect(ratios.allSatisfy { abs($0) < 1e-12 || abs($0 - 1) < 1e-12 })
        let mask = DoubleTensor(ratios.map { $0 > 0.5 ? 1 : 0 })
        #expect(mask.elements.contains(0) && mask.elements.contains(1))

        let weights = uniform([200], seed: 3031)
        let gradient = (result * weights).reduceSum().gradients(of: [input])[0]
        expectApproximatelyEqual(gradient, weights * mask, "gradient")

        let retainedGradient = (result * weights).reduceSum().gradients(of: [input], retainBackwardsGraph: true)[0]
        expectApproximatelyEqual(retainedGradient, weights * mask, "retained gradient")
        #expect(retainedGradient.reduceSum().gradients(of: [input])[0] == DoubleTensor(repeating: 0, shape: [200]))
    }

    @Test func categoricalCrossEntropyIgnoresLabels() {
        let actual = uniform([3, 4], min: 0.1, max: 0.9, seed: 3040)
        let loss = categoricalCrossEntropy(expected: Tensor<Int32, CPU>([1, -1, 3]), actual: actual)
        let selected = actual.gather(using: Tensor<Int32, CPU>([1, 0, 3]), alongAxis: 1).elements
        expectApproximatelyEqual(loss, DoubleTensor(-(Foundation.log(selected[0]) + Foundation.log(selected[2])) / 3), "loss")
    }

    @Test func exponentialLinearUnitIsExponentialForNegativeInputs() {
        let input = DoubleTensor([-2, -0.5, 0.5, 100])
        let expected = DoubleTensor([0.5 * (Foundation.exp(-2) - 1), 0.5 * (Foundation.exp(-0.5) - 1), 0.5, 100])
        expectApproximatelyEqual(input.exponentialLinearActivated(alpha: 0.5), expected, "result")
    }

    @Test func positionalEncodingMatchesDefinition() {
        let encoding = Tensor<Double, CPU>(positionalEncodingWithLength: 3, hiddenSize: 4)
        let expected = (0 ..< 3).flatMap { position in
            (0 ..< 2).flatMap { index in
                let angle = Double(position) / Foundation.pow(10000, Double(index) / 2)
                return [Foundation.sin(angle), Foundation.cos(angle)]
            }
        }
        expectApproximatelyEqual(encoding, DoubleTensor(expected, shape: [3, 4]), "encoding")
    }
}

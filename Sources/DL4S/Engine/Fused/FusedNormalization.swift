//
//  FusedNormalization.swift
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
    static func reduceMean<N: NumericType>(input: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        let input = input.detached()
        return input.reduceSum(along: axes) / Tensor(N(Composed.elementCount(of: input.shape, along: axes)))
    }

    static func reduceMeanBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        Composed.reduceMeanGradient(inputShape: input.shape, outputGradient: outputGradient.detached(), axes: axes)
    }

    static func variance<N: NumericType>(input: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        let input = input.detached()
        let mean = input.reduceMean(along: axes)
        return (input * input).reduceMean(along: axes) - mean * mean
    }

    static func varianceBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        Composed.varianceGradient(input: input.detached(), outputGradient: outputGradient.detached(), axes: axes)
    }

    static func layerNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, epsilon: N) -> Tensor<N, Device> {
        let axes = Composed.layerNormalizationAxes(input: input, scale: scale)
        return Composed.normalized(input.detached(), along: axes, epsilon: epsilon) * scale.detached() + shift.detached()
    }

    static func layerNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N) -> (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?) {
        Composed.normalizationGradients(
            input: input.detached(),
            scale: scale.detached(),
            shiftShape: shift.shape,
            outputGradient: outputGradient.detached(),
            axes: Composed.layerNormalizationAxes(input: input, scale: scale),
            epsilon: epsilon,
            computesInput: input.requiresGradient,
            computesScale: scale.requiresGradient,
            computesShift: shift.requiresGradient,
        )
    }

    static func batchNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, epsilon: N) -> (output: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>) {
        let input = input.detached()
        let mean = input.reduceMean(along: [0])
        let variance = (input * input).reduceMean(along: [0]) - mean * mean
        let normalized = (input - mean) / (variance.sqrt() + Tensor(epsilon))
        return (normalized * scale.detached() + shift.detached(), mean, variance)
    }

    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N) -> (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?) {
        Composed.normalizationGradients(
            input: input.detached(),
            scale: scale.detached(),
            shiftShape: shift.shape,
            outputGradient: outputGradient.detached(),
            axes: [0],
            epsilon: epsilon,
            computesInput: input.requiresGradient,
            computesScale: scale.requiresGradient,
            computesShift: shift.requiresGradient,
        )
    }

    static func batchNormalization<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>, epsilon: N) -> Tensor<N, Device> {
        let normalized = (input.detached() - mean.detached()) / (variance.detached().sqrt() + Tensor(epsilon))
        return normalized * scale.detached() + shift.detached()
    }

    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, Device>, scale: Tensor<N, Device>, shift: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>, outputGradient: Tensor<N, Device>, epsilon: N) -> (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?) {
        Composed.fixedNormalizationGradients(
            input: input.detached(),
            scale: scale.detached(),
            shiftShape: shift.shape,
            mean: mean.detached(),
            variance: variance.detached(),
            outputGradient: outputGradient.detached(),
            epsilon: epsilon,
            computesInput: input.requiresGradient,
            computesScale: scale.requiresGradient,
            computesShift: shift.requiresGradient,
        )
    }
}

// MARK: Composed gradients

extension Composed {
    /// Number of elements that a reduction along the given axes combines into one.
    static func elementCount(of shape: [Int], along axes: [Int]) -> Int {
        axes.map { shape[$0] }.reduce(1, *)
    }

    /// Shape of the input with every reduced axis replaced by 1.
    static func keptShape(of shape: [Int], along axes: [Int]) -> [Int] {
        var keptShape = shape
        for axis in axes {
            keptShape[axis] = 1
        }
        return keptShape
    }

    static func reduceMeanGradient<N, Device>(inputShape: [Int], outputGradient: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        let weights = Tensor<N, Device>(repeating: N.one / N(elementCount(of: inputShape, along: axes)), shape: inputShape)
        return weights * outputGradient.view(as: keptShape(of: inputShape, along: axes))
    }

    static func varianceGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int]) -> Tensor<N, Device> {
        let keptShape = keptShape(of: input.shape, along: axes)
        let centered = input - input.reduceMean(along: axes).view(as: keptShape)
        let factor = Tensor<N, Device>(N(2) / N(elementCount(of: input.shape, along: axes)))
        return factor * centered * outputGradient.view(as: keptShape)
    }

    /// Trailing axes of the input that layer normalization reduces along.
    static func layerNormalizationAxes<N, Device>(input: Tensor<N, Device>, scale: Tensor<N, Device>) -> [Int] {
        Array(input.dim - scale.dim ..< input.dim)
    }

    /// Normalizes the input along the axes to `(input - mean) / (sqrt(variance) + epsilon)`.
    static func normalized<N, Device>(_ input: Tensor<N, Device>, along axes: [Int], epsilon: N) -> Tensor<N, Device> {
        let keptShape = keptShape(of: input.shape, along: axes)
        let mean = input.reduceMean(along: axes).view(as: keptShape)
        let variance = (input * input).reduceMean(along: axes).view(as: keptShape) - mean * mean
        return (input - mean) / (variance.sqrt() + Tensor(epsilon))
    }

    /// Computes the gradients of a normalization with the statistics of the input along the given axes,
    /// followed by a scale and a shift.
    static func normalizationGradients<N, Device>(
        input: Tensor<N, Device>,
        scale: Tensor<N, Device>,
        shiftShape: [Int],
        outputGradient: Tensor<N, Device>,
        axes: [Int],
        epsilon: N,
        computesInput: Bool,
        computesScale: Bool,
        computesShift: Bool,
    ) -> (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?) {
        let keptShape = keptShape(of: input.shape, along: axes)
        let centered = input - input.reduceMean(along: axes).view(as: keptShape)
        // The variance of the centered values equals the variance of the forward pass up to rounding, and needs fewer operations.
        let standardDeviation = (centered * centered).reduceMean(along: axes).view(as: keptShape).sqrt()
        let divisor = standardDeviation + Tensor(epsilon)
        let normalized = centered / divisor

        let scaleGradient = computesScale ? (outputGradient * normalized).reducingBroadcast(to: scale.shape) : nil
        let shiftGradient = computesShift ? outputGradient.reducingBroadcast(to: shiftShape) : nil

        guard computesInput else {
            return (nil, scaleGradient, shiftGradient)
        }
        // With n = (x - mean) / d and d = sqrt(variance) + epsilon:
        // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
        let normalizedGradient = (outputGradient * scale).reducingBroadcast(to: input.shape)
        let meanGradient = normalizedGradient.reduceMean(along: axes).view(as: keptShape)
        let correlation = (normalizedGradient * normalized).reduceMean(along: axes).view(as: keptShape) * divisor / standardDeviation
        let inputGradient = (normalizedGradient - meanGradient - normalized * correlation) / divisor
        return (inputGradient, scaleGradient, shiftGradient)
    }

    /// Computes the gradients of a normalization with fixed statistics, followed by a scale and a shift.
    static func fixedNormalizationGradients<N, Device>(
        input: Tensor<N, Device>,
        scale: Tensor<N, Device>,
        shiftShape: [Int],
        mean: Tensor<N, Device>,
        variance: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        epsilon: N,
        computesInput: Bool,
        computesScale: Bool,
        computesShift: Bool,
    ) -> (input: Tensor<N, Device>?, scale: Tensor<N, Device>?, shift: Tensor<N, Device>?) {
        let divisor = variance.sqrt() + Tensor(epsilon)
        let inputGradient = computesInput ? (outputGradient * scale / divisor).reducingBroadcast(to: input.shape) : nil
        let scaleGradient = computesScale ? (outputGradient * (input - mean) / divisor).reducingBroadcast(to: scale.shape) : nil
        let shiftGradient = computesShift ? outputGradient.reducingBroadcast(to: shiftShape) : nil
        return (inputGradient, scaleGradient, shiftGradient)
    }
}

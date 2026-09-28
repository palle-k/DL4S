//
//  ComposedNormalization.swift
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

// MARK: Composed gradients

extension Composed {
    static func varianceBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, axes: [Int], inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let centered = input - input.reduceMean(along: axes).view(as: keptShape)
        let factor = Tensor<N, Device>(N(2) / N(ShapeUtil.elementCount(of: input.shape, along: axes)))
        inputGradient.add(factor * centered * outputGradient.view(as: keptShape))
    }

    /// Normalizes the input along the axes to `(input - mean) / (sqrt(variance) + epsilon)`.
    ///
    /// - Returns: The normalized input, and the mean and the biased variance with the shape of the input, where every reduced axis has the size 1.
    static func normalized<N, Device>(_ input: Tensor<N, Device>, along axes: [Int], epsilon: N) -> (normalized: Tensor<N, Device>, mean: Tensor<N, Device>, variance: Tensor<N, Device>) {
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let mean = input.reduceMean(along: axes).view(as: keptShape)
        let variance = (input * input).reduceMean(along: axes).view(as: keptShape) - mean * mean
        return ((input - mean) / (variance.sqrt() + Tensor(epsilon)), mean, variance)
    }

    /// Computes the gradients of a normalization with the statistics of the input along the given axes,
    /// followed by a scale and a shift.
    static func normalizationBackward<N, Device>(
        input: Tensor<N, Device>,
        scale: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        axes: [Int],
        epsilon: N,
        inputGradient: inout GradientAccumulator<N, Device>,
        scaleGradient: inout GradientAccumulator<N, Device>,
        shiftGradient: inout GradientAccumulator<N, Device>,
    ) {
        if shiftGradient.isRequested {
            shiftGradient.add(outputGradient.reducingBroadcast(to: shiftGradient.shape))
        }
        guard inputGradient.isRequested || scaleGradient.isRequested else {
            return
        }
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let centered = input - input.reduceMean(along: axes).view(as: keptShape)
        // The variance of the centered values equals the variance of the forward pass up to rounding, and needs fewer operations.
        let standardDeviation = (centered * centered).reduceMean(along: axes).view(as: keptShape).sqrt()
        let divisor = standardDeviation + Tensor(epsilon)
        let normalized = centered / divisor

        if scaleGradient.isRequested {
            scaleGradient.add((outputGradient * normalized).reducingBroadcast(to: scale.shape))
        }
        guard inputGradient.isRequested else {
            return
        }
        // With n = (x - mean) / d and d = sqrt(variance) + epsilon:
        // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
        let normalizedGradient = (outputGradient * scale).reducingBroadcast(to: input.shape)
        let meanGradient = normalizedGradient.reduceMean(along: axes).view(as: keptShape)
        let correlation = (normalizedGradient * normalized).reduceMean(along: axes).view(as: keptShape) * divisor / standardDeviation
        inputGradient.add((normalizedGradient - meanGradient - normalized * correlation) / divisor)
    }

    /// Computes the gradients of a normalization with fixed statistics, followed by a scale and a shift.
    static func fixedNormalizationBackward<N, Device>(
        input: Tensor<N, Device>,
        scale: Tensor<N, Device>,
        mean: Tensor<N, Device>,
        variance: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        epsilon: N,
        inputGradient: inout GradientAccumulator<N, Device>,
        scaleGradient: inout GradientAccumulator<N, Device>,
        shiftGradient: inout GradientAccumulator<N, Device>,
    ) {
        let divisor = variance.sqrt() + Tensor(epsilon)
        if inputGradient.isRequested {
            inputGradient.add((outputGradient * scale / divisor).reducingBroadcast(to: input.shape))
        }
        if scaleGradient.isRequested {
            scaleGradient.add((outputGradient * (input - mean) / divisor).reducingBroadcast(to: scale.shape))
        }
        if shiftGradient.isRequested {
            shiftGradient.add(outputGradient.reducingBroadcast(to: shiftGradient.shape))
        }
    }
}

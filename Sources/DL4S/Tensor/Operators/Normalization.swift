//
//  Normalization.swift
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

// MARK: Normalization

public extension Tensor {
    /// Normalizes every sample to a mean of 0 and a standard deviation of 1, then scales and shifts it.
    ///
    /// The result is `(self - mean) / (sqrt(variance) + epsilon) * scale + shift`.
    /// The statistics are computed along the trailing axes that `scale` has. Every leading axis is a sample axis.
    /// With a scale of the shape [hidden], a [batch, sequence, hidden] tensor is normalized per sequence element.
    ///
    /// - Parameters:
    ///   - scale: Scale, with the shape of the trailing axes of the tensor
    ///   - shift: Shift, with the shape of the trailing axes of the tensor
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: Normalized tensor with the shape of the tensor
    func layerNormalized(scale: Self, shift: Self, epsilon: Element = Element(1e-5)) -> Self {
        precondition(Array(shape.suffix(scale.dim)) == scale.shape, "The scale must have the shape of the trailing axes of the tensor.")
        precondition(shift.shape == scale.shape, "The shift must have the shape of the scale.")
        let result = Device.FusedOperations.layerNormalization(input: self, scale: scale, shift: shift, epsilon: epsilon)

        return result.attachingContext(tag: "layerNorm", sources: [self, scale, shift]) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.normalizationGradients(
                    input: self,
                    scale: scale,
                    shiftShape: shift.shape,
                    outputGradient: resultGradient,
                    axes: Composed.layerNormalizationAxes(input: self, scale: scale),
                    epsilon: epsilon,
                    computesInput: self.requiresGradient,
                    computesScale: scale.requiresGradient,
                    computesShift: shift.requiresGradient,
                )
            } else {
                Device.FusedOperations.layerNormalizationBackward(input: self, scale: scale, shift: shift, outputGradient: resultGradient, epsilon: epsilon)
            }
            return [gradients.input, gradients.scale, gradients.shift]
        }
    }

    /// Normalizes the tensor along the batch axis (axis 0) with the statistics of the batch, then scales and shifts it.
    ///
    /// The result is `(self - mean) / (sqrt(variance) + epsilon) * scale + shift`.
    ///
    /// - Parameters:
    ///   - scale: Scale, broadcastable to the shape of the tensor without the batch axis
    ///   - shift: Shift, broadcastable to the shape of the tensor without the batch axis
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: The normalized tensor, and the mean and the biased variance of the batch. The statistics have no gradient.
    func batchNormalized(scale: Self, shift: Self, epsilon: Element = Element(1e-5)) -> (output: Self, mean: Self, variance: Self) {
        let (result, mean, variance) = Device.FusedOperations.batchNormalization(input: self, scale: scale, shift: shift, epsilon: epsilon)

        let output = result.attachingContext(tag: "batchNorm", sources: [self, scale, shift]) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.normalizationGradients(
                    input: self,
                    scale: scale,
                    shiftShape: shift.shape,
                    outputGradient: resultGradient,
                    axes: [0],
                    epsilon: epsilon,
                    computesInput: self.requiresGradient,
                    computesScale: scale.requiresGradient,
                    computesShift: shift.requiresGradient,
                )
            } else {
                Device.FusedOperations.batchNormalizationBackward(input: self, scale: scale, shift: shift, outputGradient: resultGradient, epsilon: epsilon)
            }
            return [gradients.input, gradients.scale, gradients.shift]
        }
        return (output, mean, variance)
    }

    /// Normalizes the tensor with the given statistics, then scales and shifts it.
    ///
    /// The result is `(self - mean) / (sqrt(variance) + epsilon) * scale + shift`. Use it for inference with statistics
    /// that were collected during training.
    ///
    /// - Parameters:
    ///   - scale: Scale, broadcastable to the shape of the tensor without the batch axis
    ///   - shift: Shift, broadcastable to the shape of the tensor without the batch axis
    ///   - mean: Mean, broadcastable to the shape of the tensor without the batch axis. It gets no gradient.
    ///   - variance: Variance, broadcastable to the shape of the tensor without the batch axis. It gets no gradient.
    ///   - epsilon: Value added to the standard deviation
    /// - Returns: Normalized tensor with the shape of the tensor
    func batchNormalized(scale: Self, shift: Self, mean: Self, variance: Self, epsilon: Element = Element(1e-5)) -> Self {
        let mean = mean.detached()
        let variance = variance.detached()
        let result = Device.FusedOperations.batchNormalization(input: self, scale: scale, shift: shift, mean: mean, variance: variance, epsilon: epsilon)

        return result.attachingContext(tag: "batchNorm", sources: [self, scale, shift]) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.fixedNormalizationGradients(
                    input: self,
                    scale: scale,
                    shiftShape: shift.shape,
                    mean: mean,
                    variance: variance,
                    outputGradient: resultGradient,
                    epsilon: epsilon,
                    computesInput: self.requiresGradient,
                    computesScale: scale.requiresGradient,
                    computesShift: shift.requiresGradient,
                )
            } else {
                Device.FusedOperations.batchNormalizationBackward(input: self, scale: scale, shift: shift, mean: mean, variance: variance, outputGradient: resultGradient, epsilon: epsilon)
            }
            return [gradients.input, gradients.scale, gradients.shift]
        }
    }
}

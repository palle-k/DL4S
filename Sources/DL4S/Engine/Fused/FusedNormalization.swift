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
    static func variance<N: NumericType>(input: ShapedBuffer<N, Device>, axes: [Int], result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // mean((input - mean(input))²), which cannot be negative, as mean(input²) - mean(input)² can be after rounding
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let mean = math.temporary(keptShape)
        let squares = math.temporary(input.shape)
        math.mean(input, along: axes, into: mean)
        math.subtract(input, mean, into: squares)
        math.multiply(squares, squares, into: squares)
        math.mean(squares, along: axes, into: result)
    }

    static func varianceBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, axes: [Int], inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // 2 / n * (input - mean(input)) * outputGradient
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let count = axes.map { input.shape[$0] }.reduce(1, *)
        math.write(inputGradient) { dx in
            let mean = math.temporary(keptShape)
            math.mean(input, along: axes, into: mean)
            math.subtract(input, mean, into: dx)
            math.multiply(dx, outputGradient.reshaped(to: keptShape), into: dx)
            math.multiply(dx, N(2) / N(count), into: dx)
        }
    }

    static func layerNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let axes = Array(input.dim - scale.dim ..< input.dim)
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let mean = math.temporary(keptShape)
        let deviation = math.temporary(keptShape)
        let squares = math.temporary(input.shape)
        // (input - mean) / (sqrt(variance) + epsilon) * scale + shift, with the centered values in the result
        math.mean(input, along: axes, into: mean)
        math.subtract(input, mean, into: result)
        math.multiply(result, result, into: squares)
        math.mean(squares, along: axes, into: deviation)
        math.sqrt(deviation, into: deviation)
        math.add(deviation, epsilon, into: deviation)
        math.divide(result, deviation, into: result)
        math.multiply(result, scale, into: result)
        math.add(result, shift, into: result)
    }

    static func layerNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        scale: ShapedBuffer<N, Device>,
        shift: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        epsilon: N,
        inputGradient: GradientBuffer<N, Device>?,
        scaleGradient: GradientBuffer<N, Device>?,
        shiftGradient: GradientBuffer<N, Device>?,
    ) {
        normalizationBackward(
            input: input,
            scale: scale,
            outputGradient: outputGradient,
            axes: Array(input.dim - scale.dim ..< input.dim),
            epsilon: epsilon,
            inputGradient: inputGradient,
            scaleGradient: scaleGradient,
            shiftGradient: shiftGradient,
        )
    }

    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>, mean: MutableShapedBuffer<N, Device>, variance: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The statistics have the shape of the input without the batch axis, which broadcasts to the input.
        // The variance of the centered values cannot be negative, which mean(input²) - mean(input)² can be after rounding.
        let squares = math.temporary(input.shape)
        let deviation = math.temporary(mean.shape)
        math.mean(input, along: [0], into: mean)
        math.subtract(input, mean, into: result)
        math.multiply(result, result, into: squares)
        math.mean(squares, along: [0], into: variance)
        math.sqrt(variance, into: deviation)
        math.add(deviation, epsilon, into: deviation)
        math.divide(result, deviation, into: result)
        math.multiply(result, scale, into: result)
        math.add(result, shift, into: result)
    }

    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        scale: ShapedBuffer<N, Device>,
        shift: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        epsilon: N,
        inputGradient: GradientBuffer<N, Device>?,
        scaleGradient: GradientBuffer<N, Device>?,
        shiftGradient: GradientBuffer<N, Device>?,
    ) {
        normalizationBackward(
            input: input,
            scale: scale,
            outputGradient: outputGradient,
            axes: [0],
            epsilon: epsilon,
            inputGradient: inputGradient,
            scaleGradient: scaleGradient,
            shiftGradient: shiftGradient,
        )
    }

    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>, mean: ShapedBuffer<N, Device>, variance: ShapedBuffer<N, Device>, epsilon: N, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let divisor = math.temporary(variance.shape)
        math.sqrt(variance, into: divisor)
        math.add(divisor, epsilon, into: divisor)
        math.subtract(input, mean, into: result)
        math.divide(result, divisor, into: result)
        math.multiply(result, scale, into: result)
        math.add(result, shift, into: result)
    }

    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        scale: ShapedBuffer<N, Device>,
        shift: ShapedBuffer<N, Device>,
        mean: ShapedBuffer<N, Device>,
        variance: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        epsilon: N,
        inputGradient: GradientBuffer<N, Device>?,
        scaleGradient: GradientBuffer<N, Device>?,
        shiftGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let divisor = math.temporary(variance.shape)
        math.sqrt(variance, into: divisor)
        math.add(divisor, epsilon, into: divisor)
        // outputGradient * scale / divisor
        math.write(inputGradient) { dx in
            math.multiply(outputGradient, scale, into: dx)
            math.divide(dx, divisor, into: dx)
        }
        if scaleGradient != nil {
            // outputGradient * (input - mean) / divisor
            let products = math.temporary(input.shape)
            math.subtract(input, mean, into: products)
            math.divide(products, divisor, into: products)
            math.multiply(products, outputGradient, into: products)
            math.writeSum(of: products, into: scaleGradient)
        }
        math.writeSum(of: outputGradient, into: shiftGradient)
    }
}

extension FusedOperationsType {
    /// Computes the gradients of a normalization with the statistics of the input along the given axes, followed by a scale and a shift.
    static func normalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        scale: ShapedBuffer<N, Device>,
        outputGradient: ShapedBuffer<N, Device>,
        axes: [Int],
        epsilon: N,
        inputGradient: GradientBuffer<N, Device>?,
        scaleGradient: GradientBuffer<N, Device>?,
        shiftGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.writeSum(of: outputGradient, into: shiftGradient)
        guard inputGradient != nil || scaleGradient != nil else {
            return
        }
        let keptShape = ShapeUtil.keptShape(of: input.shape, along: axes)
        let statistic = math.temporary(keptShape)
        let standardDeviation = math.temporary(keptShape)
        let divisor = math.temporary(keptShape)
        let normalized = math.temporary(input.shape)
        let products = math.temporary(input.shape)
        // The variance of the centered values equals the variance of the forward pass up to rounding, and needs fewer operations.
        math.mean(input, along: axes, into: statistic)
        math.subtract(input, statistic, into: normalized)
        math.multiply(normalized, normalized, into: products)
        math.mean(products, along: axes, into: standardDeviation)
        math.sqrt(standardDeviation, into: standardDeviation)
        math.add(standardDeviation, epsilon, into: divisor)
        math.divide(normalized, divisor, into: normalized)
        if scaleGradient != nil {
            math.multiply(outputGradient, normalized, into: products)
            math.writeSum(of: products, into: scaleGradient)
        }
        // With n = (x - mean) / d and d = sqrt(variance) + epsilon:
        // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
        math.write(inputGradient) { dx in
            let normalizedGradient = products
            math.multiply(outputGradient, scale, into: normalizedGradient)
            math.mean(normalizedGradient, along: axes, into: statistic)
            math.subtract(normalizedGradient, statistic, into: dx)
            math.multiply(normalizedGradient, normalized, into: normalizedGradient)
            math.mean(normalizedGradient, along: axes, into: statistic)
            math.multiply(statistic, divisor, into: statistic)
            // Rows of equal values have the normalized values 0 and the statistic 0. The smallest normal number keeps their
            // quotient at 0, where 0 / 0 would give NaN, and does not change the other quotients.
            math.add(standardDeviation, N(Float.leastNormalMagnitude), into: standardDeviation)
            math.divide(statistic, standardDeviation, into: statistic)
            math.multiply(normalized, statistic, into: normalizedGradient)
            math.subtract(dx, normalizedGradient, into: dx)
            math.divide(dx, divisor, into: dx)
        }
    }
}

extension FusedOperationsType {
    /// Checks that the scale and the shift of a layer normalization have the shape of the trailing axes of the input.
    static func checkLayerNormalizationShapes<N>(input: ShapedBuffer<N, Device>, scale: ShapedBuffer<N, Device>, shift: ShapedBuffer<N, Device>) {
        precondition(scale.dim <= input.dim && Array(input.shape.suffix(scale.dim)) == scale.shape, "The scale must have the shape of the trailing axes of the input.")
        precondition(shift.shape == scale.shape, "The shift must have the shape of the scale.")
    }
}

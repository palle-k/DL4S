//
//  CPUFusedNormalization.swift
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

// Layer normalization works on one row at a time, so a row stays in the cache for all passes over it.
// Batch normalization reduces along the batch axis: it streams over the rows of the batch and keeps one
// accumulator per column.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func layerNormalization<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, epsilon: N) -> Tensor<N, CPU> {
        guard let rowLength = layerNormalizationRowLength(input: input, scale: scale, shift: shift) else {
            return DefaultFusedOperations<CPU>.layerNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (x, gamma, beta) = (input.elementPointer, scale.elementPointer, shift.elementPointer)
        let inverseLength = 1 / N(rowLength)
        CPUKernels.withScratch(N.self, count: rowLength) { squares in
            for row in 0 ..< input.count / rowLength {
                let values = x + row * rowLength
                let output = y + row * rowLength
                let mean = CPUKernels.sum(values, count: rowLength) * inverseLength
                for j in 0 ..< rowLength {
                    squares[j] = values[j] * values[j]
                }
                let variance = CPUKernels.sum(squares, count: rowLength) * inverseLength - mean * mean
                let inverseDivisor = 1 / (variance.sqrt() + epsilon)
                for j in 0 ..< rowLength {
                    output[j] = (values[j] - mean) * inverseDivisor * gamma[j] + beta[j]
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func layerNormalizationBackward<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, epsilon: N) -> (input: Tensor<N, CPU>?, scale: Tensor<N, CPU>?, shift: Tensor<N, CPU>?) {
        guard let rowLength = layerNormalizationRowLength(input: input, scale: scale, shift: shift), outputGradient.shape == input.shape else {
            return DefaultFusedOperations<CPU>.layerNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon)
        }
        let (x, gamma, g) = (input.elementPointer, scale.elementPointer, outputGradient.elementPointer)
        let inputGradient = input.requiresGradient ? CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil
        let scaleGradient = scale.requiresGradient ? CPUKernels.makeZeroTensor(shape: scale.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil
        let shiftGradient = shift.requiresGradient ? CPUKernels.makeZeroTensor(shape: shift.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil
        let inverseLength = 1 / N(rowLength)

        CPUKernels.withScratch(N.self, count: 3 * rowLength) { scratch in
            let (normalized, products, normalizedGradient) = (scratch, scratch + rowLength, scratch + 2 * rowLength)
            for row in 0 ..< input.count / rowLength {
                let values = x + row * rowLength
                let gradient = g + row * rowLength
                let mean = CPUKernels.sum(values, count: rowLength) * inverseLength
                for j in 0 ..< rowLength {
                    normalized[j] = values[j] - mean
                    products[j] = normalized[j] * normalized[j]
                }
                // The variance of the centered values equals the variance of the forward pass up to rounding.
                let standardDeviation = (CPUKernels.sum(products, count: rowLength) * inverseLength).sqrt()
                let divisor = standardDeviation + epsilon
                let inverseDivisor = 1 / divisor
                for j in 0 ..< rowLength {
                    normalized[j] *= inverseDivisor
                }
                if let (_, dScale) = scaleGradient {
                    for j in 0 ..< rowLength {
                        dScale[j] += gradient[j] * normalized[j]
                    }
                }
                if let (_, dShift) = shiftGradient {
                    for j in 0 ..< rowLength {
                        dShift[j] += gradient[j]
                    }
                }
                guard let (_, dx) = inputGradient else {
                    continue
                }
                // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
                for j in 0 ..< rowLength {
                    normalizedGradient[j] = gradient[j] * gamma[j]
                    products[j] = normalizedGradient[j] * normalized[j]
                }
                let meanGradient = CPUKernels.sum(normalizedGradient, count: rowLength) * inverseLength
                let correlation = CPUKernels.sum(products, count: rowLength) * inverseLength * divisor / standardDeviation
                let rowGradient = dx + row * rowLength
                for j in 0 ..< rowLength {
                    rowGradient[j] = (normalizedGradient[j] - meanGradient - normalized[j] * correlation) * inverseDivisor
                }
            }
        }
        return (inputGradient?.0, scaleGradient?.0, shiftGradient?.0)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalization<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, epsilon: N) -> (output: Tensor<N, CPU>, mean: Tensor<N, CPU>, variance: Tensor<N, CPU>) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, input.count > 0,
              let gammaColumns = broadcastColumns(scale, to: columnShape),
              let betaColumns = broadcastColumns(shift, to: columnShape)
        else {
            return DefaultFusedOperations<CPU>.batchNormalization(input: input, scale: scale, shift: shift, epsilon: epsilon)
        }
        let batchSize = input.shape[0]
        let columns = input.count / batchSize
        let x = input.elementPointer
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (mean, means) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (variance, variances) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)

        for row in 0 ..< batchSize {
            let values = x + row * columns
            for j in 0 ..< columns {
                means[j] += values[j]
                variances[j] += values[j] * values[j]
            }
        }
        let inverseBatchSize = 1 / N(batchSize)
        for j in 0 ..< columns {
            means[j] *= inverseBatchSize
            variances[j] = variances[j] * inverseBatchSize - means[j] * means[j]
        }

        withExtendedLifetime((gammaColumns, betaColumns)) {
            let (gamma, beta) = (gammaColumns.elementPointer, betaColumns.elementPointer)
            CPUKernels.withScratch(N.self, count: 2 * columns) { scratch in
                // y = x * factor + offset with factor = gamma / d and offset = beta - mean * factor
                let (factors, offsets) = (scratch, scratch + columns)
                for j in 0 ..< columns {
                    factors[j] = gamma[j] / (variances[j].sqrt() + epsilon)
                    offsets[j] = beta[j] - means[j] * factors[j]
                }
                for row in 0 ..< batchSize {
                    let (values, output) = (x + row * columns, y + row * columns)
                    for j in 0 ..< columns {
                        output[j] = values[j] * factors[j] + offsets[j]
                    }
                }
            }
        }
        return (result, mean, variance)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, epsilon: N) -> (input: Tensor<N, CPU>?, scale: Tensor<N, CPU>?, shift: Tensor<N, CPU>?) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, input.count > 0, outputGradient.shape == input.shape, let gammaColumns = broadcastColumns(scale, to: columnShape), shift.dim <= columnShape.count else {
            return DefaultFusedOperations<CPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, outputGradient: outputGradient, epsilon: epsilon)
        }
        let batchSize = input.shape[0]
        let columns = input.count / batchSize
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let gamma = gammaColumns.elementPointer
        let inverseBatchSize = 1 / N(batchSize)
        let inputGradient = input.requiresGradient ? CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil

        let (scaleColumns, scaleSums) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (shiftColumns, shiftSums) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)

        withExtendedLifetime(gammaColumns) {
            CPUKernels.withScratch(N.self, count: 6 * columns) { scratch in
                let means = scratch
                let inverseDivisors = scratch + columns
                let correlationFactors = scratch + 2 * columns
                let gradientSums = scratch + 3 * columns
                let productSums = scratch + 4 * columns
                let squares = scratch + 5 * columns
                for pointer in [means, gradientSums, productSums, squares] {
                    CPUKernels.fill(pointer, with: 0, count: columns)
                }

                for row in 0 ..< batchSize {
                    let values = x + row * columns
                    for j in 0 ..< columns {
                        means[j] += values[j]
                    }
                }
                for j in 0 ..< columns {
                    means[j] *= inverseBatchSize
                }
                // The variance of the centered values equals the variance of the forward pass up to rounding.
                for row in 0 ..< batchSize {
                    let values = x + row * columns
                    for j in 0 ..< columns {
                        let centered = values[j] - means[j]
                        squares[j] += centered * centered
                    }
                }
                for j in 0 ..< columns {
                    let standardDeviation = (squares[j] * inverseBatchSize).sqrt()
                    let divisor = standardDeviation + epsilon
                    inverseDivisors[j] = 1 / divisor
                    // Holds d / sqrt(variance) until the sums of the gradients are known.
                    correlationFactors[j] = divisor / standardDeviation
                }

                for row in 0 ..< batchSize {
                    let (values, gradients) = (x + row * columns, g + row * columns)
                    for j in 0 ..< columns {
                        let normalized = (values[j] - means[j]) * inverseDivisors[j]
                        let gradient = gradients[j]
                        let normalizedGradient = gradient * gamma[j]
                        gradientSums[j] += normalizedGradient
                        productSums[j] += normalizedGradient * normalized
                        scaleSums[j] += gradient * normalized
                        shiftSums[j] += gradient
                    }
                }

                guard let (_, dx) = inputGradient else {
                    return
                }
                for j in 0 ..< columns {
                    gradientSums[j] *= inverseBatchSize
                    correlationFactors[j] = productSums[j] * inverseBatchSize * correlationFactors[j]
                }
                // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
                for row in 0 ..< batchSize {
                    let (values, gradients, rowGradient) = (x + row * columns, g + row * columns, dx + row * columns)
                    for j in 0 ..< columns {
                        let normalized = (values[j] - means[j]) * inverseDivisors[j]
                        let normalizedGradient = gradients[j] * gamma[j]
                        rowGradient[j] = (normalizedGradient - gradientSums[j] - normalized * correlationFactors[j]) * inverseDivisors[j]
                    }
                }
            }
        }
        return (
            inputGradient?.0,
            scale.requiresGradient ? scaleColumns.reducingBroadcast(to: scale.shape) : nil,
            shift.requiresGradient ? shiftColumns.reducingBroadcast(to: shift.shape) : nil,
        )
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalization<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, mean: Tensor<N, CPU>, variance: Tensor<N, CPU>, epsilon: N) -> Tensor<N, CPU> {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, input.count > 0, let affine = fixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon) else {
            return DefaultFusedOperations<CPU>.batchNormalization(input: input, scale: scale, shift: shift, mean: mean, variance: variance, epsilon: epsilon)
        }
        let columns = input.count / input.shape[0]
        let x = input.elementPointer
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        withExtendedLifetime(affine) {
            let (factors, offsets) = (affine.factors.elementPointer, affine.offsets.elementPointer)
            for row in 0 ..< input.shape[0] {
                let (values, output) = (x + row * columns, y + row * columns)
                for j in 0 ..< columns {
                    output[j] = values[j] * factors[j] + offsets[j]
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalizationBackward<N: NumericType>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>, mean: Tensor<N, CPU>, variance: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, epsilon: N) -> (input: Tensor<N, CPU>?, scale: Tensor<N, CPU>?, shift: Tensor<N, CPU>?) {
        let columnShape = Array(input.shape.dropFirst())
        guard input.dim >= 1, input.count > 0, outputGradient.shape == input.shape, shift.dim <= columnShape.count,
              let affine = fixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon),
              let meanColumns = broadcastColumns(mean, to: columnShape)
        else {
            return DefaultFusedOperations<CPU>.batchNormalizationBackward(input: input, scale: scale, shift: shift, mean: mean, variance: variance, outputGradient: outputGradient, epsilon: epsilon)
        }
        let columns = input.count / input.shape[0]
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let inputGradient = input.requiresGradient ? CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil
        let (scaleColumns, scaleSums) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (shiftColumns, shiftSums) = CPUKernels.makeZeroTensor(shape: columnShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)

        withExtendedLifetime((affine, meanColumns)) {
            let (factors, inverseDivisors, means) = (affine.factors.elementPointer, affine.inverseDivisors.elementPointer, meanColumns.elementPointer)
            for row in 0 ..< input.shape[0] {
                let (values, gradients) = (x + row * columns, g + row * columns)
                for j in 0 ..< columns {
                    let gradient = gradients[j]
                    scaleSums[j] += gradient * (values[j] - means[j]) * inverseDivisors[j]
                    shiftSums[j] += gradient
                }
                if let (_, dx) = inputGradient {
                    let rowGradient = dx + row * columns
                    for j in 0 ..< columns {
                        rowGradient[j] = gradients[j] * factors[j]
                    }
                }
            }
        }
        return (
            inputGradient?.0,
            scale.requiresGradient ? scaleColumns.reducingBroadcast(to: scale.shape) : nil,
            shift.requiresGradient ? shiftColumns.reducingBroadcast(to: shift.shape) : nil,
        )
    }
}

extension CPUFusedOperations {
    /// Length of the normalized rows, or nil when the scale and the shift do not have the shape of the trailing axes of the input.
    static func layerNormalizationRowLength<N>(input: Tensor<N, CPU>, scale: Tensor<N, CPU>, shift: Tensor<N, CPU>) -> Int? {
        guard input.count > 0, scale.count > 0, scale.dim <= input.dim, Array(input.shape.suffix(scale.dim)) == scale.shape, shift.shape == scale.shape else {
            return nil
        }
        return scale.count
    }

    /// Broadcasts a tensor to the shape of the columns, or returns nil when its shape does not broadcast to it.
    static func broadcastColumns<N: NumericType>(_ tensor: Tensor<N, CPU>, to columnShape: [Int]) -> Tensor<N, CPU>? {
        let tensor = tensor.detached()
        if tensor.shape == columnShape {
            return tensor
        }
        guard tensor.dim <= columnShape.count, zip(tensor.shape.reversed(), columnShape.reversed()).allSatisfy({ $0 == $1 || $0 == 1 }) else {
            return nil
        }
        return tensor + Tensor(repeating: 0, shape: columnShape)
    }

    /// The factors `scale / (sqrt(variance) + epsilon)`, their divisors, and the offsets `shift - mean * factor` of a normalization
    /// with fixed statistics, with the shape of the columns.
    static func fixedNormalizationColumns<N: NumericType>(
        scale: Tensor<N, CPU>,
        shift: Tensor<N, CPU>,
        mean: Tensor<N, CPU>,
        variance: Tensor<N, CPU>,
        columnShape: [Int],
        epsilon: N,
    ) -> (factors: Tensor<N, CPU>, inverseDivisors: Tensor<N, CPU>, offsets: Tensor<N, CPU>)? {
        guard let gamma = broadcastColumns(scale, to: columnShape),
              let beta = broadcastColumns(shift, to: columnShape),
              let means = broadcastColumns(mean, to: columnShape),
              let variances = broadcastColumns(variance, to: columnShape)
        else {
            return nil
        }
        let inverseDivisors = 1 / (variances.sqrt() + Tensor(epsilon))
        let factors = gamma * inverseDivisors
        return (factors, inverseDivisors, beta - means * factors)
    }
}

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
    static func layerNormalization<N: NumericType>(input: ShapedBuffer<N, CPU>, scale: ShapedBuffer<N, CPU>, shift: ShapedBuffer<N, CPU>, epsilon: N, result: MutableShapedBuffer<N, CPU>) {
        checkLayerNormalizationShapes(input: input, scale: scale, shift: shift)
        let rowLength = scale.count
        let (x, y) = (input.elementPointer, result.elementPointer)
        let (gamma, beta) = (scale.elementPointer, shift.elementPointer)
        let inverseLength = 1 / N(rowLength)
        for row in 0 ..< input.count / rowLength {
            let (values, output) = (x + row * rowLength, y + row * rowLength)
            let mean = CPUKernels.sum(values, count: rowLength) * inverseLength
            // The variance of the centered values cannot be negative, which E[x²] - E[x]² can be after rounding.
            for j in 0 ..< rowLength {
                output[j] = values[j] - mean
            }
            let variance = CPUKernels.dot(output, output, count: rowLength) * inverseLength
            let inverseDivisor = 1 / (variance.sqrt() + epsilon)
            for j in 0 ..< rowLength {
                output[j] = output[j] * inverseDivisor * gamma[j] + beta[j]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func layerNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        scale: ShapedBuffer<N, CPU>,
        shift: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, CPU>?,
        scaleGradient: GradientBuffer<N, CPU>?,
        shiftGradient: GradientBuffer<N, CPU>?,
    ) {
        checkLayerNormalizationShapes(input: input, scale: scale, shift: shift)
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let rowLength = scale.count
        let (x, gamma, g) = (input.elementPointer, scale.elementPointer, outputGradient.elementPointer)
        // Every row adds to the scale and shift gradients, and writes its own row of the input gradient.
        let dScale = scaleGradient?.elementsToAddTo()
        let dShift = shiftGradient?.elementsToAddTo()
        let dx = inputGradient?.elementsToWrite()
        let inverseLength = 1 / N(rowLength)
        let normalized = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        let normalizedGradient = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        let rowGradient = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        defer {
            normalized.deallocate()
            normalizedGradient.deallocate()
            rowGradient.deallocate()
        }

        for row in 0 ..< input.count / rowLength {
            let (values, gradient) = (x + row * rowLength, g + row * rowLength)
            guard dScale != nil || dx != nil else {
                if let dShift {
                    for j in 0 ..< rowLength {
                        dShift[j] += gradient[j]
                    }
                }
                continue
            }
            let mean = CPUKernels.sum(values, count: rowLength) * inverseLength
            for j in 0 ..< rowLength {
                normalized[j] = values[j] - mean
            }
            // The variance of the centered values equals the variance of the forward pass up to rounding.
            let standardDeviation = (CPUKernels.dot(normalized, normalized, count: rowLength) * inverseLength).sqrt()
            let divisor = standardDeviation + epsilon
            let inverseDivisor = 1 / divisor
            // The normalized values n, their gradient dn, and the gradients of the scale and the shift. Training updates both
            // parameters, so that case gets one loop.
            if let dScale, let dShift {
                for j in 0 ..< rowLength {
                    let (n, dy) = (normalized[j] * inverseDivisor, gradient[j])
                    normalized[j] = n
                    normalizedGradient[j] = dy * gamma[j]
                    dScale[j] += dy * n
                    dShift[j] += dy
                }
            } else {
                for j in 0 ..< rowLength {
                    normalized[j] *= inverseDivisor
                    normalizedGradient[j] = gradient[j] * gamma[j]
                }
                if let dScale {
                    for j in 0 ..< rowLength {
                        dScale[j] += gradient[j] * normalized[j]
                    }
                }
                if let dShift {
                    for j in 0 ..< rowLength {
                        dShift[j] += gradient[j]
                    }
                }
            }
            guard let (dx, beta) = dx else {
                continue
            }
            // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
            let meanGradient = CPUKernels.sum(normalizedGradient, count: rowLength) * inverseLength
            // A row of equal values has the normalized values 0, so the term is 0, and the division would give NaN.
            let correlation = standardDeviation > 0 ? CPUKernels.dot(normalizedGradient, normalized, count: rowLength) * inverseLength * divisor / standardDeviation : 0
            // Without an accumulated gradient, the row is written directly.
            let target = beta == 0 ? dx + row * rowLength : rowGradient
            for j in 0 ..< rowLength {
                target[j] = (normalizedGradient[j] - meanGradient - normalized[j] * correlation) * inverseDivisor
            }
            if beta != 0 {
                CPUKernels.store(rowGradient, into: dx + row * rowLength, beta: beta, count: rowLength)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, CPU>, scale: ShapedBuffer<N, CPU>, shift: ShapedBuffer<N, CPU>, epsilon: N, result: MutableShapedBuffer<N, CPU>, mean: MutableShapedBuffer<N, CPU>, variance: MutableShapedBuffer<N, CPU>) {
        let columnShape = Array(input.shape.dropFirst())
        let math = BufferMath<N, CPU>()
        defer {
            math.release()
        }
        precondition(input.dim >= 1, "The input must have a batch axis.")
        precondition(mean.shape == columnShape && variance.shape == columnShape, "The mean and the variance must have the shape of the input without the batch axis.")
        let (gamma, beta) = (columns(of: scale, shape: columnShape, math: math), columns(of: shift, shape: columnShape, math: math))
        let batchSize = input.shape[0]
        let columns = input.count / batchSize
        let (x, y) = (input.elementPointer, result.elementPointer)
        let (means, variances) = (mean.elementPointer, variance.elementPointer)
        CPUKernels.fill(means, with: 0, count: columns)
        CPUKernels.fill(variances, with: 0, count: columns)

        for row in 0 ..< batchSize {
            let values = x + row * columns
            for j in 0 ..< columns {
                means[j] += values[j]
            }
        }
        let inverseBatchSize = 1 / N(batchSize)
        for j in 0 ..< columns {
            means[j] *= inverseBatchSize
        }
        // The variance of the centered values cannot be negative, which E[x²] - E[x]² can be after rounding.
        for row in 0 ..< batchSize {
            let values = x + row * columns
            for j in 0 ..< columns {
                let centered = values[j] - means[j]
                variances[j] += centered * centered
            }
        }
        for j in 0 ..< columns {
            variances[j] *= inverseBatchSize
        }

        // y = x * factor + offset with factor = gamma / d and offset = beta - mean * factor
        let factors = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let offsets = UnsafeMutablePointer<N>.allocate(capacity: columns)
        defer {
            factors.deallocate()
            offsets.deallocate()
        }
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

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        scale: ShapedBuffer<N, CPU>,
        shift: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, CPU>?,
        scaleGradient: GradientBuffer<N, CPU>?,
        shiftGradient: GradientBuffer<N, CPU>?,
    ) {
        let columnShape = Array(input.shape.dropFirst())
        let math = BufferMath<N, CPU>()
        defer {
            math.release()
        }
        precondition(input.dim >= 1, "The input must have a batch axis.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        precondition(ShapeUtil.broadcasts(shift.shape, to: columnShape), "The shift must be broadcastable to the shape of the input without the batch axis.")
        let gamma = columns(of: scale, shape: columnShape, math: math)
        let batchSize = input.shape[0]
        let columns = input.count / batchSize
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let inverseBatchSize = 1 / N(batchSize)

        let means = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let inverseDivisors = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let correlationFactors = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let gradientSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let productSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let squares = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let scaleSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let shiftSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        defer {
            for buffer in [means, inverseDivisors, correlationFactors, gradientSums, productSums, squares, scaleSums, shiftSums] {
                buffer.deallocate()
            }
        }
        for buffer in [means, gradientSums, productSums, squares, scaleSums, shiftSums] {
            CPUKernels.fill(buffer, with: 0, count: columns)
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
            // Holds d / sqrt(variance) until the sums of the gradients are known. A column of equal values has the
            // normalized values 0, so its term is 0, and the division would give NaN.
            correlationFactors[j] = standardDeviation > 0 ? divisor / standardDeviation : 0
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
        addColumnSums(scaleSums, columnShape: columnShape, to: scaleGradient, math: math)
        addColumnSums(shiftSums, columnShape: columnShape, to: shiftGradient, math: math)
        guard let inputGradient else {
            return
        }
        for j in 0 ..< columns {
            gradientSums[j] *= inverseBatchSize
            correlationFactors[j] = productSums[j] * inverseBatchSize * correlationFactors[j]
        }
        // dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d
        let (dx, beta) = inputGradient.elementsToWrite()
        for row in 0 ..< batchSize {
            let (values, gradients, rowGradient) = (x + row * columns, g + row * columns, dx + row * columns)
            // Without an accumulated gradient, the row is written directly.
            let target = beta == 0 ? rowGradient : squares
            for j in 0 ..< columns {
                let normalized = (values[j] - means[j]) * inverseDivisors[j]
                let normalizedGradient = gradients[j] * gamma[j]
                target[j] = (normalizedGradient - gradientSums[j] - normalized * correlationFactors[j]) * inverseDivisors[j]
            }
            if beta != 0 {
                CPUKernels.store(squares, into: rowGradient, beta: beta, count: columns)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalization<N: NumericType>(input: ShapedBuffer<N, CPU>, scale: ShapedBuffer<N, CPU>, shift: ShapedBuffer<N, CPU>, mean: ShapedBuffer<N, CPU>, variance: ShapedBuffer<N, CPU>, epsilon: N, result: MutableShapedBuffer<N, CPU>) {
        let columnShape = Array(input.shape.dropFirst())
        let math = BufferMath<N, CPU>()
        defer {
            math.release()
        }
        precondition(input.dim >= 1, "The input must have a batch axis.")
        let affine = FixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon, math: math)
        let columns = input.count / input.shape[0]
        let (x, y) = (input.elementPointer, result.elementPointer)
        let (factors, offsets) = (affine.factors, affine.offsets)
        for row in 0 ..< input.shape[0] {
            let (values, output) = (x + row * columns, y + row * columns)
            for j in 0 ..< columns {
                output[j] = values[j] * factors[j] + offsets[j]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func batchNormalizationBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        scale: ShapedBuffer<N, CPU>,
        shift: ShapedBuffer<N, CPU>,
        mean: ShapedBuffer<N, CPU>,
        variance: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        epsilon: N,
        inputGradient: GradientBuffer<N, CPU>?,
        scaleGradient: GradientBuffer<N, CPU>?,
        shiftGradient: GradientBuffer<N, CPU>?,
    ) {
        let columnShape = Array(input.shape.dropFirst())
        let math = BufferMath<N, CPU>()
        defer {
            math.release()
        }
        precondition(input.dim >= 1, "The input must have a batch axis.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let affine = FixedNormalizationColumns(scale: scale, shift: shift, mean: mean, variance: variance, columnShape: columnShape, epsilon: epsilon, math: math)
        let columns = input.count / input.shape[0]
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let dx = inputGradient?.elementsToWrite()
        let scaleSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let shiftSums = UnsafeMutablePointer<N>.allocate(capacity: columns)
        let rowGradient = UnsafeMutablePointer<N>.allocate(capacity: columns)
        defer {
            scaleSums.deallocate()
            shiftSums.deallocate()
            rowGradient.deallocate()
        }
        CPUKernels.fill(scaleSums, with: 0, count: columns)
        CPUKernels.fill(shiftSums, with: 0, count: columns)

        let (factors, inverseDivisors, means) = (affine.factors, affine.inverseDivisors, affine.means)
        for row in 0 ..< input.shape[0] {
            let (values, gradients) = (x + row * columns, g + row * columns)
            for j in 0 ..< columns {
                let gradient = gradients[j]
                scaleSums[j] += gradient * (values[j] - means[j]) * inverseDivisors[j]
                shiftSums[j] += gradient
            }
            if let (dx, beta) = dx {
                // Without an accumulated gradient, the row is written directly.
                let target = beta == 0 ? dx + row * columns : rowGradient
                for j in 0 ..< columns {
                    target[j] = gradients[j] * factors[j]
                }
                if beta != 0 {
                    CPUKernels.store(rowGradient, into: dx + row * columns, beta: beta, count: columns)
                }
            }
        }
        addColumnSums(scaleSums, columnShape: columnShape, to: scaleGradient, math: math)
        addColumnSums(shiftSums, columnShape: columnShape, to: shiftGradient, math: math)
    }
}

extension CPUFusedOperations {
    /// The elements of values that are broadcastable to the columns, repeated into an intermediate buffer of `math` when the shapes differ.
    static func columns<N: NumericType>(of values: ShapedBuffer<N, CPU>, shape columnShape: [Int], math: BufferMath<N, CPU>) -> UnsafePointer<N> {
        if values.shape == columnShape {
            return values.elementPointer
        }
        precondition(ShapeUtil.broadcasts(values.shape, to: columnShape), "The parameters must be broadcastable to the shape of the input without the batch axis.")
        let repeated = math.temporary(columnShape)
        CPUKernels.broadcast(values, into: repeated)
        return UnsafePointer(repeated.elementPointer)
    }

    /// Writes sums with the shape of the columns into the gradient of a parameter that broadcasts to the columns.
    static func addColumnSums<N: NumericType>(_ sums: UnsafePointer<N>, columnShape: [Int], to gradient: GradientBuffer<N, CPU>?, math: BufferMath<N, CPU>) {
        guard let gradient else {
            return
        }
        guard gradient.shape == columnShape else {
            let columns = math.temporary(columnShape)
            columns.elementPointer.update(from: sums, count: columns.count)
            math.writeSum(of: columns, into: gradient)
            return
        }
        gradient.write(sums)
    }
}

/// The columns of a normalization with fixed statistics: the factors `scale / (sqrt(variance) + epsilon)`, their divisors,
/// the offsets `shift - mean * factor`, and the means.
struct FixedNormalizationColumns<N: NumericType> {
    let factors: UnsafePointer<N>
    let inverseDivisors: UnsafePointer<N>
    let offsets: UnsafePointer<N>
    let means: UnsafePointer<N>

    /// Computes the columns in intermediate buffers of `math`. The parameters must be broadcastable to the columns.
    init(
        scale: ShapedBuffer<N, CPU>,
        shift: ShapedBuffer<N, CPU>,
        mean: ShapedBuffer<N, CPU>,
        variance: ShapedBuffer<N, CPU>,
        columnShape: [Int],
        epsilon: N,
        math: BufferMath<N, CPU>,
    ) {
        typealias Columns = CPUFusedOperations
        let (gamma, beta) = (Columns.columns(of: scale, shape: columnShape, math: math), Columns.columns(of: shift, shape: columnShape, math: math))
        let (means, variances) = (Columns.columns(of: mean, shape: columnShape, math: math), Columns.columns(of: variance, shape: columnShape, math: math))
        let count = columnShape.reduce(1, *)
        let (factors, inverseDivisors, offsets) = (math.temporary(columnShape).elementPointer, math.temporary(columnShape).elementPointer, math.temporary(columnShape).elementPointer)
        for j in 0 ..< count {
            inverseDivisors[j] = 1 / (variances[j].sqrt() + epsilon)
            factors[j] = gamma[j] * inverseDivisors[j]
            offsets[j] = beta[j] - means[j] * factors[j]
        }
        self.factors = UnsafePointer(factors)
        self.inverseDivisors = UnsafePointer(inverseDivisors)
        self.offsets = UnsafePointer(offsets)
        self.means = means
    }
}

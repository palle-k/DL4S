//
//  CPUFusedLoss.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.09.26.
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

// The categorical losses read one element per row, so their gradients are 0 except for one element per row.
// The kernels add these elements to the accumulated gradient, or to a new gradient that starts at 0.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func categoricalCrossEntropy<N: NumericType>(expected: ShapedBuffer<Int32, CPU>, actual: ShapedBuffer<N, CPU>, ignoreIndex: Int32, result: MutableShapedBuffer<N, CPU>) {
        let geometry = SelectionGeometry(expected: expected, actual: actual)
        var sum: N = 0
        geometry.forEachSelected(expected, ignoreIndex: ignoreIndex, actual: actual) { value, _ in
            sum += value.log()
        }
        result.elementPointer[0] = -sum / N(geometry.rows)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func categoricalCrossEntropyBackward<N: NumericType>(expected: ShapedBuffer<Int32, CPU>, actual: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, CPU>?) {
        guard let actualGradient else {
            return
        }
        precondition(outputGradient.count == 1, "The gradient of the loss must be a scalar.")
        let geometry = SelectionGeometry(expected: expected, actual: actual)
        let factor = -outputGradient.elementPointer[0] / N(geometry.rows)
        let da = actualGradient.elementsToAddTo()
        geometry.forEachSelected(expected, ignoreIndex: ignoreIndex, actual: actual) { value, position in
            da[position] += factor / value
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func categoricalNegativeLogLikelihood<N: NumericType>(expected: ShapedBuffer<Int32, CPU>, actual: ShapedBuffer<N, CPU>, ignoreIndex: Int32, result: MutableShapedBuffer<N, CPU>) {
        let geometry = SelectionGeometry(expected: expected, actual: actual)
        var sum: N = 0
        geometry.forEachSelected(expected, ignoreIndex: ignoreIndex, actual: actual) { value, _ in
            sum += value
        }
        result.elementPointer[0] = -sum / N(geometry.rows)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func categoricalNegativeLogLikelihoodBackward<N: NumericType>(expected: ShapedBuffer<Int32, CPU>, actual: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, CPU>?) {
        guard let actualGradient else {
            return
        }
        precondition(outputGradient.count == 1, "The gradient of the loss must be a scalar.")
        let geometry = SelectionGeometry(expected: expected, actual: actual)
        let factor = -outputGradient.elementPointer[0] / N(geometry.rows)
        let da = actualGradient.elementsToAddTo()
        geometry.forEachSelected(expected, ignoreIndex: ignoreIndex, actual: actual) { _, position in
            da[position] += factor
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func binaryCrossEntropy<N: NumericType>(expected: ShapedBuffer<N, CPU>, actual: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        precondition(expected.count == actual.count, "The expected and the predicted values must have the same number of elements.")
        let (e, a) = (expected.elementPointer, actual.elementPointer)
        let logarithms = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        let complementLogarithms = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            logarithms.deallocate()
            complementLogarithms.deallocate()
        }
        var sum: N = 0
        CPUKernels.forEachBlock(count: actual.count) { offset, length in
            let (eb, ab) = (e + offset, a + offset)
            for i in 0 ..< length {
                complementLogarithms[i] = 1 - ab[i]
            }
            CPUKernels.log(ab, into: logarithms, count: length)
            CPUKernels.log(complementLogarithms, into: complementLogarithms, count: length)
            // The logarithms are replaced by the terms of the loss, e * log(a) + (1 - e) * log(1 - a).
            for i in 0 ..< length {
                logarithms[i] = eb[i] * logarithms[i] + (1 - eb[i]) * complementLogarithms[i]
            }
            sum += CPUKernels.sum(logarithms, count: length)
        }
        result.elementPointer[0] = -sum / N(actual.count)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func binaryCrossEntropyBackward<N: NumericType>(
        expected: ShapedBuffer<N, CPU>,
        actual: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        expectedGradient: GradientBuffer<N, CPU>?,
        actualGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(expected.count == actual.count, "The expected and the predicted values must have the same number of elements.")
        precondition(outputGradient.count == 1, "The gradient of the loss must be a scalar.")
        let (e, a) = (expected.elementPointer, actual.elementPointer)
        let factor = outputGradient.elementPointer[0] / N(actual.count)
        if let actualGradient {
            actualGradient.writeBlocks { offset, length, da in
                let (eb, ab) = (e + offset, a + offset)
                for i in 0 ..< length {
                    let value = ab[i]
                    da[i] = factor * (value - eb[i]) / (value * (1 - value))
                }
            }
        }
        if let expectedGradient {
            let complementLogarithms = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
            defer {
                complementLogarithms.deallocate()
            }
            expectedGradient.writeBlocks { offset, length, de in
                let ab = a + offset
                for i in 0 ..< length {
                    complementLogarithms[i] = 1 - ab[i]
                }
                CPUKernels.log(ab, into: de, count: length)
                CPUKernels.log(complementLogarithms, into: complementLogarithms, count: length)
                for i in 0 ..< length {
                    de[i] = factor * (complementLogarithms[i] - de[i])
                }
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func meanSquaredError<N: NumericType>(expected: ShapedBuffer<N, CPU>, actual: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        // The kernel does not support broadcasting between the expected and the predicted values.
        guard expected.shape == actual.shape else {
            DefaultFusedOperations<CPU>.meanSquaredError(expected: expected, actual: actual, result: result)
            return
        }
        let (e, a) = (expected.elementPointer, actual.elementPointer)
        let squares = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            squares.deallocate()
        }
        var sum: N = 0
        CPUKernels.forEachBlock(count: actual.count) { offset, length in
            let (eb, ab) = (e + offset, a + offset)
            for i in 0 ..< length {
                let difference = eb[i] - ab[i]
                squares[i] = difference * difference
            }
            sum += CPUKernels.sum(squares, count: length)
        }
        result.elementPointer[0] = sum / N(meanSquaredErrorDivisor(expectedShape: expected.shape))
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func meanSquaredErrorBackward<N: NumericType>(
        expected: ShapedBuffer<N, CPU>,
        actual: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        expectedGradient: GradientBuffer<N, CPU>?,
        actualGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(outputGradient.count == 1, "The gradient of the loss must be a scalar.")
        // The kernel does not support broadcasting between the expected and the predicted values.
        guard expected.shape == actual.shape else {
            DefaultFusedOperations<CPU>.meanSquaredErrorBackward(expected: expected, actual: actual, outputGradient: outputGradient, expectedGradient: expectedGradient, actualGradient: actualGradient)
            return
        }
        let (e, a) = (expected.elementPointer, actual.elementPointer)
        let factor = 2 * outputGradient.elementPointer[0] / N(meanSquaredErrorDivisor(expectedShape: expected.shape))
        if let expectedGradient {
            expectedGradient.writeBlocks { offset, length, de in
                let (eb, ab) = (e + offset, a + offset)
                for i in 0 ..< length {
                    de[i] = factor * (eb[i] - ab[i])
                }
            }
        }
        if let actualGradient {
            actualGradient.writeBlocks { offset, length, da in
                let (eb, ab) = (e + offset, a + offset)
                for i in 0 ..< length {
                    da[i] = factor * (ab[i] - eb[i])
                }
            }
        }
    }
}

/// Shapes of a loss that selects one element per row of the predictions, at the position of the label of the row.
struct SelectionGeometry {
    /// Number of rows, including the rows with the ignored label
    let rows: Int
    /// Number of classes, the length of a row
    let classes: Int

    /// The shapes of the loss. The labels must have the shape of the predictions without the last axis.
    init<N>(expected: ShapedBuffer<Int32, CPU>, actual: ShapedBuffer<N, CPU>) {
        precondition(actual.dim >= 1 && expected.shape == Array(actual.shape.dropLast()), "The labels must have the shape of the predictions without the last axis.")
        rows = expected.count
        classes = actual.shape[actual.dim - 1]
    }

    /// Calls `body` with the selected element and its position in the predictions for every row whose label is not ignored.
    @inline(__always)
    func forEachSelected<N: NumericType>(_ expected: ShapedBuffer<Int32, CPU>, ignoreIndex: Int32, actual: ShapedBuffer<N, CPU>, _ body: (_ value: N, _ position: Int) -> Void) {
        let (labels, a) = (expected.elementPointer, actual.elementPointer)
        for row in 0 ..< rows {
            let label = labels[row]
            if label == ignoreIndex {
                continue
            }
            precondition(label >= 0 && Int(label) < classes, "The label \(label) of row \(row) is not a class.")
            let position = row * classes + Int(label)
            body(a[position], position)
        }
    }
}

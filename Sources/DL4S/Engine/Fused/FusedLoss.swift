//
//  FusedLoss.swift
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
    static func binaryCrossEntropy<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // -mean(expected * log(actual) + (1 - expected) * log(1 - actual)), with the elements in memory order
        let count = actual.count
        let (e, a) = (expected.reshaped(to: [count]), actual.reshaped(to: [count]))
        let terms = math.temporary([count])
        let complementTerms = math.temporary([count])
        let complementExpected = math.temporary([count])
        math.log(a, into: terms)
        math.multiply(terms, e, into: terms)
        math.subtract(1, a, into: complementTerms)
        math.log(complementTerms, into: complementTerms)
        math.subtract(1, e, into: complementExpected)
        math.multiply(complementTerms, complementExpected, into: complementTerms)
        math.add(terms, complementTerms, into: terms)
        math.mean(terms, along: [0], into: result)
        math.negate(result, into: result)
    }

    static func binaryCrossEntropyBackward<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, expectedGradient: GradientBuffer<N, Device>?, actualGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let count = actual.count
        let (e, a) = (expected.reshaped(to: [count]), actual.reshaped(to: [count]))
        let factor = math.temporary([])
        math.multiply(outputGradient, 1 / N(count), into: factor)
        // factor * (actual - expected) / (actual * (1 - actual))
        math.write(actualGradient) { da in
            let (gradient, divisor) = (da.reshaped(to: [count]), math.temporary([count]))
            math.subtract(a, e, into: gradient)
            math.subtract(1, a, into: divisor)
            math.multiply(divisor, a, into: divisor)
            math.divide(gradient, divisor, into: gradient)
            math.multiply(gradient, factor, into: gradient)
        }
        // factor * (log(1 - actual) - log(actual))
        math.write(expectedGradient) { de in
            let (gradient, complementLogarithms) = (de.reshaped(to: [count]), math.temporary([count]))
            math.subtract(1, a, into: complementLogarithms)
            math.log(complementLogarithms, into: complementLogarithms)
            math.log(a, into: gradient)
            math.subtract(complementLogarithms, gradient, into: gradient)
            math.multiply(gradient, factor, into: gradient)
        }
    }

    static func categoricalCrossEntropy<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, ignoreIndex: Int32, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The gather yields 0 for the rows with the ignored label, so these rows add 0 to the loss.
        let logarithms = math.temporary(actual.shape)
        math.log(actual, into: logarithms)
        categoricalNegativeLogLikelihood(expected: expected, actual: ShapedBuffer(logarithms), ignoreIndex: ignoreIndex, result: result)
    }

    static func categoricalCrossEntropyBackward<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let rows = expected.count
        let (labels, predictions) = (expected.reshaped(to: [rows]), actual.reshaped(to: [rows, actual.count / rows]))
        // -outputGradient / rows / selected, scattered to the labels. The rows with the ignored label select 0,
        // and the scatter drops their quotients.
        math.write(actualGradient) { da in
            let selected = math.temporary([rows])
            let factor = math.temporary([])
            Device.Engine.gather(expanded: predictions, context: labels, result: selected, axis: 1, ignoreIndex: ignoreIndex)
            math.multiply(outputGradient, -1 / N(rows), into: factor)
            math.divide(factor, selected, into: selected)
            Device.Engine.scatter(reduced: ShapedBuffer(selected), context: labels, result: da.reshaped(to: predictions.shape), axis: 1, ignoreIndex: ignoreIndex)
        }
    }

    static func categoricalNegativeLogLikelihood<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, ignoreIndex: Int32, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let rows = expected.count
        let selected = math.temporary([rows])
        Device.Engine.gather(expanded: actual.reshaped(to: [rows, actual.count / rows]), context: expected.reshaped(to: [rows]), result: selected, axis: 1, ignoreIndex: ignoreIndex)
        math.mean(selected, along: [0], into: result)
        math.negate(result, into: result)
    }

    static func categoricalNegativeLogLikelihoodBackward<N: NumericType>(expected: ShapedBuffer<Int32, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, ignoreIndex: Int32, actualGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let rows = expected.count
        // -outputGradient / rows at the label of every row, and 0 elsewhere
        math.write(actualGradient) { da in
            let rowGradient = math.temporary([rows])
            math.multiply(math.constant(-1 / N(rows), shape: [rows]), outputGradient, into: rowGradient)
            Device.Engine.scatter(reduced: ShapedBuffer(rowGradient), context: expected.reshaped(to: [rows]), result: da.reshaped(to: [rows, actual.count / rows]), axis: 1, ignoreIndex: ignoreIndex)
        }
    }

    static func meanSquaredError<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let differences = math.temporary(shapeForBroadcastedOperands(expected.shape, actual.shape))
        math.subtract(expected, actual, into: differences)
        math.multiply(differences, differences, into: differences)
        math.sum(differences, along: Array(differences.shape.indices), into: result)
        math.multiply(result, 1 / N(meanSquaredErrorDivisor(expectedShape: expected.shape)), into: result)
    }

    static func meanSquaredErrorBackward<N: NumericType>(expected: ShapedBuffer<N, Device>, actual: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, expectedGradient: GradientBuffer<N, Device>?, actualGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // 2 * (expected - actual) * outputGradient / divisor for the expected values, and its negative for the predictions
        let differences = math.temporary(shapeForBroadcastedOperands(expected.shape, actual.shape))
        let factor = math.temporary([])
        math.multiply(outputGradient, 2 / N(meanSquaredErrorDivisor(expectedShape: expected.shape)), into: factor)
        math.subtract(expected, actual, into: differences)
        math.multiply(differences, factor, into: differences)
        math.writeSum(of: differences, into: expectedGradient)
        if actualGradient != nil {
            math.negate(differences, into: differences)
            math.writeSum(of: differences, into: actualGradient)
        }
    }

    static func l1Loss<N: NumericType>(input: ShapedBuffer<N, Device>, scale: N, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // mean(relu(input) + relu(-input)) * scale
        let magnitudes = math.temporary(input.shape)
        let negativePart = math.temporary(input.shape)
        math.relu(input, into: magnitudes)
        math.negate(input, into: negativePart)
        math.relu(negativePart, into: negativePart)
        math.add(magnitudes, negativePart, into: magnitudes)
        math.mean(magnitudes, along: Array(input.shape.indices), into: result)
        math.multiply(result, scale, into: result)
    }

    static func l1LossBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, scale: N, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // (heaviside(input) - heaviside(-input)) * outputGradient * scale / count. The gradient at 0 is 0.
        math.write(inputGradient) { dx in
            let negative = math.temporary(input.shape)
            let factor = math.temporary([])
            math.negate(input, into: negative)
            math.heaviside(negative, into: negative)
            math.heaviside(input, into: dx)
            math.subtract(dx, negative, into: dx)
            math.multiply(outputGradient, scale / N(input.count), into: factor)
            math.multiply(dx, factor, into: dx)
        }
    }

    static func l2Loss<N: NumericType>(input: ShapedBuffer<N, Device>, scale: N, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let squares = math.temporary(input.shape)
        math.multiply(input, input, into: squares)
        math.mean(squares, along: Array(input.shape.indices), into: result)
        math.multiply(result, scale, into: result)
    }

    static func l2LossBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, scale: N, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // input * outputGradient * 2 * scale / count
        math.write(inputGradient) { dx in
            let factor = math.temporary([])
            math.multiply(outputGradient, N(2) * scale / N(input.count), into: factor)
            math.multiply(input, factor, into: dx)
        }
    }
}

extension FusedOperationsType {
    /// Divisor of the sum of squared differences: the number of rows of the expected values, or 1 for a vector or a scalar.
    static func meanSquaredErrorDivisor(expectedShape: [Int]) -> Int {
        expectedShape.count > 1 ? expectedShape[0] : 1
    }
}

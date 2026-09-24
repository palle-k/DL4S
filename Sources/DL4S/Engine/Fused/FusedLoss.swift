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
    static func binaryCrossEntropy<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>) -> Tensor<N, Device> {
        let e = expected.detached().view(as: [-1])
        let a = actual.detached().view(as: [-1])
        return (-(e * a.log() + (1 - e) * (1 - a).log())).reduceMean()
    }

    static func binaryCrossEntropyBackward<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?)) {
        let computed = Composed.binaryCrossEntropyGradients(
            expected: expected.detached(),
            actual: actual.detached(),
            outputGradient: outputGradient.detached(),
            computesExpected: expected.requiresGradient,
            computesActual: actual.requiresGradient,
        )
        Tensor.accumulate(computed.expected, into: &gradients.expected)
        Tensor.accumulate(computed.actual, into: &gradients.actual)
    }

    static func categoricalCrossEntropy<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device> {
        -Composed.selectedProbabilities(expected: expected, actual: actual.detached(), ignoreIndex: ignoreIndex).log().reduceMean()
    }

    static func categoricalCrossEntropyBackward<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32, accumulating gradient: inout Tensor<N, Device>?) {
        Tensor.accumulate(
            Composed.categoricalCrossEntropyGradient(expected: expected, actual: actual.detached(), outputGradient: outputGradient.detached(), ignoreIndex: ignoreIndex),
            into: &gradient,
        )
    }

    static func categoricalNegativeLogLikelihood<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device> {
        let expected = expected.flattened()
        return -actual.detached()
            .view(as: [expected.count, -1])
            .gather(using: expected, alongAxis: 1, ignoreIndex: ignoreIndex)
            .reduceMean()
    }

    static func categoricalNegativeLogLikelihoodBackward<N: NumericType>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32, accumulating gradient: inout Tensor<N, Device>?) {
        Tensor.accumulate(
            Composed.categoricalNegativeLogLikelihoodGradient(expected: expected, actualShape: actual.shape, outputGradient: outputGradient.detached(), ignoreIndex: ignoreIndex),
            into: &gradient,
        )
    }

    static func meanSquaredError<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>) -> Tensor<N, Device> {
        let difference = expected.detached() - actual.detached()
        return (difference * difference).reduceSum() / Tensor(Composed.meanSquaredErrorDivisor(expected: expected))
    }

    static func meanSquaredErrorBackward<N: NumericType>(expected: Tensor<N, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, accumulating gradients: inout (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?)) {
        let computed = Composed.meanSquaredErrorGradients(
            expected: expected.detached(),
            actual: actual.detached(),
            outputGradient: outputGradient.detached(),
            computesExpected: expected.requiresGradient,
            computesActual: actual.requiresGradient,
        )
        Tensor.accumulate(computed.expected, into: &gradients.expected)
        Tensor.accumulate(computed.actual, into: &gradients.actual)
    }

    static func l1Loss<N: NumericType>(input: Tensor<N, Device>, scale: N) -> Tensor<N, Device> {
        let input = input.detached()
        return (input.rectifiedLinear() + (-input).rectifiedLinear()).reduceMean() * Tensor(scale)
    }

    static func l1LossBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, accumulating gradient: inout Tensor<N, Device>?) {
        Tensor.accumulate(
            Composed.l1LossGradient(input: input.detached(), outputGradient: outputGradient.detached(), scale: scale),
            into: &gradient,
        )
    }

    static func l2Loss<N: NumericType>(input: Tensor<N, Device>, scale: N) -> Tensor<N, Device> {
        let input = input.detached()
        return (input * input).reduceMean() * Tensor(scale)
    }

    static func l2LossBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, accumulating gradient: inout Tensor<N, Device>?) {
        Tensor.accumulate(
            Composed.l2LossGradient(input: input.detached(), outputGradient: outputGradient.detached(), scale: scale),
            into: &gradient,
        )
    }
}

// MARK: Composed gradients

extension Composed {
    static func binaryCrossEntropyGradients<N, Device>(
        expected: Tensor<N, Device>,
        actual: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesExpected: Bool,
        computesActual: Bool,
    ) -> (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?) {
        let factor = outputGradient / Tensor(N(actual.count))
        let expectedGradient = computesExpected ? factor * ((1 - actual).log() - actual.log()) : nil
        let actualGradient = computesActual ? factor * (actual - expected) / (actual * (1 - actual)) : nil
        return (expectedGradient, actualGradient)
    }

    /// Returns the probability of the expected label of every row, shape [count], and 1 for rows with the ignored label.
    ///
    /// Rows with the ignored label then add log(1) = 0 to the loss.
    static func selectedProbabilities<N, Device>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device> {
        let expected = expected.flattened()
        let selected = actual
            .view(as: [expected.count, -1])
            .gather(using: expected, alongAxis: 1, ignoreIndex: ignoreIndex)
        let ignoredRows = Tensor<N, Device>(expected.elements.map { $0 == ignoreIndex ? N.one : N.zero })
        return selected + ignoredRows
    }

    static func categoricalCrossEntropyGradient<N, Device>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device> {
        let selected = selectedProbabilities(expected: expected, actual: actual, ignoreIndex: ignoreIndex)
        let classCount = actual.shape[actual.dim - 1]
        return (-outputGradient / Tensor(N(selected.count)) / selected)
            .scatter(using: expected.flattened(), alongAxis: 1, withSize: classCount, ignoreIndex: ignoreIndex)
            .view(as: actual.shape)
    }

    static func categoricalNegativeLogLikelihoodGradient<N, Device>(expected: Tensor<Int32, Device>, actualShape: [Int], outputGradient: Tensor<N, Device>, ignoreIndex: Int32) -> Tensor<N, Device> {
        let rowCount = expected.count
        let rowGradient = Tensor<N, Device>(repeating: N(-1) / N(rowCount), shape: [rowCount]) * outputGradient
        return rowGradient
            .scatter(using: expected.flattened(), alongAxis: 1, withSize: actualShape[actualShape.count - 1], ignoreIndex: ignoreIndex)
            .view(as: actualShape)
    }

    /// Divisor of the sum of squared differences: the number of rows of `expected`, or 1 for a vector or a scalar.
    static func meanSquaredErrorDivisor<N, Device>(expected: Tensor<N, Device>) -> N {
        N(expected.dim > 1 ? expected.shape[0] : 1)
    }

    static func meanSquaredErrorGradients<N, Device>(
        expected: Tensor<N, Device>,
        actual: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        computesExpected: Bool,
        computesActual: Bool,
    ) -> (expected: Tensor<N, Device>?, actual: Tensor<N, Device>?) {
        let expectedGradient = 2 * (expected - actual) * outputGradient / Tensor(meanSquaredErrorDivisor(expected: expected))
        return (
            computesExpected ? expectedGradient.reducingBroadcast(to: expected.shape) : nil,
            computesActual ? (-expectedGradient).reducingBroadcast(to: actual.shape) : nil,
        )
    }

    static func l1LossGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N) -> Tensor<N, Device> {
        let factor = outputGradient * Tensor(scale / N(input.count))
        return (input.heaviside() - (-input).heaviside()) * factor
    }

    static func l2LossGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N) -> Tensor<N, Device> {
        let factor = outputGradient * Tensor(N(2) * scale / N(input.count))
        return input * factor
    }
}

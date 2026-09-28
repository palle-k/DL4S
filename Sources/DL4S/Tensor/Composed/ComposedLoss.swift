//
//  ComposedLoss.swift
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
    static func binaryCrossEntropyBackward<N, Device>(
        expected: Tensor<N, Device>,
        actual: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        expectedGradient: inout GradientAccumulator<N, Device>,
        actualGradient: inout GradientAccumulator<N, Device>,
    ) {
        // The loss compares the elements in memory order, so the operands can have different shapes with the same count.
        let (e, a) = (expected.view(as: [-1]), actual.view(as: [-1]))
        let factor = outputGradient / Tensor(N(actual.count))
        if expectedGradient.isRequested {
            expectedGradient.add((factor * ((1 - a).log() - a.log())).view(as: expected.shape))
        }
        if actualGradient.isRequested {
            actualGradient.add((factor * (a - e) / (a * (1 - a))).view(as: actual.shape))
        }
    }

    static func categoricalCrossEntropyBackward<N, Device>(expected: Tensor<Int32, Device>, actual: Tensor<N, Device>, outputGradient: Tensor<N, Device>, ignoreIndex: Int32, actualGradient: inout GradientAccumulator<N, Device>) {
        guard actualGradient.isRequested else {
            return
        }
        let expected = expected.flattened()
        let classCount = actual.shape[actual.dim - 1]
        let selected = actual
            .view(as: [expected.count, -1])
            .gather(using: expected, alongAxis: 1, ignoreIndex: ignoreIndex)
        // The rows with the ignored label select 0. The scatter drops their quotients.
        actualGradient.add(
            (-outputGradient / Tensor(N(expected.count)) / selected)
                .scatter(using: expected, alongAxis: 1, withSize: classCount, ignoreIndex: ignoreIndex)
                .view(as: actual.shape),
        )
    }

    static func categoricalNegativeLogLikelihoodBackward<N, Device>(expected: Tensor<Int32, Device>, actualShape: [Int], outputGradient: Tensor<N, Device>, ignoreIndex: Int32, actualGradient: inout GradientAccumulator<N, Device>) {
        guard actualGradient.isRequested else {
            return
        }
        let rowCount = expected.count
        let rowGradient = Tensor<N, Device>(repeating: N(-1) / N(rowCount), shape: [rowCount]) * outputGradient
        actualGradient.add(
            rowGradient
                .scatter(using: expected.flattened(), alongAxis: 1, withSize: actualShape[actualShape.count - 1], ignoreIndex: ignoreIndex)
                .view(as: actualShape),
        )
    }

    static func meanSquaredErrorBackward<N, Device>(
        expected: Tensor<N, Device>,
        actual: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        expectedGradient: inout GradientAccumulator<N, Device>,
        actualGradient: inout GradientAccumulator<N, Device>,
    ) {
        let differenceGradient = 2 * (expected - actual) * outputGradient / Tensor(N(Device.FusedOperations.meanSquaredErrorDivisor(expectedShape: expected.shape)))
        if expectedGradient.isRequested {
            expectedGradient.add(differenceGradient.reducingBroadcast(to: expected.shape))
        }
        if actualGradient.isRequested {
            actualGradient.add((-differenceGradient).reducingBroadcast(to: actual.shape))
        }
    }

    /// The gradient at 0 is 0.
    static func l1LossBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let factor = outputGradient * Tensor(scale / N(input.count))
        inputGradient.add((input.heaviside() - (-input).heaviside()) * factor)
    }

    static func l2LossBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, scale: N, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        inputGradient.add(input * (outputGradient * Tensor(N(2) * scale / N(input.count))))
    }
}

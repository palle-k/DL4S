//
//  Loss.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//  Copyright (c) 2019 - Palle Klewitz
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

// MARK: Losses

/// Computes the (element-wise) binary cross entropy loss on the given and expected probabilities and
/// uses the mean as a reduction.
/// expected and predicted are assumed to be in the interval (0, 1).
///
/// The binary cross entropy loss is defined as
///
///     -expected * log(predicted) - (1 - expected) * log(1 - predicted)
///
/// - Parameters:
///   - expected: Expected values
///   - actual: Predicted values
/// - Returns: Loss, scalar value
public func binaryCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    precondition(expected.count == actual.count, "Expected and predicted values must have the same number of elements.")
    let result = Device.FusedOperations.binaryCrossEntropy(expected: expected, actual: actual)
    return result.attachingContext(tag: "binaryCrossEntropy", sources: [expected, actual]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            let computed = Composed.binaryCrossEntropyGradients(
                expected: expected,
                actual: actual,
                outputGradient: resultGradient,
                computesExpected: expected.requiresGradient,
                computesActual: actual.requiresGradient,
            )
            Tensor.accumulate(computed.expected, into: &gradients[0])
            Tensor.accumulate(computed.actual, into: &gradients[1])
        } else {
            var accumulated = (expected: gradients[0].take(), actual: gradients[1].take())
            Device.FusedOperations.binaryCrossEntropyBackward(expected: expected, actual: actual, outputGradient: resultGradient, accumulating: &accumulated)
            gradients[0] = accumulated.expected
            gradients[1] = accumulated.actual
        }
    }
}

/// Computes the categorical cross entropy loss on the given expected probabilities and the expected labels and
/// uses the mean as a reduction.
/// predicted values are assumed to be in the interval (0, 1)
///
/// The categorical cross entropy loss is defined as
///
///     -log(predicted[expected])
///
/// - Parameters:
///   - expected: Expected labels
///   - actual: Predicted values
///   - ignoreIndex: Value in expected, which is ignored. Ignored labels add 0 to the loss, but count for the mean.
/// - Returns: Loss, scalar value
public func categoricalCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Int32, Device>, actual: Tensor<Element, Device>, ignoreIndex: Int32 = -1) -> Tensor<Element, Device> {
    precondition(expected.dim + 1 == actual.dim, "Dimensionality of actual sequence must be one larger than expected dimensionality.")
    precondition(expected.shape == actual.shape.dropLast(), "Shape of expected sequence must be equal to shape of actual sequence minus last axis")

    let result = Device.FusedOperations.categoricalCrossEntropy(expected: expected, actual: actual, ignoreIndex: ignoreIndex)
    return result.attachingContext(tag: "categoricalCrossEntropy", sources: [actual]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            Tensor.accumulate(Composed.categoricalCrossEntropyGradient(expected: expected, actual: actual, outputGradient: resultGradient, ignoreIndex: ignoreIndex), into: &gradients[0])
        } else {
            Device.FusedOperations.categoricalCrossEntropyBackward(expected: expected, actual: actual, outputGradient: resultGradient, ignoreIndex: ignoreIndex, accumulating: &gradients[0])
        }
    }
}

/// Computes the categorical negative log likelihood (NLL) loss on the given expected probabilities and the expected labels and
/// uses the mean as a reduction.
/// Predicted values are assumed to be in the interval (-infinity, 0).
///
/// NLL loss should be used in conjunction with logSoftmax.
///
/// The categorical NLL  loss is defined as
///
///     -predicted[expected]
///
/// - Parameters:
///   - expected: Expected labels
///   - actual: Predicted values
///   - ignoreIndex: Value in expected, which is ignored. Ignored labels add 0 to the loss, but count for the mean.
/// - Returns: Loss, scalar value
public func categoricalNegativeLogLikelihood<Element: NumericType, Device: DeviceType>(expected: Tensor<Int32, Device>, actual: Tensor<Element, Device>, ignoreIndex: Int32 = -1) -> Tensor<Element, Device> {
    precondition(expected.dim + 1 == actual.dim, "Dimensionality of actual sequence must be one larger than expected dimensionality.")
    precondition(expected.shape == actual.shape.dropLast(), "Shape of expected sequence must be equal to shape of actual sequence minus last axis")

    let result = Device.FusedOperations.categoricalNegativeLogLikelihood(expected: expected, actual: actual, ignoreIndex: ignoreIndex)
    let actualShape = actual.shape
    return result.attachingContext(tag: "negativeLogLikelihood", sources: [actual]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            Tensor.accumulate(Composed.categoricalNegativeLogLikelihoodGradient(expected: expected, actualShape: actualShape, outputGradient: resultGradient, ignoreIndex: ignoreIndex), into: &gradients[0])
        } else {
            Device.FusedOperations.categoricalNegativeLogLikelihoodBackward(expected: expected, actual: actual, outputGradient: resultGradient, ignoreIndex: ignoreIndex, accumulating: &gradients[0])
        }
    }
}

/// Computes the sum of squared differences between the given predicted and expected values, divided by the number of rows.
///
/// The divisor is `expected.shape[0]` when `expected` has two or more axes, and 1 otherwise.
///
/// - Parameters:
///   - expected: Expected values
///   - actual: Predicted values, broadcastable with the expected values
public func meanSquaredError<Element, Device>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let result = Device.FusedOperations.meanSquaredError(expected: expected, actual: actual)
    return result.attachingContext(tag: "meanSquaredError", sources: [expected, actual]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            let computed = Composed.meanSquaredErrorGradients(
                expected: expected,
                actual: actual,
                outputGradient: resultGradient,
                computesExpected: expected.requiresGradient,
                computesActual: actual.requiresGradient,
            )
            Tensor.accumulate(computed.expected, into: &gradients[0])
            Tensor.accumulate(computed.actual, into: &gradients[1])
        } else {
            var accumulated = (expected: gradients[0].take(), actual: gradients[1].take())
            Device.FusedOperations.meanSquaredErrorBackward(expected: expected, actual: actual, outputGradient: resultGradient, accumulating: &accumulated)
            gradients[0] = accumulated.expected
            gradients[1] = accumulated.actual
        }
    }
}

/// Computes the L2 loss of the given tensor, `mean(vector * vector) * loss`.
/// - Parameters:
///   - vector: Tensor to apply weight decay on
///   - loss: Weight decay importance scaling factor
public func l2loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    let result = Device.FusedOperations.l2Loss(input: vector, scale: loss)
    return result.attachingContext(tag: "l2loss", sources: [vector]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            Tensor.accumulate(Composed.l2LossGradient(input: vector, outputGradient: resultGradient, scale: loss), into: &gradients[0])
        } else {
            Device.FusedOperations.l2LossBackward(input: vector, outputGradient: resultGradient, scale: loss, accumulating: &gradients[0])
        }
    }
}

/// Computes the L1 loss of the given tensor, `mean(abs(vector)) * loss`.
/// - Parameters:
///   - vector: Tensor to apply weight decay on
///   - loss: Weight decay importance scaling factor
public func l1loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    let result = Device.FusedOperations.l1Loss(input: vector, scale: loss)
    return result.attachingContext(tag: "l1loss", sources: [vector]) { resultGradient, gradients in
        if resultGradient.requiresGradient {
            Tensor.accumulate(Composed.l1LossGradient(input: vector, outputGradient: resultGradient, scale: loss), into: &gradients[0])
        } else {
            Device.FusedOperations.l1LossBackward(input: vector, outputGradient: resultGradient, scale: loss, accumulating: &gradients[0])
        }
    }
}

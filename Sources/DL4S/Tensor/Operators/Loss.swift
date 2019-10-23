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

/// Computes the (element-wise) binary cross entropy loss on the given and expected probabilities.
/// expected and predicted are assumed to be in the interval (0, 1)
///
/// The binary cross entropy loss is defined as
///
///     -expected * log(predicted) - (1 - expected) * log(1 - predicted)
///
/// - Parameters:
///   - expected: Expected values
///   - actual: Predicted values
public func binaryCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    OperationGroup.capture(named: "BinaryCrossEntropy") {
        let e = expected.view(as: [-1])
        let a = actual.view(as: [-1])
        
        let p1 = e * a.log()
        let p2 = (1 - e) * (1 - a).log()
        return (-(p1 + p2)).reduceMean()
    }
}

/// Computes the categorical cross entropy loss on the given expected probabilities and the expected labels
/// predicted values are assumed to be in the interval (0, 1)
///
/// The categorical cross entropy loss is defined as
///
///     -log(predicted[expected])
///
/// - Parameters:
///   - expected: Expected labels
///   - actual: Predicted values
public func categoricalCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Int32, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    OperationGroup.capture(named: "CategoricalCrossEntropy") {
        precondition(expected.dim + 1 == actual.dim, "Dimensionality of actual sequence must be one larger than expected dimensionality.")
        
        var result = Tensor<Element, Device>(0)
        
        for index in iterate(expected.shape) {
            result -= log(actual[index + [Int(expected[index].item)]])
        }
        
        return result / Tensor<Element, Device>(Element(expected.count))
    }
}


/// Computes the element-wise mean squared error between the given predicted and expected values
///
/// - Parameters:
///   - expected: Expected values
///   - actual: Predicted values
public func meanSquaredError<Element, Device>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    OperationGroup.capture(named: "MeanSquaredError") {
        let diff = expected - actual
        let s = sum(diff * diff)
        return s  / Tensor(Element(expected.dim > 1 ? expected.shape[0] : 1))
    }
}

/// Computes the l2 weight decay loss of the given tensor
/// - Parameters:
///   - vector: Tensor to apply weight decay on
///   - loss: Weight decay importance scaling factor
public func l2loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    return mean(vector * vector) * Tensor(loss)
}

/// Computes the l1 weight decay loss of the given tensor
/// - Parameters:
///   - vector: Tensor to apply weight decay on
///   - loss: Weight decay importance scaling factor
public func l1loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    // max(x, -x)
    return leakyRelu(vector, leakage: -1) * Tensor(loss)
}

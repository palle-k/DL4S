//
//  Loss.swift
//  DL4S
//
//  |  ||
//  || |_
//
//  Created by Palle Klewitz on 14.03.19.
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


public func binaryCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let e = expected.view(as: -1)
    let a = actual.view(as: -1)
    
    let p1 = e * log(a)
    let p2 = (1 - e) * log(1 - a)
    return mean(-(p1 + p2))
}

public func categoricalCrossEntropy<Element: NumericType, Device: DeviceType>(expected: Tensor<Int32, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let expectedView = expected.view(as: -1)
    let actualView = actual.view(as: expectedView.shape[0], -1)
    
    var result = Tensor<Element, Device>(0)
    
    for i in 0 ..< expectedView.shape[0] {
        // expected is not included in compute graph
        result = result - log(actualView[i, Int(expectedView[i].item)])
    }
    
    return result / Tensor<Element, Device>(Element(expected.count))
}

public func meanSquaredError<Element, Device>(expected: Tensor<Element, Device>, actual: Tensor<Element, Device>) -> Tensor<Element, Device> {
    let diff = expected - actual
    let s = sum(diff * diff)
    return s  / Tensor(Element(expected.dim > 1 ? expected.shape[0] : 1))
}

public func l2loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    return mean(vector * vector) * Tensor(loss)
}

public func l1loss<Element, Device>(_ vector: Tensor<Element, Device>, loss: Element) -> Tensor<Element, Device> {
    // max(x, -x)
    return leakyRelu(vector, leakage: -1) * Tensor(loss)
}

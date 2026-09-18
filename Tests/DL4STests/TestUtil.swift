//
//  TestUtil.swift
//  DL4STests
//
//  Created by Palle Klewitz on 31.08.26.
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

import DL4S
import Foundation
import Testing

extension Trait where Self == ConditionTrait {
    /// Runs the test only when the `DL4S_LONG_TESTS` environment variable is set.
    ///
    /// Tests that train real models take minutes. CI does not run them. Set `DL4S_LONG_TESTS=1` to run them locally.
    static var longRunning: Self {
        .enabled(if: ProcessInfo.processInfo.environment["DL4S_LONG_TESTS"] != nil, "Set DL4S_LONG_TESTS=1 to run long tests.")
    }

    /// Runs the test only when a short training run completes in reasonable time.
    ///
    /// That is the case with an accelerated backend (Accelerate or MKL), in a release build, or when `DL4S_LONG_TESTS` is set.
    /// With the generic fallback in a debug build, the MNIST sample tests need more than an hour.
    static var trainsModel: Self {
        #if canImport(Accelerate) || MKL_ENABLE
        let isAccelerated = true
        #else
        let isAccelerated = false
        #endif
        #if DEBUG
        let isDebugBuild = true
        #else
        let isDebugBuild = false
        #endif
        let longTestsEnabled = ProcessInfo.processInfo.environment["DL4S_LONG_TESTS"] != nil
        return .enabled(if: isAccelerated || !isDebugBuild || longTestsEnabled, "Training runs with the generic fallback in a debug build are too slow. Build in release, or set DL4S_LONG_TESTS=1.")
    }
}

/// Records an issue when `actual` differs from `expected` by more than `accuracy`.
func expectEqual<Value: FloatingPoint>(_ actual: Value, _ expected: Value, accuracy: Value, sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(abs(actual - expected) <= accuracy, "\(actual) is not within \(accuracy) of \(expected)", sourceLocation: sourceLocation)
}

/// Records an issue when the shapes differ or when the sum of squared differences is larger than `tolerance`.
func expectClose<Element: NumericType & BinaryFloatingPoint>(_ actual: Tensor<Element, CPU>, _ expected: Tensor<Element, CPU>, tolerance: Element = 1e-6, sourceLocation: SourceLocation = #_sourceLocation) {
    guard actual.shape == expected.shape else {
        Issue.record("shape \(actual.shape) differs from \(expected.shape)", sourceLocation: sourceLocation)
        return
    }
    let difference = actual.detached() - expected.detached()
    let distance = (difference * difference).reduceSum().item
    #expect(distance <= tolerance, "squared distance \(distance) is larger than \(tolerance): \(actual) vs \(expected)", sourceLocation: sourceLocation)
}

/// Central difference estimate of the gradient of the sum of `function` at `point`.
func numericalGradient(of function: (Tensor<Double, CPU>) -> Tensor<Double, CPU>, at point: Tensor<Double, CPU>, step: Double = 1e-5) -> Tensor<Double, CPU> {
    let elements = point.detached().elements
    var gradient = [Double](repeating: 0, count: elements.count)
    for index in elements.indices {
        var forward = elements
        forward[index] += step
        var backward = elements
        backward[index] -= step
        let forwardValue = function(Tensor(forward, shape: point.shape)).reduceSum().item
        let backwardValue = function(Tensor(backward, shape: point.shape)).reduceSum().item
        gradient[index] = (forwardValue - backwardValue) / (2 * step)
    }
    return Tensor(gradient, shape: point.shape)
}

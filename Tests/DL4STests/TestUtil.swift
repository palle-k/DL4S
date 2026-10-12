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

@testable import DL4S
import Foundation
import Testing

extension Trait where Self == ConditionTrait {
    /// Runs the test only when the `DL4S_LONG_TESTS` environment variable is set.
    ///
    /// Tests that train real models take minutes. CI does not run them. Set `DL4S_LONG_TESTS=1` to run them locally.
    static var longRunning: Self {
        .enabled(if: longTestsEnabled, "Set DL4S_LONG_TESTS=1 to run long tests.")
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
        return .enabled(if: isAccelerated || !isDebugBuild || longTestsEnabled, "Training runs with the generic fallback in a debug build are too slow. Build in release, or set DL4S_LONG_TESTS=1.")
    }

    /// Runs the test only in a release build or when `DL4S_LONG_TESTS` is set.
    ///
    /// The CPU references of large operations need minutes in a debug build.
    static var releaseBuild: Self {
        .enabled(if: !isDebugBuild || longTestsEnabled, "The CPU reference is too slow in a debug build. Build in release, or set DL4S_LONG_TESTS=1.")
    }

    private static var longTestsEnabled: Bool {
        ProcessInfo.processInfo.environment["DL4S_LONG_TESTS"] != nil
    }

    private static var isDebugBuild: Bool {
        #if DEBUG
        true
        #else
        false
        #endif
    }
}

extension Tensor where Element: RandomizableType {
    /// Returns a tensor with uniformly distributed elements from a `WyHash` generator with the given seed.
    ///
    /// The same arguments always give the same elements.
    static func uniform(_ shape: [Int], min: Element = -.one, max: Element = .one, seed: UInt64, requiresGradient: Bool = false) -> Self {
        var generator = WyHash(seed: seed)
        return Self(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient, using: &generator)
    }
}

// MARK: Closeness checks

// The checks have different tolerance semantics, so a test keeps the check that its tolerance was chosen for.

/// Records an issue when `actual` differs from `expected` by more than `accuracy`.
func expectEqual<Value: FloatingPoint>(_ actual: Value, _ expected: Value, accuracy: Value, sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(abs(actual - expected) <= accuracy, "\(actual) is not within \(accuracy) of \(expected)", sourceLocation: sourceLocation)
}

/// Records an issue when the shapes differ or when the sum of squared differences is larger than `tolerance`.
///
/// The tolerance is absolute and applies to the whole tensor.
func expectClose<Element: NumericType & BinaryFloatingPoint>(_ actual: Tensor<Element, CPU>, _ expected: Tensor<Element, CPU>, tolerance: Element = 1e-6, sourceLocation: SourceLocation = #_sourceLocation) {
    guard actual.shape == expected.shape else {
        Issue.record("shape \(actual.shape) differs from \(expected.shape)", sourceLocation: sourceLocation)
        return
    }
    let difference = actual.detached() - expected.detached()
    let distance = (difference * difference).reduceSum().item
    #expect(distance <= tolerance, "squared distance \(distance) is larger than \(tolerance): \(actual) vs \(expected)", sourceLocation: sourceLocation)
}

/// Records an issue when the shapes differ or when the largest difference of two elements is larger than
/// `tolerance` times the largest magnitude of `expected`, plus `tolerance`.
///
/// The tolerance is relative to the largest element of `expected`, so small elements of a tensor with large elements get a large tolerance.
func expectApproximatelyEqual<Element: NumericType & BinaryFloatingPoint>(_ actual: Tensor<Element, CPU>, _ expected: Tensor<Element, CPU>, tolerance: Element = 1e-6, _ comment: String, sourceLocation: SourceLocation = #_sourceLocation) {
    guard actual.shape == expected.shape else {
        Issue.record("\(comment): shape \(actual.shape) differs from \(expected.shape)", sourceLocation: sourceLocation)
        return
    }
    let difference = zip(actual.elements, expected.elements).map { abs($0 - $1) }.max() ?? 0
    let magnitude = expected.elements.map(abs).max() ?? 0
    #expect(difference <= tolerance * (1 + magnitude), "\(comment): difference \(difference): \(actual) vs \(expected)", sourceLocation: sourceLocation)
}

/// Records an issue when the shapes differ or when an element differs from the expected element by more than `tolerance`
/// times the magnitude of the expected element, or by more than `tolerance` for an expected element with a magnitude below 1.
///
/// The tolerance applies to every element separately.
func expectElementwiseClose<Element: NumericType & BinaryFloatingPoint>(_ actual: Tensor<Element, CPU>, _ expected: Tensor<Element, CPU>, tolerance: Element, _ comment: String, sourceLocation: SourceLocation = #_sourceLocation) {
    guard actual.shape == expected.shape else {
        Issue.record("\(comment): shape \(actual.shape) differs from \(expected.shape)", sourceLocation: sourceLocation)
        return
    }
    let difference = zip(actual.elements, expected.elements).map { abs($0 - $1) / Swift.max(1, abs($1)) }.max() ?? 0
    #expect(difference <= tolerance, "\(comment): relative difference \(difference)", sourceLocation: sourceLocation)
}

// MARK: Gradient checks

/// Returns the gradients of the sum of `function` with respect to the sources at the positions `differentiable`,
/// estimated with central differences.
///
/// - Parameters:
///   - function: Function of the sources. The sources that it gets do not require a gradient.
///   - sources: Point at which the gradients are estimated.
///   - differentiable: Positions of the sources whose gradients are returned. All sources when nil.
///   - step: Distance from the point to the two evaluations of `function`.
/// - Returns: One gradient for each position in `differentiable`.
func numericalGradients(of function: ([Tensor<Double, CPU>]) -> Tensor<Double, CPU>, at sources: [Tensor<Double, CPU>], differentiable: [Int]? = nil, step: Double = 1e-5) -> [Tensor<Double, CPU>] {
    let plainSources = sources.map { $0.detached() }
    return (differentiable ?? Array(sources.indices)).map { position in
        let elements = plainSources[position].elements
        let gradient = elements.indices.map { index in
            func value(at offset: Double) -> Double {
                var shifted = elements
                shifted[index] += offset
                var arguments = plainSources
                arguments[position] = Tensor(shifted, shape: plainSources[position].shape)
                return function(arguments).reduceSum().item
            }
            return (value(at: step) - value(at: -step)) / (2 * step)
        }
        return Tensor(gradient, shape: plainSources[position].shape)
    }
}

/// Returns the gradient of the sum of `function` at `point`, estimated with central differences.
func numericalGradient(of function: (Tensor<Double, CPU>) -> Tensor<Double, CPU>, at point: Tensor<Double, CPU>, step: Double = 1e-5) -> Tensor<Double, CPU> {
    numericalGradients(of: { function($0[0]) }, at: [point], step: step)[0]
}

/// Records an issue when a gradient of the sum of `function` at `points` differs from its central difference estimate,
/// with and without a recorded backward graph.
///
/// - Parameters:
///   - function: Function whose gradients are checked.
///   - points: Sources of `function`. Every source must require a gradient.
///   - tolerance: Largest sum of squared differences of a gradient, see ``expectClose(_:_:tolerance:sourceLocation:)``.
func expectGradientsMatchCentralDifferences(
    of function: ([Tensor<Double, CPU>]) -> Tensor<Double, CPU>,
    at points: [Tensor<Double, CPU>],
    tolerance: Double = 1e-14,
    sourceLocation: SourceLocation = #_sourceLocation,
) {
    let expected = numericalGradients(of: function, at: points)
    for retainsGraph in [false, true] {
        let gradients = function(points).reduceSum().gradients(of: points, retainBackwardsGraph: retainsGraph)
        for (gradient, expectedGradient) in zip(gradients, expected) {
            expectClose(gradient, expectedGradient, tolerance: tolerance, sourceLocation: sourceLocation)
        }
    }
}

/// Records an issue when the gradient of the sum of `function` at `point` differs from its central difference estimate,
/// with and without a recorded backward graph.
func expectGradientsMatchCentralDifferences(
    of function: (Tensor<Double, CPU>) -> Tensor<Double, CPU>,
    at point: Tensor<Double, CPU>,
    tolerance: Double = 1e-14,
    sourceLocation: SourceLocation = #_sourceLocation,
) {
    expectGradientsMatchCentralDifferences(of: { function($0[0]) }, at: [point], tolerance: tolerance, sourceLocation: sourceLocation)
}

// MARK: Operations

/// Returns the result and the mask of the dropout requirement of the device of `input`.
func fusedDropout<Element: FloatingPointNumericType, Device: DeviceType>(_ input: Tensor<Element, Device>, rate: Float) -> (output: Tensor<Element, Device>, mask: Tensor<Element, Device>) {
    var (output, mask) = (Tensor<Element, Device>(uninitializedShape: input.shape), Tensor<Element, Device>(uninitializedShape: input.shape))
    withExtendedLifetime(input) {
        Device.FusedOperations.dropout(input: input.values, rate: rate, result: output.mutableValues, mask: mask.mutableValues)
    }
    return (output, mask)
}

/// Runs `body` on `threadCount` new threads at the same time and returns when all threads are done.
///
/// - Parameters:
///   - threadCount: Number of threads.
///   - name: Prefix of the thread names. Each thread gets the name `<name>.<index>`.
///   - body: Work of a thread. It gets the index of the thread.
func runOnThreads(_ threadCount: Int, name: String, _ body: @escaping @Sendable (_ threadIndex: Int) -> Void) {
    let group = DispatchGroup()
    for index in 0 ..< threadCount {
        group.enter()
        // Raw threads make sure that the bodies run in parallel, which structured concurrency and dispatch queues do not.
        let thread = Thread {
            body(index)
            group.leave()
        }
        thread.name = "\(name).\(index)"
        thread.start()
    }
    group.wait()
}

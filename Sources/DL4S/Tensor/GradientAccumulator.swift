//
//  GradientAccumulator.swift
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

/// Accumulated gradient of one source of an operation during backpropagation.
///
/// A backward closure computes the gradient of a source only when the accumulator ``isRequested``, and adds it with
/// ``add(_:)``. The backward pass of a fused operation gets the buffer of the gradient from ``buffer()`` instead.
///
/// No other tensor references the storage of the accumulated gradient, so a gradient without a graph is added in place.
struct GradientAccumulator<Element: NumericType, Device: DeviceType>: Sendable {
    /// Whether the source requires a gradient.
    let isRequested: Bool

    /// Shape of the source and of its gradient.
    let shape: [Int]

    /// Sum of the gradients that were added, or nil when no gradient was added yet.
    var value: Tensor<Element, Device>?

    /// Creates an accumulator for the gradient of a source.
    /// - Parameters:
    ///   - isRequested: Whether the source requires a gradient
    ///   - shape: Shape of the source
    ///   - value: Gradient that was accumulated before, or nil
    init(isRequested: Bool, shape: [Int], value: Tensor<Element, Device>? = nil) {
        self.isRequested = isRequested
        self.shape = shape
        self.value = value
    }

    /// Creates an accumulator for a source that does not require a gradient.
    static var notRequested: Self {
        Self(isRequested: false, shape: [])
    }

    /// Adds a gradient to the accumulated gradient, or stores it when no gradient was added yet.
    ///
    /// When neither tensor records a gradient graph, the gradient is added in place.
    /// - Parameter gradient: Gradient with the shape of the source
    mutating func add(_ gradient: Tensor<Element, Device>) {
        precondition(isRequested, "The source does not require a gradient.")
        precondition(gradient.shape == shape, "The gradient must have the shape of its source.")
        guard var existing = value.take() else {
            value = gradient
            return
        }
        guard !gradient.requiresGradient, !existing.requiresGradient else {
            value = existing + gradient
            return
        }
        // The accumulated gradient was taken out of the accumulator, so the write does not copy it.
        let target = existing.mutableValues.values
        Device.Engine.vAdd(lhs: Buffer(target), rhs: gradient.values.values, result: target, count: gradient.count)
        value = existing
    }

    /// Returns the buffer that a backward requirement of ``FusedOperationsType`` writes the gradient into, or nil when the
    /// gradient is not requested.
    ///
    /// Without an accumulated gradient, the accumulator stores a new gradient whose elements are not initialized, and the
    /// requirement writes every element. The buffer is valid while the accumulator holds the gradient.
    mutating func buffer() -> GradientBuffer<Element, Device>? {
        guard isRequested else {
            return nil
        }
        let adds = value != nil
        var gradient = value.take() ?? Tensor(uninitializedShape: shape)
        precondition(gradient.shape == shape, "The accumulated gradient must have the shape of its source.")
        // The gradient was taken out of the accumulator, so the write access does not copy it.
        let buffer = GradientBuffer(values: gradient.mutableValues, adds: adds)
        value = gradient
        return buffer
    }

    /// Returns the accumulator and leaves an accumulator without a value in its place.
    ///
    /// The returned accumulator is then the only reference to the storage of the accumulated gradient.
    mutating func take() -> Self {
        let taken = self
        value = nil
        return taken
    }
}

extension Tensor {
    /// Creates a tensor without context, whose elements are not initialized.
    init(uninitializedShape shape: [Int]) {
        self.init(using: Device.Memory.allocateBuffer(withShape: shape, type: Element.self), context: nil)
    }
}

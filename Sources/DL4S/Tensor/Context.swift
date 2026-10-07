//
//  Context.swift
//  DL4S
//
//  Created by Palle Klewitz on 19.10.19.
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
//

import Foundation

@usableFromInline
struct TensorContext<Element: NumericType, Device: DeviceType>: Sendable {
    /// Describes how the gradient of the result flows back to the sources of an operation.
    @usableFromInline
    enum BackpropagateFunction: Sendable {
        /// One closure per source.
        ///
        /// Each closure receives the gradient of the result and the accumulated gradient of its source.
        /// It adds the gradient of the result with respect to its source to the accumulator and returns the sum.
        case perSource([@Sendable (Tensor<Element, Device>, consuming Tensor<Element, Device>?) -> Tensor<Element, Device>])

        // swiftformat:disable spaceAroundBrackets
        // Operations use this form when one kernel produces the gradients of all sources at once, so the kernel runs once per
        // backward pass and no state needs to be shared between closures.
        /// One closure for all sources.
        ///
        /// The closure receives the gradient of the result and one accumulator per source.
        /// It returns the accumulated gradient of every source in source order, or nil for a source that does not require a gradient.
        case allSources(@Sendable (Tensor<Element, Device>, consuming [Tensor<Element, Device>?]) -> [Tensor<Element, Device>?])
        // swiftformat:enable spaceAroundBrackets
    }

    var tag: String?
    var sources: [Tensor<Element, Device>]
    var backpropagate: BackpropagateFunction
    #if DEBUG
    let operationStack = OperationGroup.operationStack
    #endif

    init(tag: String?, sources: [Tensor<Element, Device>], backpropagate: [@Sendable (Tensor<Element, Device>) -> Tensor<Element, Device>]) {
        self.init(tag: tag, sources: sources, backpropagateAccumulate: backpropagate.map { function in
            { resultGradient, accumulator in
                let gradient = function(resultGradient)
                return accumulator.map { $0 + gradient } ?? gradient
            }
        })
    }

    init(tag: String?, sources: [Tensor<Element, Device>], backpropagateAccumulate: [@Sendable (Tensor<Element, Device>, consuming Tensor<Element, Device>?) -> Tensor<Element, Device>]) {
        self.tag = tag
        self.sources = sources
        backpropagate = .perSource(backpropagateAccumulate)
    }

    // swiftformat:disable spaceAroundBrackets
    /// Creates a context with one backpropagation closure for all sources.
    ///
    /// - Parameters:
    ///   - tag: Name of the operation for graph output.
    ///   - sources: Tensors that the operation reads.
    ///   - backpropagateAll: Closure that receives the gradient of the result and owns one accumulator per source, and returns the accumulated gradient of every source.
    ///     It returns nil for a source that does not require a gradient.
    init(tag: String?, sources: [Tensor<Element, Device>], backpropagateAll: @escaping @Sendable (Tensor<Element, Device>, consuming [Tensor<Element, Device>?]) -> [Tensor<Element, Device>?]) {
        self.tag = tag
        self.sources = sources
        backpropagate = .allSources(backpropagateAll)
    }
    // swiftformat:enable spaceAroundBrackets
}

extension Tensor {
    /// Attaches the context of an operation to its result.
    ///
    /// The backward closure receives the gradient of the result and one accumulator per source in source order.
    /// It adds the gradient of every source whose accumulator is requested, with differentiable tensor operations.
    /// - Parameters:
    ///   - tag: Name of the operation for graph output. It is only evaluated when a source requires a gradient.
    ///   - sources: Tensors that the operation reads.
    ///   - backpropagate: Closure that adds the gradients of the sources to their accumulators.
    /// - Returns: The tensor with the context, or the tensor without changes when no source requires a gradient.
    func attachingContext(
        tag: @autoclosure () -> String,
        sources: [Self],
        backpropagate: @escaping @Sendable (_ resultGradient: Self, _ gradients: inout [GradientAccumulator<Element, Device>]) -> Void,
    ) -> Self {
        guard sources.contains(where: { $0.requiresGradient }) else {
            return self
        }
        let (isRequested, shapes) = (sources.map { $0.requiresGradient }, sources.map { $0.shape })
        var result = self
        result.context = TensorContext(tag: tag(), sources: sources, backpropagateAll: { resultGradient, accumulated in
            var accumulated = consume accumulated
            // The values are taken out of the array, so that the accumulators are the only references to their storage.
            var gradients = shapes.indices.map { index in
                GradientAccumulator<Element, Device>(isRequested: isRequested[index], shape: shapes[index], value: accumulated[index].take())
            }
            backpropagate(resultGradient, &gradients)
            return gradients.indices.map { gradients[$0].value.take() }
        })
        result.requiresGradient = true
        return result
    }

    /// Attaches the context of an operation with one source to its result.
    ///
    /// See ``attachingContext(tag:sources:backpropagate:)`` for the closure.
    func attachingContext(
        tag: @autoclosure () -> String,
        source: Self,
        backpropagate: @escaping @Sendable (_ resultGradient: Self, _ gradient: inout GradientAccumulator<Element, Device>) -> Void,
    ) -> Self {
        guard source.requiresGradient else {
            return self
        }
        // Most operations have one source, so this form uses the context with one closure per source, which needs no arrays of accumulators.
        let (tag, shape) = (tag(), source.shape)
        var result = self
        result.context = TensorContext(tag: tag, sources: [source], backpropagateAccumulate: [{ resultGradient, accumulated in
            var gradient = GradientAccumulator<Element, Device>(isRequested: true, shape: shape, value: accumulated)
            backpropagate(resultGradient, &gradient)
            guard let value = gradient.value.take() else {
                preconditionFailure("The backward pass of \(tag) did not compute the gradient of its source.")
            }
            return value
        }])
        result.requiresGradient = true
        return result
    }

    /// Attaches the context of a fused operation to its result.
    ///
    /// The context has two forms of the backward pass, and selects one in every backward pass:
    /// - `composed` computes the gradients with differentiable tensor operations. The context calls it when the gradient of
    ///   the result or an accumulated gradient requires a gradient, so that autograd can derive higher derivatives.
    /// - `fused` calls the backward requirement of ``FusedOperationsType`` with the elements of the gradient of the result and
    ///   one gradient buffer per source, which is nil when the source does not require a gradient. The context calls it in all
    ///   other cases.
    /// - Parameters:
    ///   - tag: Name of the operation for graph output. It is only evaluated when a source requires a gradient.
    ///   - sources: Tensors that the operation reads.
    ///   - composed: Closure that adds the gradients of the sources to their accumulators with tensor operations.
    ///   - fused: Closure that writes the gradients of the sources into their buffers.
    /// - Returns: The tensor with the context, or the tensor without changes when no source requires a gradient.
    func attachingContext(
        tag: @autoclosure () -> String,
        sources: [Self],
        composed: @escaping @Sendable (_ resultGradient: Self, _ gradients: inout [GradientAccumulator<Element, Device>]) -> Void,
        fused: @escaping @Sendable (_ resultGradient: ShapedBuffer<Element, Device>, _ gradients: [GradientBuffer<Element, Device>?]) -> Void,
    ) -> Self {
        attachingContext(tag: tag(), sources: sources) { resultGradient, gradients in
            if Self.recordsGraph(resultGradient, gradients) {
                composed(resultGradient, &gradients)
            } else {
                fused(resultGradient.values, gradients.indices.map { gradients[$0].buffer() })
            }
        }
    }

    /// Attaches the context of a fused operation with one source to its result.
    ///
    /// See ``attachingContext(tag:sources:composed:fused:)`` for the closures.
    func attachingContext(
        tag: @autoclosure () -> String,
        source: Self,
        composed: @escaping @Sendable (_ resultGradient: Self, _ gradient: inout GradientAccumulator<Element, Device>) -> Void,
        fused: @escaping @Sendable (_ resultGradient: ShapedBuffer<Element, Device>, _ gradient: GradientBuffer<Element, Device>) -> Void,
    ) -> Self {
        attachingContext(tag: tag(), source: source) { resultGradient, gradient in
            if Self.recordsGraph(resultGradient, [gradient]) {
                composed(resultGradient, &gradient)
            } else if let buffer = gradient.buffer() {
                fused(resultGradient.values, buffer)
            }
        }
    }

    /// Attaches the context of a fused operation with two sources to its result.
    ///
    /// See ``attachingContext(tag:sources:composed:fused:)`` for the closures.
    func attachingContext(
        tag: @autoclosure () -> String,
        sources first: Self,
        _ second: Self,
        composed: @escaping @Sendable (_ resultGradient: Self, _ firstGradient: inout GradientAccumulator<Element, Device>, _ secondGradient: inout GradientAccumulator<Element, Device>) -> Void,
        fused: @escaping @Sendable (_ resultGradient: ShapedBuffer<Element, Device>, _ firstGradient: GradientBuffer<Element, Device>?, _ secondGradient: GradientBuffer<Element, Device>?) -> Void,
    ) -> Self {
        guard first.requiresGradient || second.requiresGradient else {
            return self
        }
        return attachingContext(tag: tag(), sources: [first, second]) { resultGradient, gradients in
            var (firstGradient, secondGradient) = (gradients[0].take(), gradients[1].take())
            composed(resultGradient, &firstGradient, &secondGradient)
            (gradients[0], gradients[1]) = (firstGradient, secondGradient)
        } fused: { resultGradient, gradients in
            fused(resultGradient, gradients[0], gradients[1])
        }
    }

    /// Attaches the context of a fused operation with two sources and an optional third source, such as a bias, to its result.
    ///
    /// Without a third source, `composed` receives an accumulator that is not requested in its place, and `fused` receives nil.
    /// See ``attachingContext(tag:sources:composed:fused:)`` for the closures.
    func attachingContext(
        tag: @autoclosure () -> String,
        sources first: Self,
        _ second: Self,
        _ third: Self?,
        composed: @escaping @Sendable (
            _ resultGradient: Self,
            _ firstGradient: inout GradientAccumulator<Element, Device>,
            _ secondGradient: inout GradientAccumulator<Element, Device>,
            _ thirdGradient: inout GradientAccumulator<Element, Device>,
        ) -> Void,
        fused: @escaping @Sendable (
            _ resultGradient: ShapedBuffer<Element, Device>,
            _ firstGradient: GradientBuffer<Element, Device>?,
            _ secondGradient: GradientBuffer<Element, Device>?,
            _ thirdGradient: GradientBuffer<Element, Device>?,
        ) -> Void,
    ) -> Self {
        guard first.requiresGradient || second.requiresGradient || (third?.requiresGradient ?? false) else {
            return self
        }
        let hasThird = third != nil
        return attachingContext(tag: tag(), sources: [first, second] + (third.map { [$0] } ?? [])) { resultGradient, gradients in
            var (firstGradient, secondGradient) = (gradients[0].take(), gradients[1].take())
            var thirdGradient = hasThird ? gradients[2].take() : .notRequested
            composed(resultGradient, &firstGradient, &secondGradient, &thirdGradient)
            (gradients[0], gradients[1]) = (firstGradient, secondGradient)
            if hasThird {
                gradients[2] = thirdGradient
            }
        } fused: { resultGradient, gradients in
            fused(resultGradient, gradients[0], gradients[1], hasThird ? gradients[2] : nil)
        }
    }

    /// Whether a backward pass must record a gradient graph: the gradient of the result or an accumulated gradient requires a gradient.
    private static func recordsGraph(_ resultGradient: Self, _ gradients: [GradientAccumulator<Element, Device>]) -> Bool {
        resultGradient.requiresGradient || gradients.contains { $0.value?.requiresGradient ?? false }
    }

    /// Sums the tensor along the axes that broadcasting expanded, so that the result has the given shape.
    ///
    /// This is the gradient of a broadcast from `shape` to the shape of the tensor.
    /// - Parameter targetShape: Shape that broadcasts to the shape of the tensor.
    func reducingBroadcast(to targetShape: [Int]) -> Self {
        if shape == targetShape {
            return self
        }
        let paddedShape = Array(repeating: 1, count: dim - targetShape.count) + targetShape
        let reducedAxes = zip(paddedShape, shape).enumerated()
            .filter { $1.0 == 1 && $1.1 > 1 }
            .map { $0.offset }
        return reduceSum(along: reducedAxes).view(as: targetShape)
    }
}

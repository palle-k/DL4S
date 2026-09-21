//
//  Sequential.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
//  Copyright (c) 2019 - 2026 - Palle Klewitz
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

/// A layer that runs a sequence of layers.
///
/// Build with the ``LayerBuilder`` result builder. The outputs of every layer must match the inputs of
/// the next layer, and all layers must have the same `Parameter` and `Device` types.
/// A block needs at least two layers.
///
/// ```swift
/// let model = Sequential {
///     Dense<Float, CPU>(inputSize: 32, outputSize: 64)
///     Relu<Float, CPU>()
///     Dense<Float, CPU>(inputSize: 64, outputSize: 10)
///     Softmax<Float, CPU>()
/// }
/// ```
///
/// `model` has the type `Sequential<Dense<Float, CPU>, Relu<Float, CPU>, Dense<Float, CPU>, Softmax<Float, CPU>>`.
///
/// The tensors of the layers get their positions in the block as path segments, for example `0.weights` and
/// `2.bias`. A nested block is one element of the outer block, so its tensors get paths such as `1.0.weights`.
public struct Sequential<First: LayerType, each Middle: LayerType, Last: LayerType>: LayerType
    where Last.Parameter == First.Parameter, Last.Device == First.Device
{
    public typealias Inputs = First.Inputs
    public typealias Outputs = Last.Outputs
    public typealias Parameter = First.Parameter
    public typealias Device = First.Device

    public var first: First

    public var middle: (repeat each Middle)

    public var last: Last

    /// Tag for debugging purposes
    public var tag: String?

    // The builder checks that the layers fit together. This initializer is not public, so a block cannot be
    // created with mismatched layers.
    init(first: First, middle: (repeat each Middle), last: Last) {
        self.first = first
        self.middle = middle
        self.last = last
    }

    public mutating func visitTensors(_ visitor: inout TensorVisitor<First.Parameter, First.Device>) {
        visitor.element(&first)
        middle = (repeat Self.visit(each middle, with: &visitor))
        visitor.element(&last)
    }

    private static func visit<Layer: LayerType>(_ layer: Layer, with visitor: inout TensorVisitor<First.Parameter, First.Device>) -> Layer {
        var layer = layer
        visitor.element(&layer)
        return layer
    }

    public func callAsFunction(_ inputs: First.Inputs) -> Last.Outputs {
        if let tag {
            OperationGroup.capture(named: tag) {
                forward(inputs)
            }
        } else {
            forward(inputs)
        }
    }

    private func forward(_ inputs: First.Inputs) -> Last.Outputs {
        // Need to use Any because parameter packs do not allow expression of adjacent element requirements.
        // Though the type was already checked through the result builder.
        var value: Any = first(inputs)
        for layer in repeat each middle {
            value = Self.apply(layer, to: value)
        }
        // swiftlint:disable:next force_cast
        return last(value as! Last.Inputs)
    }

    // helper function makes parameter packs less painful to handle
    private static func apply<Layer: LayerType>(_ layer: Layer, to value: Any) -> Any {
        // swiftlint:disable:next force_cast
        layer(value as! Layer.Inputs)
    }
}

extension Sequential: Sendable where First: Sendable, repeat each Middle: Sendable, Last: Sendable {}

extension Sequential: Codable where First: Codable, repeat each Middle: Codable, Last: Codable {
    // The layers are encoded as a list in their order. The tag is a debugging aid and is not encoded.
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let first = try container.decode(First.self)
        let middle = try (repeat container.decode((each Middle).self))
        let last = try container.decode(Last.self)
        self.init(first: first, middle: (repeat each middle), last: last)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(first)
        repeat try container.encode(each middle)
        try container.encode(last)
    }
}

/// A result builder that creates a ``Sequential`` block from a sequence of layers.
///
/// The builder verifies that the outputs of every layer match the inputs of the next layer, and that all layers
/// have the same `Parameter` and `Device` types.
@resultBuilder
public enum LayerBuilder {
    public struct Start<Layer: LayerType> {
        let layer: Layer
    }

    public struct Partial<First: LayerType, each Middle: LayerType, Last: LayerType> {
        let first: First
        let middle: (repeat each Middle)
        let last: Last
    }

    /// Starts a block with its first layer.
    ///
    /// - Parameter first: First layer of the block.
    /// - Returns: A block with one layer.
    public static func buildPartialBlock<Layer: LayerType>(first: Layer) -> Start<Layer> {
        Start(layer: first)
    }

    /// Appends the second layer to a block.
    ///
    /// - Parameters:
    ///   - accumulated: Block with the first layer.
    ///   - next: Layer to append.
    /// - Returns: A block with two layers.
    public static func buildPartialBlock<First: LayerType, Next: LayerType>(accumulated: Start<First>, next: Next) -> Partial<First, Next>
        where Next.Inputs == First.Outputs, Next.Parameter == First.Parameter, Next.Device == First.Device
    {
        Partial(first: accumulated.layer, middle: (), last: next)
    }

    /// Appends a layer to a block with two or more layers.
    ///
    /// - Parameters:
    ///   - accumulated: Layers of the block that come before `next`.
    ///   - next: Layer to append.
    /// - Returns: The block with `next` at the end.
    public static func buildPartialBlock<First: LayerType, each Middle: LayerType, Last: LayerType, Next: LayerType>(accumulated: Partial<First, repeat each Middle, Last>, next: Next) -> Partial<First, repeat each Middle, Last, Next>
        where Next.Inputs == Last.Outputs, Next.Parameter == First.Parameter, Next.Device == First.Device
    {
        Partial(first: accumulated.first, middle: (repeat each accumulated.middle, accumulated.last), last: next)
    }

    /// Creates the ``Sequential`` layer for a finished block.
    ///
    /// - Parameter partial: Block with two or more layers.
    /// - Returns: The sequential layer.
    public static func buildFinalResult<First: LayerType, each Middle: LayerType, Last: LayerType>(_ partial: Partial<First, repeat each Middle, Last>) -> Sequential<First, repeat each Middle, Last>
        where Last.Parameter == First.Parameter, Last.Device == First.Device
    {
        Sequential(first: partial.first, middle: (repeat each partial.middle), last: partial.last)
    }
}

public extension Sequential {
    /// Creates a sequential layer from the layers in the builder closure.
    ///
    /// ```swift
    /// Sequential {
    ///     Dense<Float, CPU>(inputSize: 32, outputSize: 64)
    ///     Relu<Float, CPU>()
    ///     Dense<Float, CPU>(inputSize: 64, outputSize: 10)
    ///     Softmax<Float, CPU>()
    /// }
    /// ```
    ///
    /// - Parameter build: Builder closure with at least two layers.
    init(@LayerBuilder _ build: () -> Self) {
        self = build()
    }
}

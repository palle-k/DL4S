//
//  Sequential.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.10.19.
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

/// A sequential layer that concatenates the computations of two other layers.
///
/// With result builders, a sequential layer can be used to express sequential models in a type safe way.
///
/// Example:
/// ```
/// let model = Sequential {
///     Dense<Float, CPU>(inputSize: 32, outputSize: 64)
///     Relu<Float, CPU>()
///     Dense<Float, CPU>(inputSize: 64, outputSize: 10)
///     Softmax<Float, CPU>()
/// }
/// ```
/// The builder nests from the left: each layer is appended to the sequence of the layers before it.
/// `model` has the type `Sequential<Sequential<Sequential<Dense<Float, CPU>, Relu<Float, CPU>>, Dense<Float, CPU>>, Softmax<Float, CPU>>`.
public struct Sequential<First: LayerType, Second: LayerType>: LayerType where First.Outputs == Second.Inputs, First.Parameter == Second.Parameter, First.Device == Second.Device {
    /// First transform
    public var first: First
    
    /// Second transform
    public var second: Second
    
    /// Tag for debugging purposes
    public var tag: String? = nil
    
    public var parameters: [Tensor<First.Parameter, First.Device>] {
        get {
            first.parameters + second.parameters
        }
    }
    
    /// A sequential layer that concatenates the computations of two other layers.
    /// - Parameters:
    ///   - first: First transform
    ///   - second: Second transform
    public init(first: First, second: Second) {
        self.first = first
        self.second = second
    }
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<First.Parameter, First.Device>> & Sendable] {
        let firstPaths = parameterPaths(of: \.first)
        let secondPaths = parameterPaths(of: \.second)
        return firstPaths + secondPaths
    }
    
    public func callAsFunction(_ inputs: First.Inputs) -> Second.Outputs {
        if let tag = self.tag {
            return OperationGroup.capture(named: tag) {
                second.callAsFunction(first.callAsFunction(inputs))
            }
        } else {
            return second.callAsFunction(first.callAsFunction(inputs))
        }
    }
}

extension Sequential: Codable where First: Codable, Second: Codable {}

/// A layer builder can be used to create sequences of layers
@resultBuilder
public enum LayerBuilder {}

public extension LayerBuilder {
    /// Starts a sequence with its first layer.
    ///
    /// - Parameter first: First layer of the block.
    /// - Returns: The layer itself.
    static func buildPartialBlock<Layer: LayerType>(first: Layer) -> Layer {
        first
    }
    
    /// Appends a layer to the sequence.
    ///
    /// - Parameters:
    ///   - accumulated: Layers of the block that come before `next`, combined into one layer.
    ///   - next: Layer to append.
    /// - Returns: A sequential layer that runs `accumulated` and then `next`.
    static func buildPartialBlock<Accumulated: LayerType, Next: LayerType>(accumulated: Accumulated, next: Next) -> Sequential<Accumulated, Next>
        where Accumulated.Outputs == Next.Inputs, Accumulated.Parameter == Next.Parameter, Accumulated.Device == Next.Device
    {
        Sequential(first: accumulated, second: next)
    }
}

public extension Sequential {
    /// Creates a sequential layer with the sequence of transforms, that is specified in the provided layer builder builder closure.
    ///
    /// Example:
    /// ```
    /// Sequential {
    ///     Dense<Float, CPU>(inputSize: 32, outputSize: 64)
    ///     Relu<Float, CPU>()
    ///     Dense<Float, CPU>(inputSize: 64, outputSize: 10)
    ///     Softmax<Float, CPU>()
    /// }
    /// ```
    ///
    /// - Parameter build: Build block (function builder)
    init(@LayerBuilder _ build: () -> Self) {
        self = build()
    }
}

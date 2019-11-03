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
/// With function builders, a sequential layer can be used to express sequential models in a type safe way.
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
/// `model` will have a type `Sequential<Sequential<Dense<Float, CPU>, Relu<Float, CPU>>, Sequential<Dense<Float, CPU>, Softmax<Float, CPU>>>`.
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
    
    public var parameterPaths: [WritableKeyPath<Self, Tensor<First.Parameter, First.Device>>] {
        let firstPaths = first.parameterPaths.map {
            (\Self.first).appending(path: $0)
        }
        let secondPaths = second.parameterPaths.map {
            (\Self.second).appending(path: $0)
        }
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
@_functionBuilder
public enum LayerBuilder {}

public extension LayerBuilder {
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B>(_ a: A, _ b: B) -> Sequential<A, B>
        where A.Outputs == B.Inputs, A.Parameter == B.Parameter, A.Device == B.Device
    {
        Sequential(first: a, second: b)
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C>(_ a: A, _ b: B, _ c: C) -> Sequential<Sequential<A, B>, C> {
        buildBlock(buildBlock(a, b), c)
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D>(_ a: A, _ b: B, _ c: C, _ d: D) -> Sequential<Sequential<A, B>, Sequential<C, D>> {
        buildBlock(buildBlock(a, b), buildBlock(c, d))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E) -> Sequential<Sequential<Sequential<A, B>, C>, Sequential<D, E>> {
        buildBlock(buildBlock(a, b, c), buildBlock(d, e))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F) -> Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<E, F>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G) -> Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, G>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H) -> Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g, h))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, I> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), i)
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<I, J>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<I, J>, K>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<I, J>, Sequential<K, L>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<Sequential<I, J>, K>, Sequential<L, M>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<Sequential<I, J>, Sequential<K, L>>, Sequential<M, N>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N, _ o: O) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<Sequential<I, J>, Sequential<K, L>>, Sequential<Sequential<M, N>, O>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n, o))
    }
    
    /// Creates a sequential layer by concatenating the forward functions of the given layers.
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N, _ o: O, _ p: P) -> Sequential<Sequential<Sequential<Sequential<A, B>, Sequential<C, D>>, Sequential<Sequential<E, F>, Sequential<G, H>>>, Sequential<Sequential<Sequential<I, J>, Sequential<K, L>>, Sequential<Sequential<M, N>, Sequential<O, P>>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n, o, p))
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

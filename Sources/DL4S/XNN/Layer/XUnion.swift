//
//  XUnion.swift
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

public struct XUnion<First: XLayer, Second: XLayer>: XLayer where First.Outputs == Second.Inputs, First.Parameter == Second.Parameter, First.Device == Second.Device {
    public var first: First
    public var second: Second
    
    public var tag: String? = nil
    
    public var parameters: [XTensor<First.Parameter, First.Device>] {
        get {
            first.parameters + second.parameters
        }
        set {
            let c = first.parameters.count
            first.parameters = Array(newValue[..<c])
            second.parameters = Array(newValue[c...])
        }
    }
    
    public init(first: First, second: Second) {
        self.first = first
        self.second = second
    }
    
    public static var parameters: [WritableKeyPath<XUnion<First, Second>, XTensor<First.Parameter, First.Device>>] {
        First.parameters.map((\Self.first).appending(path:)) +
            Second.parameters.map((\Self.second).appending(path:))
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

extension XUnion: Codable where First: Codable, Second: Codable {}

@_functionBuilder
public struct LayerBuilder {}

public extension LayerBuilder {
    static func buildBlock<A, B>(_ a: A, _ b: B) -> XUnion<A, B>
        where A.Outputs == B.Inputs, A.Parameter == B.Parameter, A.Device == B.Device
    {
        XUnion(first: a, second: b)
    }
    
    static func buildBlock<A, B, C>(_ a: A, _ b: B, _ c: C) -> XUnion<XUnion<A, B>, C> {
        buildBlock(buildBlock(a, b), c)
    }
    
    static func buildBlock<A, B, C, D>(_ a: A, _ b: B, _ c: C, _ d: D) -> XUnion<XUnion<A, B>, XUnion<C, D>> {
        buildBlock(buildBlock(a, b), buildBlock(c, d))
    }
    
    static func buildBlock<A, B, C, D, E>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E) -> XUnion<XUnion<XUnion<A, B>, C>, XUnion<D, E>> {
        buildBlock(buildBlock(a, b, c), buildBlock(d, e))
    }
    
    static func buildBlock<A, B, C, D, E, F>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<E, F>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f))
    }
    
    static func buildBlock<A, B, C, D, E, F, G>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, G>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g, h))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, I> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), i)
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<I, J>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<I, J>, K>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<I, J>, XUnion<K, L>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<XUnion<I, J>, K>, XUnion<L, M>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<XUnion<I, J>, XUnion<K, L>>, XUnion<M, N>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N, _ o: O) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<XUnion<I, J>, XUnion<K, L>>, XUnion<XUnion<M, N>, O>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n, o))
    }
    
    static func buildBlock<A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L, _ m: M, _ n: N, _ o: O, _ p: P) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<XUnion<I, J>, XUnion<K, L>>, XUnion<XUnion<M, N>, XUnion<O, P>>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l, m, n, o, p))
    }
}

public extension XUnion {
    init(@LayerBuilder _ build: () -> Self) {
        self = build()
    }
}

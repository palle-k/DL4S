//
//  XLayer.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

import Foundation

@dynamicCallable
public protocol XLayer {
    associatedtype Inputs
    associatedtype Outputs
    associatedtype Parameter: NumericType
    associatedtype Device: DeviceType
    
    static var parameters: [WritableKeyPath<Self, XTensor<Parameter, Device>>] { get }
    var parameters: [XTensor<Parameter, Device>] { get set }
    
    func callAsFunction(_ inputs: Inputs) -> Outputs
}

public extension XLayer {
    func dynamicallyCall(withArguments arguments: [Inputs]) -> Outputs {
        return callAsFunction(arguments[0])
    }
}

public struct XDense<Element: RandomizableType, Device: DeviceType>: XLayer {
    public static var parameters: [WritableKeyPath<XDense<Element, Device>, XTensor<Element, Device>>] {[
        \.weights,
        \.bias
    ]}
    
    public var weights: XTensor<Element, Device>
    public var bias: XTensor<Element, Device>
    
    public var parameters: [XTensor<Element, Device>] {
        get {
            [weights, bias]
        }
        set {
            (weights, bias) = (newValue[0], newValue[1])
        }
    }
    
    public init(inputSize: Int, outputSize: Int) {
        weights = XTensor(xavierNormalWithShape: [inputSize, outputSize], requiresGradient: true)
        bias = XTensor(repeating: 0, shape: [outputSize], requiresGradient: true)
        
        #if DEBUG
        weights.tag = "W"
        bias.tag = "b"
        #endif
    }
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        inputs.matMul(weights) + bias
    }
}

public struct XTanh<Element: NumericType, Device: DeviceType>: XLayer {
    public static var parameters: [WritableKeyPath<XTanh<Element, Device>, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} set {} }
    
    public init() {}
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        inputs.tanh()
    }
}

public struct XSigmoid<Element: NumericType, Device: DeviceType>: XLayer {
    public static var parameters: [WritableKeyPath<XSigmoid<Element, Device>, XTensor<Element, Device>>] {[]}
    public var parameters: [XTensor<Element, Device>] { get {[]} set {} }
    
    public init() {}
    
    public func callAsFunction(_ inputs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        inputs.sigmoid()
    }
}

public struct XUnion<First: XLayer, Second: XLayer>: XLayer where First.Outputs == Second.Inputs, First.Parameter == Second.Parameter, First.Device == Second.Device {
    public var first: First
    public var second: Second
    
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
        return second.callAsFunction(first.callAsFunction(inputs))
    }
}

@_functionBuilder
public struct LayerBuilder {
    static func buildBlock<A: XLayer, B: XLayer>(_ a: A, _ b: B) -> XUnion<A, B>
        where A.Outputs == B.Inputs, A.Parameter == B.Parameter, A.Device == B.Device
    {
        XUnion(first: a, second: b)
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer>(_ a: A, _ b: B, _ c: C) -> XUnion<XUnion<A, B>, C> {
        buildBlock(buildBlock(a, b), c)
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D) -> XUnion<XUnion<A, B>, XUnion<C, D>> {
        buildBlock(buildBlock(a, b), buildBlock(c, d))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E) -> XUnion<XUnion<XUnion<A, B>, C>, XUnion<D, E>> {
        buildBlock(buildBlock(a, b, c), buildBlock(d, e))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<E, F>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, G>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer, H: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H) -> XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>> {
        buildBlock(buildBlock(a, b, c, d), buildBlock(e, f, g, h))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer, H: XLayer, I: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, I> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), i)
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer, H: XLayer, I: XLayer, J: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<I, J>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer, H: XLayer, I: XLayer, J: XLayer, K: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<I, J>, K>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k))
    }
    
    static func buildBlock<A: XLayer, B: XLayer, C: XLayer, D: XLayer, E: XLayer, F: XLayer, G: XLayer, H: XLayer, I: XLayer, J: XLayer, K: XLayer, L: XLayer>(_ a: A, _ b: B, _ c: C, _ d: D, _ e: E, _ f: F, _ g: G, _ h: H, _ i: I, _ j: J, _ k: K, _ l: L) -> XUnion<XUnion<XUnion<XUnion<A, B>, XUnion<C, D>>, XUnion<XUnion<E, F>, XUnion<G, H>>>, XUnion<XUnion<I, J>, XUnion<K, L>>> {
        buildBlock(buildBlock(a, b, c, d, e, f, g, h), buildBlock(i, j, k, l))
    }
}

public extension XUnion {
    init(@LayerBuilder _ build: () -> Self) {
        self = build()
    }
}

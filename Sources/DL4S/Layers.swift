//
//  Layers.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.02.19.
//

import Foundation


public protocol Layer {
    associatedtype Element: NumericType
    
    var parameters: [Vector<Element>] { get }
    var trainable: Bool { get nonmutating set }
    
    func forward(_ inputs: [Vector<Element>]) -> Vector<Element>
}

public extension Layer {
    func forward(_ inputs: Vector<Element>...) -> Vector<Element> {
        return forward(inputs)
    }
    
    func asAny() -> AnyLayer<Element> {
        return AnyLayer(self)
    }
}

public struct AnyLayer<Element: NumericType>: Layer {
    
    private let getParams: () -> [Vector<Element>]
    private let performForward: ([Vector<Element>]) -> (Vector<Element>)
    private let getTrainable: () -> Bool
    private let setTrainable: (Bool) -> ()
    
    public var parameters: [Vector<Element>] {
        return getParams()
    }
    
    public var trainable: Bool {
        get {
            return getTrainable()
        }
        nonmutating set {
            setTrainable(newValue)
        }
    }
    
    public init<L: Layer>(_ layer: L) where L.Element == Element {
        self.getParams = {layer.parameters}
        self.performForward = {layer.forward($0)}
        self.getTrainable = {layer.trainable}
        self.setTrainable = {layer.trainable = $0}
    }
    
    public func forward(_ inputs: [Vector<Element>]) -> Vector<Element> {
        return performForward(inputs)
    }
}

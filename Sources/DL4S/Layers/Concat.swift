//
//  Concat.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.03.19.
//

import Foundation


public class Concat<Element: NumericType, Device: DeviceType>: Layer, Codable {
    public typealias Input = Element
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public init() {}
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        return stack(inputs.map {$0.T}).T
    }
}

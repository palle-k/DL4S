//
//  Dropout.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.03.19.
//

import Foundation


/// Sets some forwarded values to zero with a given probability during forward operations.
public class Dropout<Element: NumericType, Device: DeviceType>: Layer {
    public typealias Input = Element
    
    public var isTrainable: Bool {
        get {
            return false
        }
        set {
            // noop
        }
    }
    
    public var parameters: [Tensor<Element, Device>] {
        return []
    }
    
    /// If the Dropout layer is active, dropout is applied during forward operations.
    /// If the Dropout layer is inactive, it does not alter its input.
    public var isActive: Bool = true
    
    
    /// Rate with which dropout is applied (between 0: no dropout and 1: drop out everything)
    public var dropoutRate: Float
    
    
    /// Creates a Dropout layer with a given dropout probability between 0: no dropout and 1: drop out everything
    ///
    /// Sets some forwarded values to zero with a given probability during forward operations.
    ///
    /// - Parameter rate: Rate with which dropout is applied (between 0: no dropout and 1: drop out everything)
    public init(rate: Float) {
        self.dropoutRate = rate
    }
    
    public func forward(_ inputs: [Tensor<Element, Device>]) -> Tensor<Element, Device> {
        if isActive {
            let x = inputs[0]
            let mask: Tensor<Element, Device> = Random.bernoulli(p: (1 - dropoutRate), shape: Array(x.shape.dropFirst()))
            mask.tag = "DropoutMask"
            return x * mask
        } else {
            return inputs[0]
        }
    }
}

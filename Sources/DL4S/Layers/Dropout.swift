//
//  Dropout.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.03.19.
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

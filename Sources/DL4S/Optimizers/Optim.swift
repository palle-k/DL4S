//
//  Optim.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
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


/// An optimizer updates the values of tensors to reduce the value of an arbitrary
/// differentiable loss function
public protocol Optimizer {
    associatedtype Element: NumericType
    associatedtype Device: DeviceType
    
    /// Parameters that will be optimized in every step
    var parameters: [Tensor<Element, Device>] { get }
    
    /// Performs a single optimization step.
    func step()
    
    // Resets the internal state of the optimizer.
    func reset()
    
    // Fills the gradient of the optimized parameters with zeros as a preparation for the next optimization step
    func zeroGradient()
}

// Using a default implementation crashes the compiler.
public extension Optimizer {
    func zeroGradient() {
        for param in parameters {
            param.zeroGradient()
        }
    }
}

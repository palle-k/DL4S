//
//  VecArithmetic.swift
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



private struct AdditionOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    @_specialize(where Element == Float, Device == CPU)
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.shapedGradient else {
            return
        }
        
        if let lhsGradient = lhs.shapedGradient {
            Device.Engine.unbroadcastAdd(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, lhsGradient: lhsGradient)
        }
        if let rhsGradient = rhs.shapedGradient {
            // adding is commutative, just swap everything
            Device.Engine.unbroadcastAdd(lhs: rhs.shapedValues, rhs: lhs.shapedValues, resultGradient: vectorGradient, lhsGradient: rhsGradient)
        }
    }
    
    var symbol: String {
        return "+"
    }
}

private struct SubtractionOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.shapedGradient else {
            return
        }
        
        if let lhsGradient = lhs.shapedGradient {
            Device.Engine.unbroadcastSub(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, lhsGradient: lhsGradient)
        }
        if let rhsGradient = rhs.shapedGradient {
            Device.Engine.unbroadcastSub(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, rhsGradient: rhsGradient)
        }
    }
    
    var symbol: String {
        return "-"
    }
    
}

private struct MultiplicationOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.shapedGradient else {
            return
        }
        
        if let lhsGradient = lhs.shapedGradient {
            Device.Engine.unbroadcastMul(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, lhsGradient: lhsGradient)
        }
        if let rhsGradient = rhs.shapedGradient {
            // multiplying is commutative, just swap everything
            Device.Engine.unbroadcastMul(lhs: rhs.shapedValues, rhs: lhs.shapedValues, resultGradient: vectorGradient, lhsGradient: rhsGradient)
        }
    }
    
    var symbol: String {
        return "×"
    }
}

private struct DivisionOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.shapedGradient else {
            return
        }
        if let lhsGradient = lhs.shapedGradient {
            Device.Engine.unbroadcastDiv(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, lhsGradient: lhsGradient)
        }
        if let rhsGradient = rhs.shapedGradient {
            Device.Engine.unbroadcastDiv(lhs: lhs.shapedValues, rhs: rhs.shapedValues, resultGradient: vectorGradient, rhsGradient: rhsGradient)
        }
    }
    
    var symbol: String {
        return "÷"
    }
}

public extension Tensor {
    static func + (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: AdditionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Device.Engine.broadcastAdd(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        return result
    }
    
    static func - (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: SubtractionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Device.Engine.broadcastSub(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        return result
    }
    
    static func * (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Device.Engine.broadcastMul(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        return result
    }
    
    static func / (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Device.Engine.broadcastDiv(lhs: lhs.shapedValues, rhs: rhs.shapedValues, result: result.shapedValues)
        
        return result
    }
    
    static func += (lhs: inout Tensor<Element, Device>, rhs: Tensor<Element, Device>) {
        lhs = lhs + rhs
    }
    
    static func -= (lhs: inout Tensor<Element, Device>, rhs: Tensor<Element, Device>) {
        lhs = lhs - rhs
    }
}

public prefix func - <Element, Device>(value: Tensor<Element, Device>) -> Tensor<Element, Device> {
    return 0 - value
}

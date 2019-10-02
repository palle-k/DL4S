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
import TensorFlow


private struct AdditionOperation<Element: NumericType, Device: DeviceType>: BinaryTensorOperation {
    var lhs: Tensor<Element, Device>
    var rhs: Tensor<Element, Device>
    
    @_specialize(where Element == Float, Device == CPU)
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.shapedGradient else {
            return
        }
        
        if let lhsGradient = lhs.shapedGradient {
            if lhs.shape == rhs.shape {
                Device.Engine.broadcastAdd(lhs: vectorGradient, rhs: lhsGradient, result: lhsGradient)
            } else {
                let tmp = Device.Memory.allocateBuffer(withShape: lhs.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp)
                }
                
                let lhsPadded = Array(repeating: 1, count: vector.dim - lhs.dim) + lhs.shape
                let lhsReducedAxes = zip(lhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmpReducedShape = lhsPadded
                
                for a in lhsReducedAxes.reversed() {
                    tmpReducedShape.remove(at: a)
                }
                
                Device.Engine.reduceSum(values: vectorGradient, result: tmp.reshaped(to: tmpReducedShape), axes: lhsReducedAxes)
                Device.Engine.broadcastAdd(lhs: tmp, rhs: lhsGradient, result: lhsGradient)
            }
        }
        if let rhsGradient = rhs.shapedGradient {
            if lhs.shape == rhs.shape {
                Device.Engine.broadcastAdd(lhs: vectorGradient, rhs: rhsGradient, result: rhsGradient)
            } else {
                let tmp = Device.Memory.allocateBuffer(withShape: rhs.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp)
                }
                
                let rhsPadded = Array(repeating: 1, count: vector.dim - rhs.dim) + rhs.shape
                let rhsReducedAxes = zip(rhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmpReducedShape = rhsPadded
                
                for a in rhsReducedAxes.reversed() {
                    tmpReducedShape.remove(at: a)
                }
                
                Device.Engine.reduceSum(values: vectorGradient, result: tmp.reshaped(to: tmpReducedShape), axes: rhsReducedAxes)
                Device.Engine.broadcastAdd(lhs: tmp, rhs: rhsGradient, result: rhsGradient)
            }
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
            if lhs.shape == rhs.shape {
                Device.Engine.broadcastAdd(lhs: vectorGradient, rhs: lhsGradient, result: lhsGradient)
            } else {
                let tmp = Device.Memory.allocateBuffer(withShape: lhs.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp)
                }
                
                let lhsPadded = Array(repeating: 1, count: vector.dim - lhs.dim) + lhs.shape
                let lhsReducedAxes = zip(lhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmpReducedShape = lhsPadded
                
                for a in lhsReducedAxes.reversed() {
                    tmpReducedShape.remove(at: a)
                }
                
                Device.Engine.reduceSum(values: vectorGradient, result: tmp.reshaped(to: tmpReducedShape), axes: lhsReducedAxes)
                Device.Engine.broadcastAdd(lhs: tmp, rhs: lhsGradient, result: lhsGradient)
            }
        }
        if let rhsGradient = rhs.shapedGradient {
            if lhs.shape == rhs.shape {
                Device.Engine.broadcastSub(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient)
            } else {
                let tmp = Device.Memory.allocateBuffer(withShape: rhs.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp)
                }
                
                let rhsPadded = Array(repeating: 1, count: vector.dim - rhs.dim) + rhs.shape
                let rhsReducedAxes = zip(rhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmpReducedShape = rhsPadded
                
                for a in rhsReducedAxes.reversed() {
                    tmpReducedShape.remove(at: a)
                }
                
                Device.Engine.reduceSum(values: vectorGradient, result: tmp.reshaped(to: tmpReducedShape), axes: rhsReducedAxes)
                Device.Engine.broadcastSub(lhs: rhsGradient, rhs: tmp, result: rhsGradient)
            }
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
            if lhs.shape == rhs.shape {
                Device.Engine.vMA(lhs: rhs.values, rhs: vectorGradient.values, add: lhsGradient.values, result: lhsGradient.values, count: lhsGradient.count)
            } else {
                let tmp1 = Device.Memory.allocateBuffer(withShape: lhs.shape, type: Element.self)
                let tmp2 = Device.Memory.allocateBuffer(withShape: vector.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp1)
                    Device.Memory.free(tmp2)
                }
                
                let lhsPadded = Array(repeating: 1, count: vector.dim - lhs.dim) + lhs.shape
                let lhsReducedAxes = zip(lhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmp1reducedShape = lhsPadded
                
                for a in lhsReducedAxes.reversed() {
                    tmp1reducedShape.remove(at: a)
                }
                
                Device.Engine.broadcastMul(lhs: rhs.shapedValues, rhs: vectorGradient, result: tmp2)
                Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: lhsReducedAxes)
                Device.Engine.broadcastAdd(lhs: tmp1, rhs: lhsGradient, result: lhsGradient)
            }
        }
        if let rhsGradient = rhs.shapedGradient {
            if lhs.shape == rhs.shape {
                Device.Engine.vMA(lhs: lhs.values, rhs: vectorGradient.values, add: rhsGradient.values, result: rhsGradient.values, count: rhsGradient.count)
            } else {
                let tmp1 = Device.Memory.allocateBuffer(withShape: rhs.shape, type: Element.self)
                let tmp2 = Device.Memory.allocateBuffer(withShape: vector.shape, type: Element.self)
                defer {
                    Device.Memory.free(tmp1)
                    Device.Memory.free(tmp2)
                }
                
                let rhsPadded = Array(repeating: 1, count: vector.dim - rhs.dim) + rhs.shape
                let rhsReducedAxes = zip(rhsPadded, vector.shape).enumerated()
                    .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                
                var tmp1reducedShape = rhsPadded
                
                for a in rhsReducedAxes.reversed() {
                    tmp1reducedShape.remove(at: a)
                }
                
                Device.Engine.broadcastMul(lhs: lhs.shapedValues, rhs: vectorGradient, result: tmp2)
                Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: rhsReducedAxes)
                Device.Engine.broadcastAdd(lhs: tmp1, rhs: rhsGradient, result: rhsGradient)
            }
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
            let tmp1 = Device.Memory.allocateBuffer(withShape: lhs.shape, type: Element.self)
            let tmp2 = Device.Memory.allocateBuffer(withShape: vector.shape, type: Element.self)
            defer {
                Device.Memory.free(tmp1)
                Device.Memory.free(tmp2)
            }
            
            let lhsPadded = Array(repeating: 1, count: vector.dim - lhs.dim) + lhs.shape
            let lhsReducedAxes = zip(lhsPadded, vector.shape).enumerated()
                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
            
            var tmp1reducedShape = lhsPadded
            
            for a in lhsReducedAxes.reversed() {
                tmp1reducedShape.remove(at: a)
            }
            
            Device.Engine.broadcastDiv(lhs: vectorGradient, rhs: rhs.shapedValues, result: tmp2)
            Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: lhsReducedAxes)
            Device.Engine.broadcastAdd(lhs: tmp1, rhs: lhsGradient, result: lhsGradient)
        }
        if let rhsGradient = rhs.shapedGradient {
            let tmp1 = Device.Memory.allocateBuffer(withShape: rhs.shape, type: Element.self)
            let tmp2 = Device.Memory.allocateBuffer(withShape: vector.shape, type: Element.self)
            defer {
                Device.Memory.free(tmp1)
                Device.Memory.free(tmp2)
            }
            
            let rhsPadded = Array(repeating: 1, count: vector.dim - rhs.dim) + rhs.shape
            let rhsReducedAxes = zip(rhsPadded, vector.shape).enumerated()
                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
            
            var tmp1reducedShape = rhsPadded
            
            for a in rhsReducedAxes.reversed() {
                tmp1reducedShape.remove(at: a)
            }
            
            Device.Engine.broadcastMul(lhs: vectorGradient, rhs: lhs.shapedValues, result: tmp2)
            Device.Engine.broadcastDiv(lhs: tmp2, rhs: rhs.shapedValues, result: tmp2)
            Device.Engine.broadcastDiv(lhs: tmp2, rhs: rhs.shapedValues, result: tmp2)
            Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: rhsReducedAxes)
            Device.Engine.broadcastSub(lhs: rhsGradient, rhs: tmp1, result: rhsGradient)
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

//@differentiable(vjp: Swift.print where Element: NumericType)
//func addTF<Element, Device>(_ lhs: Tensor<Element, Device>, _ rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
//    
//}

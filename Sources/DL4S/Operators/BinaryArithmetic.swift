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
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = Device.Engine.sum(val: vectorGradient, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                Device.Engine.vAdd(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = Device.Engine.sum(val: vectorGradient, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee + sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient.advanced(by: offset), result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                Device.Engine.vAdd(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    Device.Engine.vAdd(lhs: rhsGradient, rhs: vectorGradient.advanced(by: offset), result: rhsGradient, count: rhs.count)
                }
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
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = Device.Engine.sum(val: vectorGradient, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                Device.Engine.vSub(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = Device.Engine.sum(val: vectorGradient, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee - sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient.advanced(by: offset), result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                Device.Engine.vSub(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                Device.Engine.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    Device.Engine.vSub(lhs: rhsGradient, rhs: vectorGradient.advanced(by: offset), result: rhsGradient, count: rhs.count)
                }
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
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                lhsGradient.pointee = lhsGradient.pointee + Device.Engine.dot(lhs: vectorGradient, rhs: rhs.values, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                Device.Engine.vsMulVAdd(lhs: vectorGradient, rhs: lhs.values.pointee, add: rhsGradient, result: rhsGradient, count: vector.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Device.Engine.vsMulVAdd(lhs: vectorGradient, rhs: rhs.values.pointee, add: lhsGradient, result: lhsGradient, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                rhsGradient.pointee = rhsGradient.pointee + Device.Engine.dot(lhs: vectorGradient, rhs: lhs.values, count: vector.count)
            }
        } else if lhs.dim < rhs.dim {
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    Device.Engine.vMA(lhs: rhs.values.advanced(by: offset), rhs: vectorGradient.advanced(by: offset), add: lhsGradient, result: lhsGradient, count: lhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    Device.Engine.vMA(lhs: lhs.values, rhs: vectorGradient.advanced(by: offset), add: rhsGradient.advanced(by: offset), result: rhsGradient.advanced(by: offset), count: lhs.count)
                }
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    Device.Engine.vMA(lhs: rhs.values, rhs: vectorGradient.advanced(by: offset), add: lhsGradient.advanced(by: offset), result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    Device.Engine.vMA(lhs: lhs.values.advanced(by: offset), rhs: vectorGradient.advanced(by: offset), add: rhsGradient, result: rhsGradient, count: rhs.count)
                }
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
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            let temp = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                Device.Engine.vDiv(lhs: vectorGradient, rhs: rhs.values, result: temp, count: rhs.count)
                let sum = Device.Engine.sum(val: temp, count: rhs.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                Device.Engine.vSquare(values: rhs.values, result: temp, count: rhs.count)
                Device.Engine.vNeg(val: temp, result: temp, count: rhs.count)
                Device.Engine.svDiv(lhs: lhs.values.pointee, rhs: temp, result: temp, count: rhs.count)
                Device.Engine.vMA(lhs: temp, rhs: vectorGradient, add: rhsGradient, result: rhsGradient, count: rhs.count)
            }
            
            Device.Memory.free(temp)
            
        } else if rhs.dim == 0 {
            let temp = Device.Memory.allocateBuffer(withCapacity: vector.count, type: Element.self)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                Device.Engine.vsMul(lhs: vectorGradient, rhs: rhs.values.pointee, result: temp, count: vector.count)
                Device.Engine.vAdd(lhs: temp, rhs: lhsGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                let rhsVal = rhs.values.pointee
                let invRhsSq = 1 / (-rhsVal * rhsVal)
                Device.Engine.vsMul(lhs: lhs.values, rhs: invRhsSq, result: temp, count: lhs.count)
                Device.Engine.vMul(lhs: temp, rhs: vectorGradient, result: temp, count: lhs.count)
                
                rhsGradient.pointee = rhsGradient.pointee + Device.Engine.sum(val: temp, count: lhs.count)
            }
            
            Device.Memory.free(temp)
            
        } else if lhs.dim < rhs.dim {
            let tempA = Device.Memory.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            let tempB = Device.Memory.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            
            if rhs.requiresGradient {
                Device.Engine.vSquare(values: rhs.values, result: tempA, count: rhs.count)
                Device.Engine.vNeg(val: tempA, result: tempA, count: rhs.count)
                Device.Engine.vDiv(lhs: vectorGradient, rhs: tempA, result: tempA, count: rhs.count)
            }
            
            
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                
                // dvector/drhs = -lhs/rhs^2 * dvector = lhs * tempA
                // dvector/dlhs = 1/rhs * dvector
                
                if rhs.requiresGradient {
                    Device.Engine.vMul(lhs: lhs.values, rhs: tempA.advanced(by: offset), result: tempA.advanced(by: offset), count: lhs.count)
                }
                
                if let lhsGradient = lhs.gradient {
                    Device.Engine.vDiv(lhs: vectorGradient.advanced(by: offset), rhs: rhs.values.advanced(by: offset), result: tempB, count: lhs.count)
                    Device.Engine.vAdd(lhs: lhsGradient, rhs: tempB, result: lhsGradient, count: lhs.count)
                }
                
            }
            
            if let rhsGradient = rhs.gradient {
                Device.Engine.vAdd(lhs: rhsGradient, rhs: tempA, result: rhsGradient, count: rhs.count)
            }
            Device.Memory.free(tempA)
            Device.Memory.free(tempB)
            
        } else /*if rhs.dim <= lhs.dim*/ {
            let lhsTimesGrad = Device.Memory.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            let negInvRhsSq = Device.Memory.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            let invRhs = Device.Memory.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            
            if rhs.requiresGradient {
                Device.Engine.vSquare(values: rhs.values, result: negInvRhsSq, count: rhs.count)
                Device.Engine.svDiv(lhs: -1, rhs: negInvRhsSq, result: negInvRhsSq, count: rhs.count)
                
                Device.Engine.vMul(lhs: lhs.values, rhs: vectorGradient, result: lhsTimesGrad, count: lhs.count)
            }
            
            if lhs.requiresGradient {
                Device.Engine.svDiv(lhs: 1, rhs: rhs.values, result: invRhs, count: rhs.count)
            }
            
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                
                // dvector/dlhs = dvector / rhs
                // dvector/drhs = (lhs * dvector) (/ -rhs^2)
                
                if let lhsGradient = lhs.gradient {
                    Device.Engine.vMA(lhs: vectorGradient.advanced(by: offset), rhs: invRhs, add: lhsGradient.advanced(by: offset), result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                
                if let rhsGradient = rhs.gradient {
                    Device.Engine.vMA(lhs: lhsTimesGrad.advanced(by: offset), rhs: negInvRhsSq, add: rhsGradient, result: rhsGradient, count: rhs.count)
                }
            }
            
            Device.Memory.free(lhsTimesGrad)
            Device.Memory.free(negInvRhsSq)
            Device.Memory.free(invRhs)
        }
    }
    
    var symbol: String {
        return "÷"
    }
}

public extension Tensor {
    static func + (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: AdditionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Device.Engine.vsAdd(lhs: rhs.values, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Device.Engine.vsAdd(lhs: lhs.values, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Device.Engine.vAdd(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Device.Engine.vAdd(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func - (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: SubtractionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Device.Engine.vsAdd(lhs: rhs.values, rhs: -lhs.values.pointee, result: result.values, count: result.count)
            Device.Engine.vNeg(val: result.values, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Device.Engine.vsAdd(lhs: lhs.values, rhs: -rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Device.Engine.vSub(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Device.Engine.vSub(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func * (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Device.Engine.vsMul(lhs: rhs.values, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Device.Engine.vsMul(lhs: lhs.values, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Device.Engine.vMul(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Device.Engine.vMul(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func / (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, Device>(
            shape: resultShape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Device.Engine.svDiv(lhs: lhs.values.pointee, rhs: rhs.values, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Device.Engine.vsMul(lhs: lhs.values, rhs: 1 / rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Device.Engine.vDiv(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Device.Engine.vDiv(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
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

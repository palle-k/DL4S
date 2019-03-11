//
//  VecArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct AdditionOperation<Element: NumericType, DeviceType: Device>: BinaryTensorOperation {
    var lhs: Tensor<Element, DeviceType>
    var rhs: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = DeviceType.EngineType.sum(val: vectorGradient, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vAdd(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = DeviceType.EngineType.sum(val: vectorGradient, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee + sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient.advanced(by: offset), result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vAdd(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    DeviceType.EngineType.vAdd(lhs: rhsGradient, rhs: vectorGradient.advanced(by: offset), result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "+"
    }
}

private struct SubtractionOperation<Element: NumericType, DeviceType: Device>: BinaryTensorOperation {
    var lhs: Tensor<Element, DeviceType>
    var rhs: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = DeviceType.EngineType.sum(val: vectorGradient, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vSub(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = DeviceType.EngineType.sum(val: vectorGradient, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee - sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient.advanced(by: offset), result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vSub(lhs: rhsGradient, rhs: vectorGradient, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: vectorGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    DeviceType.EngineType.vSub(lhs: rhsGradient, rhs: vectorGradient.advanced(by: offset), result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "-"
    }
    
}

private struct MultiplicationOperation<Element: NumericType, DeviceType: Device>: BinaryTensorOperation {
    var lhs: Tensor<Element, DeviceType>
    var rhs: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                lhsGradient.pointee = lhsGradient.pointee + DeviceType.EngineType.dot(lhs: vectorGradient, rhs: rhs.values, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vsMulVAdd(lhs: vectorGradient, rhs: lhs.values.pointee, add: rhsGradient, result: rhsGradient, count: vector.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                DeviceType.EngineType.vsMulVAdd(lhs: vectorGradient, rhs: rhs.values.pointee, add: lhsGradient, result: lhsGradient, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                rhsGradient.pointee = rhsGradient.pointee + DeviceType.EngineType.dot(lhs: vectorGradient, rhs: lhs.values, count: vector.count)
            }
        } else if lhs.dim < rhs.dim {
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    DeviceType.EngineType.vMA(lhs: rhs.values.advanced(by: offset), rhs: vectorGradient.advanced(by: offset), add: lhsGradient, result: lhsGradient, count: lhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    DeviceType.EngineType.vMA(lhs: lhs.values, rhs: vectorGradient.advanced(by: offset), add: rhsGradient.advanced(by: offset), result: rhsGradient.advanced(by: offset), count: lhs.count)
                }
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    DeviceType.EngineType.vMA(lhs: rhs.values, rhs: vectorGradient.advanced(by: offset), add: lhsGradient.advanced(by: offset), result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    DeviceType.EngineType.vMA(lhs: lhs.values.advanced(by: offset), rhs: vectorGradient.advanced(by: offset), add: rhsGradient, result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "×"
    }
}

private struct DivisionOperation<Element: NumericType, DeviceType: Device>: BinaryTensorOperation {
    var lhs: Tensor<Element, DeviceType>
    var rhs: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            let temp: Buffer<Element, DeviceType> = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                DeviceType.EngineType.vDiv(lhs: vectorGradient, rhs: rhs.values, result: temp, count: rhs.count)
                let sum = Element.sum(val: temp, count: rhs.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                DeviceType.EngineType.vSquare(values: rhs.values, result: temp, count: rhs.count)
                DeviceType.EngineType.vNeg(val: temp, result: temp, count: rhs.count)
                DeviceType.EngineType.svDiv(lhs: lhs.values.pointee, rhs: temp, result: temp, count: rhs.count)
                DeviceType.EngineType.vMA(lhs: temp, rhs: vectorGradient, add: rhsGradient, result: rhsGradient, count: rhs.count)
            }
            
            DeviceType.MemoryOperatorType.free(temp)
            
        } else if rhs.dim == 0 {
            let temp: Buffer<Element, DeviceType> = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                DeviceType.EngineType.vsMul(lhs: vectorGradient, rhs: rhs.values.pointee, result: temp, count: vector.count)
                Element.vAdd(lhs: temp, rhs: lhsGradient, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                let rhsVal = rhs.values.pointee
                let invRhsSq = 1 / (-rhsVal * rhsVal)
                DeviceType.EngineType.vsMul(lhs: lhs.values, rhs: invRhsSq, result: temp, count: lhs.count)
                DeviceType.EngineType.vMul(lhs: temp, rhs: vectorGradient, result: temp, count: lhs.count)
                
                rhsGradient.pointee = rhsGradient.pointee + Element.sum(val: temp, count: lhs.count)
            }
            
            DeviceType.MemoryOperatorType.free(temp)
            
        } else if lhs.dim < rhs.dim {
            let tempA = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            let tempB = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            
            if rhs.requiresGradient {
                DeviceType.EngineType.vSquare(values: rhs.values, result: tempA, count: rhs.count)
                DeviceType.EngineType.vNeg(val: tempA, result: tempA, count: rhs.count)
                DeviceType.EngineType.vDiv(lhs: vectorGradient, rhs: tempA, result: tempA, count: rhs.count)
            }
            
            
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                
                // dvector/drhs = -lhs/rhs^2 * dvector = lhs * tempA
                // dvector/dlhs = 1/rhs * dvector
                
                if rhs.requiresGradient {
                    DeviceType.EngineType.vMul(lhs: lhs.values, rhs: tempA.advanced(by: offset), result: tempA.advanced(by: offset), count: lhs.count)
                }
                
                if let lhsGradient = lhs.gradient {
                    DeviceType.EngineType.vDiv(lhs: vectorGradient.advanced(by: offset), rhs: rhs.values.advanced(by: offset), result: tempB, count: lhs.count)
                    DeviceType.EngineType.vAdd(lhs: lhsGradient, rhs: tempB, result: lhsGradient, count: lhs.count)
                }
                
            }
            
            if let rhsGradient = rhs.gradient {
                DeviceType.EngineType.vAdd(lhs: rhsGradient, rhs: tempA, result: rhsGradient, count: rhs.count)
            }
            DeviceType.MemoryOperatorType.free(tempA)
            DeviceType.MemoryOperatorType.free(tempB)
            
        } else /*if rhs.dim <= lhs.dim*/ {
            let lhsTimesGrad = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: lhs.count, type: Element.self)
            let negInvRhsSq = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            let invRhs = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: rhs.count, type: Element.self)
            
            if rhs.requiresGradient {
                DeviceType.EngineType.vSquare(values: rhs.values, result: negInvRhsSq, count: rhs.count)
                DeviceType.EngineType.svDiv(lhs: -1, rhs: negInvRhsSq, result: negInvRhsSq, count: rhs.count)
                
                DeviceType.EngineType.vMul(lhs: lhs.values, rhs: vectorGradient, result: lhsTimesGrad, count: lhs.count)
            }
            
            if lhs.requiresGradient {
                DeviceType.EngineType.svDiv(lhs: 1, rhs: rhs.values, result: invRhs, count: rhs.count)
            }
            
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                
                // dvector/dlhs = dvector / rhs
                // dvector/drhs = (lhs * dvector) (/ -rhs^2)
                
                if let lhsGradient = lhs.gradient {
                    DeviceType.EngineType.vMA(lhs: vectorGradient.advanced(by: offset), rhs: invRhs, add: lhsGradient.advanced(by: offset), result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                
                if let rhsGradient = rhs.gradient {
                    DeviceType.EngineType.vMA(lhs: lhsTimesGrad.advanced(by: offset), rhs: negInvRhsSq, add: rhsGradient, result: rhsGradient, count: rhs.count)
                }
            }
            
            DeviceType.MemoryOperatorType.free(lhsTimesGrad)
            DeviceType.MemoryOperatorType.free(negInvRhsSq)
            DeviceType.MemoryOperatorType.free(invRhs)
        }
    }
    
    var symbol: String {
        return "÷"
    }
}

public extension Tensor {
    static func + (lhs: Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, DeviceType>(
            shape: resultShape,
            parent: nil,
            context: AdditionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            DeviceType.EngineType.vsAdd(lhs: rhs.values, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            DeviceType.EngineType.vsAdd(lhs: lhs.values, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vAdd(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vAdd(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func - (lhs: Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, DeviceType>(
            shape: resultShape,
            parent: nil,
            context: SubtractionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            DeviceType.EngineType.vsAdd(lhs: rhs.values, rhs: -lhs.values.pointee, result: result.values, count: result.count)
            DeviceType.EngineType.vNeg(val: result.values, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Element.vsAdd(lhs: lhs.values, rhs: -rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vSub(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vSub(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func * (lhs: Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, DeviceType>(
            shape: resultShape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            DeviceType.EngineType.vsMul(lhs: rhs.values, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            DeviceType.EngineType.vsMul(lhs: lhs.values, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vMul(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vMul(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func / (lhs: Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element, DeviceType>(
            shape: resultShape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            DeviceType.EngineType.svDiv(lhs: lhs.values.pointee, rhs: rhs.values, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            DeviceType.EngineType.vsMul(lhs: lhs.values, rhs: 1 / rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vDiv(lhs: lhs.values, rhs: rhs.values.advanced(by: offset), result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                DeviceType.EngineType.vDiv(lhs: lhs.values.advanced(by: offset), rhs: rhs.values, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func += (lhs: inout Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) {
        lhs = lhs + rhs
    }
    
    static func -= (lhs: inout Tensor<Element, DeviceType>, rhs: Tensor<Element, DeviceType>) {
        lhs = lhs - rhs
    }
}

public prefix func - <Element, DeviceType>(value: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    return 0 - value
}

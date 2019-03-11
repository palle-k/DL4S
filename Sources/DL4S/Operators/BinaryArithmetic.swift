//
//  VecArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct AdditionOperation<Element: NumericType>: BinaryTensorOperation {
    var lhs: Tensor<Element>
    var rhs: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = Element.sum(val: vectorGradient.immutable, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                Element.vAdd(lhs: rhsGradient.immutable, rhs: vectorGradient.immutable, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.immutable, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = Element.sum(val: vectorGradient.immutable, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee + sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.advanced(by: offset).immutable, result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                Element.vAdd(lhs: rhsGradient.immutable, rhs: vectorGradient.immutable, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.immutable, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    Element.vAdd(lhs: rhsGradient.immutable, rhs: vectorGradient.advanced(by: offset).immutable, result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "+"
    }
}

private struct SubtractionOperation<Element: NumericType>: BinaryTensorOperation {
    var lhs: Tensor<Element>
    var rhs: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                let sum = Element.sum(val: vectorGradient.immutable, count: vector.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                Element.vSub(lhs: rhsGradient.immutable, rhs: vectorGradient.immutable, result: rhsGradient, count: rhs.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.immutable, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let sum = Element.sum(val: vectorGradient.immutable, count: vector.count)
                rhsGradient.pointee = rhsGradient.pointee - sum
            }
        } else if lhs.dim < rhs.dim {
            if let lhsGradient = lhs.gradient {
                let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                    Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.advanced(by: offset).immutable, result: lhsGradient, count: lhs.count)
                }
            }
            if let rhsGradient = rhs.gradient {
                Element.vSub(lhs: rhsGradient.immutable, rhs: vectorGradient.immutable, result: rhsGradient, count: rhs.count)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            if let lhsGradient = lhs.gradient {
                Element.vAdd(lhs: lhsGradient.immutable, rhs: vectorGradient.immutable, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
                for idx in iterate(Array(shapePrefix)) {
                    let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                    Element.vSub(lhs: rhsGradient.immutable, rhs: vectorGradient.advanced(by: offset).immutable, result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "-"
    }
    
}

private struct MultiplicationOperation<Element: NumericType>: BinaryTensorOperation {
    var lhs: Tensor<Element>
    var rhs: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                lhsGradient.pointee = lhsGradient.pointee + Element.dot(lhs: vectorGradient.immutable, rhs: rhs.values.immutable, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                Element.vsMulVAdd(lhs: vectorGradient.immutable, rhs: lhs.values.pointee, add: rhsGradient.immutable, result: rhsGradient, count: vector.count)
            }
        } else if rhs.dim == 0 {
            if let lhsGradient = lhs.gradient {
                Element.vsMulVAdd(lhs: vectorGradient.immutable, rhs: rhs.values.pointee, add: lhsGradient.immutable, result: lhsGradient, count: vector.count)
            }
            if let rhsGradient = rhs.gradient {
                rhsGradient.pointee = rhsGradient.pointee + Element.dot(lhs: vectorGradient.immutable, rhs: lhs.values.immutable, count: vector.count)
            }
        } else if lhs.dim < rhs.dim {
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    Element.vMA(lhs: rhs.values.advanced(by: offset).immutable, rhs: vectorGradient.advanced(by: offset).immutable, add: lhsGradient.immutable, result: lhsGradient, count: lhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    Element.vMA(lhs: lhs.values.immutable, rhs: vectorGradient.advanced(by: offset).immutable, add: rhsGradient.advanced(by: offset).immutable, result: rhsGradient.advanced(by: offset), count: lhs.count)
                }
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                if let lhsGradient = lhs.gradient {
                    Element.vMA(lhs: rhs.values.immutable, rhs: vectorGradient.advanced(by: offset).immutable, add: lhsGradient.advanced(by: offset).immutable, result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                if let rhsGradient = rhs.gradient {
                    Element.vMA(lhs: lhs.values.advanced(by: offset).immutable, rhs: vectorGradient.advanced(by: offset).immutable, add: rhsGradient.immutable, result: rhsGradient, count: rhs.count)
                }
            }
        }
    }
    
    var symbol: String {
        return "×"
    }
}

private struct DivisionOperation<Element: NumericType>: BinaryTensorOperation {
    var lhs: Tensor<Element>
    var rhs: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        if lhs.dim == 0 {
            let temp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                Element.vDiv(lhs: vectorGradient.immutable, rhs: rhs.values.immutable, result: temp, count: rhs.count)
                let sum = Element.sum(val: temp.immutable, count: rhs.count)
                lhsGradient.pointee = lhsGradient.pointee + sum
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                Element.vSquare(values: rhs.values.immutable, result: temp, count: rhs.count)
                Element.vNeg(val: temp.immutable, result: temp, count: rhs.count)
                Element.svDiv(lhs: lhs.values.pointee, rhs: temp.immutable, result: temp, count: rhs.count)
                Element.vMA(lhs: temp.immutable, rhs: vectorGradient.immutable, add: rhsGradient.immutable, result: rhsGradient, count: rhs.count)
            }
            
            CPUAllocator.free(temp)
            
        } else if rhs.dim == 0 {
            let temp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
            
            if let lhsGradient = lhs.gradient {
                // lhs.gradient += vectorGradient / rhs.values
                Element.vsMul(lhs: vectorGradient.immutable, rhs: rhs.values.pointee, result: temp, count: vector.count)
                Element.vAdd(lhs: temp.immutable, rhs: lhsGradient.immutable, result: lhsGradient, count: lhs.count)
            }
            if let rhsGradient = rhs.gradient {
                // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vectorGradient
                let rhsVal = rhs.values.pointee
                let invRhsSq = 1 / (-rhsVal * rhsVal)
                Element.vsMul(lhs: lhs.values.immutable, rhs: invRhsSq, result: temp, count: lhs.count)
                Element.vMul(lhs: temp.immutable, rhs: vectorGradient.immutable, result: temp, count: lhs.count)
                
                rhsGradient.pointee = rhsGradient.pointee + Element.sum(val: temp.immutable, count: lhs.count)
            }
            
            CPUAllocator.free(temp)
            
        } else if lhs.dim < rhs.dim {
            let tempA: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: rhs.count)
            let tempB: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: lhs.count)
            
            if rhs.requiresGradient {
                Element.vSquare(values: rhs.values.immutable, result: tempA, count: rhs.count)
                Element.vNeg(val: tempA.immutable, result: tempA, count: rhs.count)
                Element.vDiv(lhs: vectorGradient.immutable, rhs: tempA.immutable, result: tempA, count: rhs.count)
            }
            
            
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                
                // dvector/drhs = -lhs/rhs^2 * dvector = lhs * tempA
                // dvector/dlhs = 1/rhs * dvector
                
                if rhs.requiresGradient {
                    Element.vMul(lhs: lhs.values.immutable, rhs: tempA.advanced(by: offset).immutable, result: tempA.advanced(by: offset), count: lhs.count)
                }
                
                if let lhsGradient = lhs.gradient {
                    Element.vDiv(lhs: vectorGradient.advanced(by: offset).immutable, rhs: rhs.values.advanced(by: offset).immutable, result: tempB, count: lhs.count)
                    Element.vAdd(lhs: lhsGradient.immutable, rhs: tempB.immutable, result: lhsGradient, count: lhs.count)
                }
                
            }
            
            if let rhsGradient = rhs.gradient {
                Element.vAdd(lhs: rhsGradient.immutable, rhs: tempA.immutable, result: rhsGradient, count: rhs.count)
            }
            CPUAllocator.free(tempA)
            CPUAllocator.free(tempB)
            
        } else /*if rhs.dim <= lhs.dim*/ {
            let lhsTimesGrad: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: lhs.count)
            let negInvRhsSq: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: rhs.count)
            let invRhs: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: rhs.count)
            
            if rhs.requiresGradient {
                Element.vSquare(values: rhs.values.immutable, result: negInvRhsSq, count: rhs.count)
                Element.svDiv(lhs: -1, rhs: negInvRhsSq.immutable, result: negInvRhsSq, count: rhs.count)
                
                Element.vMul(lhs: lhs.values.immutable, rhs: vectorGradient.immutable, result: lhsTimesGrad, count: lhs.count)
            }
            
            if lhs.requiresGradient {
                Element.svDiv(lhs: 1, rhs: rhs.values.immutable, result: invRhs, count: rhs.count)
            }
            
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                
                // dvector/dlhs = dvector / rhs
                // dvector/drhs = (lhs * dvector) (/ -rhs^2)
                
                if let lhsGradient = lhs.gradient {
                    Element.vMA(lhs: vectorGradient.advanced(by: offset).immutable, rhs: invRhs.immutable, add: lhsGradient.advanced(by: offset).immutable, result: lhsGradient.advanced(by: offset), count: rhs.count)
                }
                
                if let rhsGradient = rhs.gradient {
                    Element.vMA(lhs: lhsTimesGrad.advanced(by: offset).immutable, rhs: negInvRhsSq.immutable, add: rhsGradient.immutable, result: rhsGradient, count: rhs.count)
                }
            }
            
            CPUAllocator.free(lhsTimesGrad)
            CPUAllocator.free(negInvRhsSq)
            CPUAllocator.free(invRhs)
        }
    }
    
    var symbol: String {
        return "÷"
    }
}

public extension Tensor {
    static func + (lhs: Tensor<Element>, rhs: Tensor<Element>) -> Tensor<Element> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element>(
            shape: resultShape,
            parent: nil,
            context: AdditionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Element.vsAdd(lhs: rhs.values.immutable, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Element.vsAdd(lhs: lhs.values.immutable, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Element.vAdd(lhs: lhs.values.immutable, rhs: rhs.values.advanced(by: offset).immutable, result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Element.vAdd(lhs: lhs.values.advanced(by: offset).immutable, rhs: rhs.values.immutable, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func - (lhs: Tensor<Element>, rhs: Tensor<Element>) -> Tensor<Element> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element>(
            shape: resultShape,
            parent: nil,
            context: SubtractionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Element.vsAdd(lhs: rhs.values.immutable, rhs: -lhs.values.pointee, result: result.values, count: result.count)
            Element.vNeg(val: result.values.immutable, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Element.vsAdd(lhs: lhs.values.immutable, rhs: -rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Element.vSub(lhs: lhs.values.immutable, rhs: rhs.values.advanced(by: offset).immutable, result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Element.vSub(lhs: lhs.values.advanced(by: offset).immutable, rhs: rhs.values.immutable, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func * (lhs: Tensor<Element>, rhs: Tensor<Element>) -> Tensor<Element> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element>(
            shape: resultShape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Element.vsMul(lhs: rhs.values.immutable, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Element.vsMul(lhs: lhs.values.immutable, rhs: rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Element.vMul(lhs: lhs.values.immutable, rhs: rhs.values.advanced(by: offset).immutable, result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Element.vMul(lhs: lhs.values.advanced(by: offset).immutable, rhs: rhs.values.immutable, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func / (lhs: Tensor<Element>, rhs: Tensor<Element>) -> Tensor<Element> {
        precondition(
            (lhs.dim >= rhs.dim && Array(lhs.shape.dropFirst(lhs.dim - rhs.dim)) == rhs.shape) ||
                (rhs.dim > lhs.dim && Array(rhs.shape.dropFirst(rhs.dim - lhs.dim)) == lhs.shape),
            "Suffix of shape of one operand must match shape of other operand"
        )
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Tensor<Element>(
            shape: resultShape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == 0 {
            Element.svDiv(lhs: lhs.values.pointee, rhs: rhs.values.immutable, result: result.values, count: result.count)
        } else if rhs.dim == 0 {
            Element.vsMul(lhs: lhs.values.immutable, rhs: 1 / rhs.values.pointee, result: result.values, count: result.count)
        } else if lhs.dim < rhs.dim {
            assert(rhs.shape.suffix(lhs.dim) == lhs.shape)
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Element.vDiv(lhs: lhs.values.immutable, rhs: rhs.values.advanced(by: offset).immutable, result: result.values.advanced(by: offset), count: lhs.count)
            }
        } else /*if lhs.dim > rhs.dim*/ {
            assert(lhs.shape.suffix(rhs.dim) == rhs.shape)
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
                Element.vDiv(lhs: lhs.values.advanced(by: offset).immutable, rhs: rhs.values.immutable, result: result.values.advanced(by: offset), count: rhs.count)
            }
        }
        
        return result
    }
    
    static func += (lhs: inout Tensor<Element>, rhs: Tensor<Element>) {
        lhs = lhs + rhs
    }
    
    static func -= (lhs: inout Tensor<Element>, rhs: Tensor<Element>) {
        lhs = lhs - rhs
    }
}

public prefix func - <Element>(value: Tensor<Element>) -> Tensor<Element> {
    return 0 - value
}

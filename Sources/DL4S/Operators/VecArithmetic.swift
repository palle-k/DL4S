//
//  VecArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct AdditionOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
//        if lhs.dim == rhs.dim {
//            Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.immutable, result: lhs.gradient, count: lhs.count)
//            Element.vAdd(lhs: rhs.gradient.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, count: rhs.count)
//        } else {
//            let sum = Element.sum(val: vector.gradient.immutable, count: vector.count)
//
//            if lhs.dim == 0 {
//                lhs.gradient.pointee = lhs.gradient.pointee + sum
//                Element.vAdd(lhs: rhs.gradient.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, count: rhs.count)
//            } else {
//                Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.immutable, result: lhs.gradient, count: lhs.count)
//                rhs.gradient.pointee = rhs.gradient.pointee + sum
//            }
//        }
        
        if lhs.dim == 0 {
            let sum = Element.sum(val: vector.gradient.immutable, count: vector.count)
            lhs.gradient.pointee = lhs.gradient.pointee + sum
            Element.vAdd(lhs: rhs.gradient.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, count: rhs.count)
        } else if rhs.dim == 0 {
            let sum = Element.sum(val: vector.gradient.immutable, count: vector.count)
            Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.immutable, result: lhs.gradient, count: lhs.count)
            rhs.gradient.pointee = rhs.gradient.pointee + sum
        } else if lhs.dim < rhs.dim {
            let shapePrefix = rhs.shape.prefix(rhs.dim - lhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, rhs.strides.prefix(rhs.dim - lhs.dim)).map(*).reduce(0, +)
                Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.advanced(by: offset), result: <#T##UnsafeMutableBufferPointer<NumericType>#>, count: <#T##Int#>)
            }
        } else /*if rhs.dim <= lhs.dim*/ {
            let shapePrefix = lhs.shape.prefix(lhs.dim - rhs.dim)
            for idx in iterate(Array(shapePrefix)) {
                let offset = zip(idx, lhs.strides.prefix(lhs.dim - rhs.dim)).map(*).reduce(0, +)
            }
        }
        
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct SubtractionOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        if lhs.dim == rhs.dim {
            Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.immutable, result: lhs.gradient, count: lhs.count)
            Element.vSub(lhs: rhs.gradient.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, count: rhs.count)
        } else {
            let sum = Element.sum(val: vector.gradient.immutable, count: vector.count)
            
            if lhs.dim == 0 {
                lhs.gradient.pointee = lhs.gradient.pointee + sum
                Element.vSub(lhs: rhs.gradient.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, count: rhs.count)
            } else {
                Element.vAdd(lhs: lhs.gradient.immutable, rhs: vector.gradient.immutable, result: lhs.gradient, count: lhs.count)
                rhs.gradient.pointee = rhs.gradient.pointee - sum
            }
        }
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct MultiplicationOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        if lhs.dim == rhs.dim {
            Element.vMA(lhs: rhs.values.immutable, rhs: vector.gradient.immutable, add: lhs.gradient, result: lhs.gradient, count: lhs.count)
            Element.vMA(lhs: lhs.values.immutable, rhs: vector.gradient.immutable, add: rhs.gradient, result: rhs.gradient, count: rhs.count)
        } else if lhs.dim == 0 {
            lhs.gradient.pointee = lhs.gradient.pointee + Element.dot(lhs: vector.gradient.immutable, rhs: rhs.values.immutable, count: vector.count)
            Element.vsMulVAdd(lhs: vector.gradient.immutable, rhs: lhs.values.pointee, add: rhs.gradient.immutable, result: rhs.gradient, count: vector.count)
        } else {
            Element.vsMulVAdd(lhs: vector.gradient.immutable, rhs: rhs.values.pointee, add: lhs.gradient.immutable, result: lhs.gradient, count: vector.count)
            rhs.gradient.pointee = rhs.gradient.pointee + Element.dot(lhs: vector.gradient.immutable, rhs: lhs.values.immutable, count: vector.count)
        }
    
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct DivisionOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        
        if lhs.dim == rhs.dim {

            // lhs.gradient += vector.gradient / rhs.values
            Element.vDiv(lhs: vector.gradient.immutable, rhs: rhs.values.immutable, result: temp, count: rhs.count)
            Element.vAdd(lhs: temp.immutable, rhs: lhs.gradient.immutable, result: lhs.gradient, count: lhs.count)
            
            // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vector.gradient
            Element.vSquare(values: rhs.values.immutable, result: temp, count: rhs.count)
            Element.vNeg(val: temp.immutable, result: temp, count: rhs.count)
            Element.vDiv(lhs: lhs.values.immutable, rhs: temp.immutable, result: temp, count: rhs.count)
            Element.vMA(lhs: temp.immutable, rhs: vector.gradient.immutable, add: rhs.gradient, result: rhs.gradient, count: rhs.count)
            
        } else if lhs.dim == 0 {
            // lhs.gradient += vector.gradient / rhs.values
            Element.vDiv(lhs: vector.gradient.immutable, rhs: rhs.values.immutable, result: temp, count: rhs.count)
            let sum = Element.sum(val: temp.immutable, count: rhs.count)
            lhs.gradient.pointee = lhs.gradient.pointee + sum
            
            // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vector.gradient
            Element.vSquare(values: rhs.values.immutable, result: temp, count: rhs.count)
            Element.vNeg(val: temp.immutable, result: temp, count: rhs.count)
            Element.svDiv(lhs: lhs.values.pointee, rhs: temp.immutable, result: temp, count: rhs.count)
            Element.vMA(lhs: temp.immutable, rhs: vector.gradient.immutable, add: rhs.gradient, result: rhs.gradient, count: rhs.count)
        } else {
            // lhs.gradient += vector.gradient / rhs.values
            Element.vsMul(lhs: vector.gradient.immutable, rhs: rhs.values.pointee, result: temp, count: vector.count)
            Element.vAdd(lhs: temp.immutable, rhs: lhs.gradient.immutable, result: lhs.gradient, count: lhs.count)
            
            // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vector.gradient
            let rhsVal = rhs.values.pointee
            let invRhsSq = 1 / (-rhsVal * rhsVal)
            Element.vsMul(lhs: lhs.values.immutable, rhs: invRhsSq, result: temp, count: lhs.count)
            Element.vMul(lhs: temp.immutable, rhs: vector.gradient.immutable, result: temp, count: lhs.count)
            
            rhs.gradient.pointee = rhs.gradient.pointee + Element.sum(val: temp.immutable, count: lhs.count)
        }
        
        Allocator.free(temp)
        
        lhs._backwards()
        rhs._backwards()
    }
}

public extension Vector {
    static func + (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(
            (lhs.dim >= rhs.dim && lhs.shape.suffix(lhs.dim - rhs.dim) == rhs.shape) ||
            (rhs.dim > lhs.dim && rhs.shape.suffix(rhs.dim - lhs.dim) == lhs.shape),
            "Vectors must have same shape for addition.")
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Vector<Element>(
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
    
    static func - (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape || lhs.shape == [] || rhs.shape == [], "Vectors must have same shape for subtraction.")
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Vector<Element>(
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
    
    static func * (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape || lhs.shape == [] || rhs.shape == [], "Vectors must have same shape for multiplication.")
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Vector<Element>(
            shape: resultShape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == rhs.dim {
            Element.vMul(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values, count: result.count)
        } else if lhs.dim == 0 {
            Element.vsMul(lhs: rhs.values.immutable, rhs: lhs.values.pointee, result: result.values, count: result.count)
        } else {
            Element.vsMul(lhs: lhs.values.immutable, rhs: rhs.values.pointee, result: result.values, count: result.count)
        }
        
        
        return result
    }
    
    static func / (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape || lhs.shape == [] || rhs.shape == [], "Vectors must have same shape for division.")
        
        let resultShape = lhs.dim >= rhs.dim ? lhs.shape : rhs.shape
        
        let result = Vector<Element>(
            shape: resultShape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        if lhs.dim == rhs.dim {
            Element.vDiv(lhs: lhs.values.immutable, rhs: rhs.values.immutable, result: result.values, count: result.count)
        } else if lhs.dim == 0 {
            Element.svDiv(lhs: lhs.values.pointee, rhs: rhs.values.immutable, result: result.values, count: result.count)
        } else {
            Element.vsMul(lhs: lhs.values.immutable, rhs: 1 / rhs.values.pointee, result: result.values, count: result.count)
        }
        
        return result
    }
}

public prefix func - <Element>(value: Vector<Element>) -> Vector<Element> {
    return 0 - value
}

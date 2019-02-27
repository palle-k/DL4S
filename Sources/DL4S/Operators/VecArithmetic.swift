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
        Element.vAdd(lhs: lhs.gradient, rhs: vector.gradient, result: lhs.gradient, count: lhs.count)
        Element.vAdd(lhs: rhs.gradient, rhs: vector.gradient, result: rhs.gradient, count: rhs.count)
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct SubtractionOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        Element.vAdd(lhs: lhs.gradient, rhs: vector.gradient, result: lhs.gradient, count: lhs.count)
        Element.vSub(lhs: rhs.gradient, rhs: vector.gradient, result: rhs.gradient, count: rhs.count)
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct MultiplicationOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        Element.vMA(lhs: rhs.values, rhs: vector.gradient, add: lhs.gradient, result: lhs.gradient, count: lhs.count)
        Element.vMA(lhs: lhs.values, rhs: vector.gradient, add: rhs.gradient, result: rhs.gradient, count: rhs.count)
        
        lhs._backwards()
        rhs._backwards()
    }
}

private struct DivisionOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutablePointer<Element> = Allocator.allocate(count: vector.count)
        
        // lhs.gradient += rhs.values * vector.gradient
        Element.svDiv(lhs: 1, rhs: rhs.values, result: temp, count: rhs.count)
        Element.vMA(lhs: temp, rhs: vector.gradient, add: lhs.gradient, result: lhs.gradient, count: lhs.count)
        
        // rhs.gradient += lhs.values / ((-1) * rhs.values ** 2) * vector.gradient
        Element.vSquare(values: rhs.values, result: temp, count: rhs.count)
        Element.vNeg(val: temp, result: temp, count: rhs.count)
        Element.vDiv(lhs: lhs.values, rhs: temp, result: temp, count: rhs.count)
        Element.vMA(lhs: temp, rhs: vector.gradient, add: rhs.gradient, result: rhs.gradient, count: rhs.count)
        
        Allocator.free(temp)
        
        lhs._backwards()
        rhs._backwards()
    }
}



public extension Vector {
    static func + (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape, "Vectors must have same shape for addition.")
        
        let result = Vector<Element>(
            shape: lhs.shape,
            parent: nil,
            context: AdditionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Element.vAdd(lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count)
        
        return result
    }
    
    static func - (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape, "Vectors must have same shape for addition.")
        
        let result = Vector<Element>(
            shape: lhs.shape,
            parent: nil,
            context: SubtractionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Element.vSub(lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count)
        
        return result
    }
    
    static func * (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape, "Vectors must have same shape for addition.")
        
        let result = Vector<Element>(
            shape: lhs.shape,
            parent: nil,
            context: MultiplicationOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Element.vMul(lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count)
        
        return result
    }
    
    static func / (lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
        precondition(lhs.shape == rhs.shape, "Vectors must have same shape for addition.")
        
        let result = Vector<Element>(
            shape: lhs.shape,
            parent: nil,
            context: DivisionOperation(lhs: lhs, rhs: rhs).asAny()
        )
        
        Element.vDiv(lhs: lhs.values, rhs: rhs.values, result: result.values, count: result.count)
        
        return result
    }
}

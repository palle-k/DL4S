//
//  ReductionArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct SumContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let fac = vector.item
        Element.vsAdd(lhs: source.gradient.immutable, rhs: fac, result: source.gradient, count: source.count)
        
        source._backwards()
    }
}


public func sum<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: [],
        parent: nil,
        context: SumContext(source: vector).asAny()
    )
    
    result.values[0] = Element.sum(val: vector.values.immutable, count: vector.count)
    
    return result
}

private struct MaxContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    let maxI: Int
    
    func backwards(from vector: Vector<Element>) {
        source.gradient[maxI] = source.gradient[maxI] + vector.gradientItem
    }
}

public func max<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let (arg, max) = Element.argmax(values: vector.values.immutable, count: vector.count)
    
    let result = Vector(max)
    result.context = MaxContext(source: vector, maxI: arg).asAny()
    return result
}

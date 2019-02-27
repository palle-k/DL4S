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
        Element.vsAdd(lhs: source.gradient, rhs: fac, result: source.gradient, count: source.count)
    }
}


public func sum<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: [],
        parent: nil,
        context: SumContext(source: vector).asAny()
    )
    
    result.values[0] = Element.sum(val: vector.values, count: vector.count)
    
    return result
}

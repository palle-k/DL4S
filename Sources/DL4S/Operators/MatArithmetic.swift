//
//  MatArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct MatmulOperation<Element: NumericType>: BinaryVectorOperation {
    var lhs: Vector<Element>
    var rhs: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        
    }
}


public func mmul<Element: NumericType>(_ lhs: Vector<Element>, rhs: Vector<Element>) -> Vector<Element> {
    precondition(lhs.dim == 2 && rhs.dim == 2, "Matrix multiplication operands must both be two-dimensional.")
    precondition(lhs.shape[1] == rhs.shape[0], "Matrix multiplication operands must have matching shapes (lhs.shape[1] == rhs.shape[0])")
    
    let result = Vector<Element>(
        shape: [lhs.shape[0], rhs.shape[1]],
        parent: nil,
        context: MatmulOperation(lhs: lhs, rhs: rhs).asAny()
    )
    
    Element.matMul(lhs: lhs.values, rhs: rhs.values, result: result.values, lhsRows: lhs.shape[0], lhsCols: lhs.shape[1], rhsCols: rhs.shape[1])
    
    return result
}

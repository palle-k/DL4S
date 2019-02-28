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
        let temp1: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: lhs.count)
        let temp2: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: rhs.count)
        
        let temp3: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: lhs.count)
        let temp4: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: rhs.count)
        
        // Element.matMulAddInPlace(
        //     lhs: lhs.values.immutable,
        //     rhs: vector.gradient.immutable,
        //     result: rhs.gradient,
        //     lhsShape: (lhs.shape[0], lhs.shape[1]),
        //     rhsShape: (vector.shape[0], vector.shape[1]),
        //     resultShape: (rhs.shape[0], rhs.shape[1]),
        //     transposeFirst: true,
        //     transposeSecond: false
        // )
        // Element.matMulAddInPlace(
        //     lhs: vector.values.immutable,
        //     rhs: rhs.values.immutable,
        //     result: lhs.gradient,
        //     lhsShape: (vector.shape[0], vector.shape[1]),
        //     rhsShape: (rhs.shape[0], rhs.shape[1]),
        //     resultShape: (lhs.shape[0], lhs.shape[1]),
        //     transposeFirst: false,
        //     transposeSecond: true
        // )
        
        Element.transpose(val: lhs.values.immutable, result: temp1, srcRows: lhs.shape[0], srcCols: lhs.shape[1])
        Element.transpose(val: rhs.values.immutable, result: temp2, srcRows: rhs.shape[0], srcCols: rhs.shape[1])
        
        //Element.matMulAddInPlace(lhs: temp1.immutable, rhs: vector.gradient.immutable, result: rhs.gradient, lhsShape: (lhs.shape[1], lhs.shape[0]), rhsShape: (vector.shape[0], vector.shape[1]), resultShape: (rhs.shape[0], rhs.shape[1]), transposeFirst: false, transposeSecond: false)
        //Element.matMulAddInPlace(lhs: vector.gradient.immutable, rhs: temp2.immutable, result: lhs.gradient, lhsShape: (vector.shape[0], vector.shape[1]), rhsShape: (rhs.shape[1], rhs.shape[0]), resultShape: (lhs.shape[0], lhs.shape[1]), transposeFirst: false, transposeSecond: false)
        
        Element.matMul(lhs: temp1.immutable, rhs: vector.gradient.immutable, result: temp4, lhsRows: lhs.shape[1], lhsCols: lhs.shape[0], rhsCols: vector.shape[1])
        Element.matMul(lhs: vector.gradient.immutable, rhs: temp2.immutable, result: temp3, lhsRows: vector.shape[0], lhsCols: vector.shape[1], rhsCols: rhs.shape[0])
        
        Element.vAdd(lhs: temp3.immutable, rhs: lhs.gradient.immutable, result: lhs.gradient, count: lhs.count)
        Element.vAdd(lhs: temp4.immutable, rhs: rhs.gradient.immutable, result: rhs.gradient, count: rhs.count)
        
        Allocator.free(temp1)
        Allocator.free(temp2)
        Allocator.free(temp3)
        Allocator.free(temp4)
        
        lhs._backwards()
        rhs._backwards()
    }
}


public func mmul<Element: NumericType>(_ lhs: Vector<Element>, _ rhs: Vector<Element>) -> Vector<Element> {
    precondition(1 ... 2 ~= lhs.dim && 1 ... 2 ~= rhs.dim, "Matrix multiplication operands must both be one or two dimensional.")
    // lhs.dim == 2 and rhs.dim == 2 implies matching shapes
    precondition(!(lhs.dim == 2 && rhs.dim == 2) || lhs.shape[1] == rhs.shape[0], "Matrix multiplication operands must have matching shapes.")
    
    let resultShape: [Int]
    let resultViewShape: [Int]
    
    let lhsView: Vector<Element>
    let rhsView: Vector<Element>
    
    switch (lhs.dim, rhs.dim) {
    case (1, 1):
        resultShape = [1, 1]
        resultViewShape = []
        lhsView = lhs.view(as: 1, -1)
        rhsView = rhs.view(as: -1, 1)
    case (1, 2):
        lhsView = lhs.view(as: 1, -1)
        rhsView = rhs
        resultShape = [1, rhs.shape[1]]
        resultViewShape = [rhs.shape[1]]
    case (2, 1):
        lhsView = lhs
        rhsView = rhs.view(as: -1, 1)
        resultShape = [lhs.shape[0], 1]
        resultViewShape = [lhs.shape[0]]
    case (_, _):
        lhsView = lhs
        rhsView = rhs
        resultShape = [lhs.shape[0], rhs.shape[1]]
        resultViewShape = [lhs.shape[0], rhs.shape[1]]
    }
    
    let result = Vector<Element>(
        shape: resultShape,
        parent: nil,
        context: MatmulOperation(lhs: lhsView, rhs: rhsView).asAny()
    )
    
    Element.matMul(lhs: lhsView.values.immutable, rhs: rhsView.values.immutable, result: result.values, lhsRows: lhsView.shape[0], lhsCols: lhsView.shape[1], rhsCols: rhsView.shape[1])
    
    return result.view(as: resultViewShape)
}

struct TransposeOperation<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        Element.transpose(val: vector.gradient.immutable, result: temp, srcRows: vector.shape[0], srcCols: vector.shape[1])
        Element.vAdd(lhs: source.gradient.immutable, rhs: temp.immutable, result: source.gradient, count: vector.count)
        Allocator.free(temp)
        
        source._backwards()
    }
}

public extension Vector {
    var T: Vector<Element> {
        precondition(dim <= 2, "Dimensionality for vector transpose must be smaller than or equal to 2")
        
        if dim <= 1 {
            return self
        } else {
            let result = Vector<Element>(
                shape: [self.shape[1], self.shape[0]],
                parent: nil,
                context: TransposeOperation(source: self).asAny()
            )
            
            Element.transpose(val: values.immutable, result: result.values, srcRows: shape[0], srcCols: shape[1])
            
            return result
        }
    }
}

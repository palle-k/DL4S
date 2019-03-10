//
//  MatArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct MatmulOperation<Element: NumericType>: BinaryTensorOperation {
    var lhs: Tensor<Element>
    var rhs: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        
        if let lhsGradient = lhs.gradient {
            let temp2: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: rhs.count)
            let temp3: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: lhs.count)
            
            Element.transpose(val: rhs.values.immutable, result: temp2, srcRows: rhs.shape[0], srcCols: rhs.shape[1])
            Element.matMul(lhs: vectorGradient.immutable, rhs: temp2.immutable, result: temp3, lhsRows: vector.shape[0], lhsCols: vector.shape[1], rhsCols: rhs.shape[0])
            Element.vAdd(lhs: temp3.immutable, rhs: lhsGradient.immutable, result: lhsGradient, count: lhs.count)
            
            Allocator.free(temp2)
            Allocator.free(temp3)
        }
        if let rhsGradient = rhs.gradient {
            let temp1: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: lhs.count)
            let temp4: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: rhs.count)
            
            Element.transpose(val: lhs.values.immutable, result: temp1, srcRows: lhs.shape[0], srcCols: lhs.shape[1])
            Element.matMul(lhs: temp1.immutable, rhs: vectorGradient.immutable, result: temp4, lhsRows: lhs.shape[1], lhsCols: lhs.shape[0], rhsCols: vector.shape[1])
            Element.vAdd(lhs: temp4.immutable, rhs: rhsGradient.immutable, result: rhsGradient, count: rhs.count)
            
            Allocator.free(temp1)
            Allocator.free(temp4)
        }
        
        backpropagate()
    }
    
    var symbol: String {
        return "matmul"
    }
}


public func mmul<Element: NumericType>(_ lhs: Tensor<Element>, _ rhs: Tensor<Element>) -> Tensor<Element> {
    precondition(1 ... 2 ~= lhs.dim && 1 ... 2 ~= rhs.dim, "Matrix multiplication operands must both be one or two dimensional.")
    // lhs.dim == 2 and rhs.dim == 2 implies matching shapes
    precondition(!(lhs.dim == 2 && rhs.dim == 2) || lhs.shape[1] == rhs.shape[0], "Matrix multiplication operands must have matching shapes.")
    
    let resultShape: [Int]
    let resultViewShape: [Int]
    
    let lhsView: Tensor<Element>
    let rhsView: Tensor<Element>
    
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
    
    let result = Tensor<Element>(
        shape: resultShape,
        parent: nil,
        context: lhs.requiresGradient || rhs.requiresGradient ? MatmulOperation(lhs: lhsView, rhs: rhsView).asAny() : nil
    )
    
    Element.matMul(lhs: lhsView.values.immutable, rhs: rhsView.values.immutable, result: result.values, lhsRows: lhsView.shape[0], lhsCols: lhsView.shape[1], rhsCols: rhsView.shape[1])
    
    return result.view(as: resultViewShape)
}

struct TransposeOperation<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient, let sourceGradient = source.gradient else {
            return
        }
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        Element.transpose(val: vectorGradient.immutable, result: temp, srcRows: vector.shape[0], srcCols: vector.shape[1])
        Element.vAdd(lhs: sourceGradient.immutable, rhs: temp.immutable, result: sourceGradient, count: vector.count)
        Allocator.free(temp)
        
        backpropagate()
    }
    
    var symbol: String {
        return "transpose"
    }
}

public extension Tensor {
    var T: Tensor<Element> {
        precondition(dim <= 2, "Dimensionality for vector transpose must be smaller than or equal to 2")
        
        if dim <= 1 {
            return self
        } else {
            let result = Tensor<Element>(
                shape: [self.shape[1], self.shape[0]],
                parent: nil,
                context: requiresGradient ? TransposeOperation(source: self).asAny() : nil
            )
            
            Element.transpose(val: values.immutable, result: result.values, srcRows: shape[0], srcCols: shape[1])
            
            return result
        }
    }
}

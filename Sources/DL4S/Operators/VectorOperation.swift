//
//  VectorOperation.swift
//  DL4S
//
//  Created by Palle Klewitz on 26.02.19.
//

import Foundation


protocol VectorOperation {
    associatedtype Element: NumericType
    
    var sourceVectors: [Vector<Element>] { get }
    func backwards(from vector: Vector<Element>)
    func zeroGradient()
}

extension VectorOperation {
    func zeroGradient() {
        sourceVectors.forEach { v in
            v.zeroGradient()
        }
    }
    
    func asAny() -> AnyVectorOperation<Element> {
        return AnyVectorOperation(operation: self)
    }
}

struct AnyVectorOperation<Element: NumericType>: VectorOperation {
    var sourceVectors: [Vector<Element>] {
        return _getSource()
    }
    
    private let _getSource: () -> [Vector<Element>]
    private let _backwards: (Vector<Element>) -> ()
    private let _zeroGrad: () -> ()
    
    init<Op: VectorOperation>(operation: Op) where Op.Element == Element {
        self._getSource = {operation.sourceVectors}
        self._backwards = operation.backwards
        self._zeroGrad = operation.zeroGradient
    }
    
    func backwards(from vector: Vector<Element>) {
        _backwards(vector)
    }
    
    func zeroGradient() {
        _zeroGrad()
    }
}

protocol UnaryVectorOperation: VectorOperation {
    var source: Vector<Element> { get }
}

extension UnaryVectorOperation {
    var sourceVectors: [Vector<Element>] {
        return [source]
    }
}

protocol BinaryVectorOperation: VectorOperation {
    var lhs: Vector<Element> { get }
    var rhs: Vector<Element> { get }
}

extension BinaryVectorOperation {
    var sourceVectors: [Vector<Element>] {
        return [lhs, rhs]
    }
}

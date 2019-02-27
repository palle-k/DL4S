//
//  UnaryOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct ExpContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutablePointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.exp(val: source.values, result: temp, count: source.count)
        Element.vMA(lhs: temp, rhs: vector.gradient, add: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}

private struct LogContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutablePointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.vDiv(lhs: vector.gradient, rhs: source.values, result: temp, count: source.count)
        Element.vAdd(lhs: temp, rhs: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}

private struct TanhContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutablePointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.tanh(val: source.values, result: temp, count: source.count)
        Element.vSquare(values: temp, result: temp, count: source.count)
        Element.vNeg(val: temp, result: temp, count: source.count)
        Element.vsAdd(lhs: temp, rhs: 1, result: temp, count: source.count)
        Element.vMA(lhs: temp, rhs: vector.gradient, add: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}


public func exp<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: ExpContext(source: vector).asAny()
    )
    
    Element.exp(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func log<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: LogContext(source: vector).asAny()
    )
    
    Element.log(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func tanh<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: TanhContext(source: vector).asAny()
    )
    
    Element.tanh(val: vector.values, result: result.values, count: result.count)
    
    return result
}



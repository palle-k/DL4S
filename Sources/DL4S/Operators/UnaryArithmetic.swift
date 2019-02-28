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
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.exp(val: source.values.immutable, result: temp, count: source.count)
        Element.vMA(lhs: temp.immutable, rhs: vector.gradient.immutable, add: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}

private struct LogContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.vDiv(lhs: vector.gradient.immutable, rhs: source.values.immutable, result: temp, count: source.count)
        Element.vAdd(lhs: temp.immutable, rhs: source.gradient.immutable, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}

private struct TanhContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.tanh(val: source.values.immutable, result: temp, count: source.count)
        Element.vSquare(values: temp.immutable, result: temp, count: source.count)
        Element.vNeg(val: temp.immutable, result: temp, count: source.count)
        Element.vsAdd(lhs: temp.immutable, rhs: 1, result: temp, count: source.count)
        Element.vMA(lhs: temp.immutable, rhs: vector.gradient.immutable, add: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp)
        
        source._backwards()
    }
}

private struct ReluContext<Element: NumericType>: UnaryVectorOperation {
    var source: Vector<Element>
    
    func backwards(from vector: Vector<Element>) {
        let temp1: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        let temp2: UnsafeMutableBufferPointer<Element> = Allocator.allocate(count: vector.count)
        
        Element.fill(value: 0.5, result: temp1, count: vector.count)
        Element.fill(value: 0.5, result: temp2, count: vector.count)
        Element.copysign(values: temp1.immutable, signs: source.values.immutable, result: temp1, count: vector.count)
        
        // temp1[x] == 0 if source[x] <= 0 else temp1[x] == 1 (Relu mask)
        Element.vAdd(lhs: temp1.immutable, rhs: temp2.immutable, result: temp1, count: vector.count)
        Element.vMA(lhs: temp1.immutable, rhs: vector.gradient.immutable, add: source.gradient, result: source.gradient, count: source.count)
        
        Allocator.free(temp1)
        Allocator.free(temp2)
        
        source._backwards()
    }
}


public func exp<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: ExpContext(source: vector).asAny()
    )
    
    Element.exp(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func log<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: LogContext(source: vector).asAny()
    )
    
    Element.log(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func tanh<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: TanhContext(source: vector).asAny()
    )
    
    Element.tanh(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func relu<Element>(_ vector: Vector<Element>) -> Vector<Element> {
    let result = Vector<Element>(
        shape: vector.shape,
        parent: nil,
        context: ReluContext(source: vector).asAny()
    )
    
    Element.relu(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}


public func binaryCrossEntropy(expected: Vector<Float>, actual: Vector<Float>) -> Vector<Float> {
    let e = expected
    let a = actual.view(as: -1)
    
    let p1 = e * log(a)
    let p2 = (1 - e) * log(1 - a)
    return sum(-(p1 + p2))
}

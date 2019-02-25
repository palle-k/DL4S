//
//  Variable.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//

import Foundation

public class Variable: ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral {
    public var value: Float
    public internal(set) var gradient: Float
    var context: OperationContext?
    
    public init(value: Float) {
        self.value = value
        self.gradient = 0
        self.context = nil
    }
    
    internal init(value: Float, gradient: Float = 0, context: OperationContext) {
        self.value = value
        self.gradient = gradient
        self.context = context
    }
    
    public required init(floatLiteral value: Float) {
        self.value = value
        self.gradient = 0
        self.context = nil
    }
    
    public required init(integerLiteral value: Int) {
        self.value = Float(value)
        self.gradient = 0
        self.context = nil
    }
    
    public func zeroGradient() {
        gradient = 0
        context?.zeroGradient()
    }
    
    func _backwards() {
        context?.backwards(from: self)
    }
    
    public func backwards() {
        gradient = 1
        _backwards()
    }
}

extension Array where Element == Variable {
    public init(repeating: Float, count: Int) {
        self = (0 ..< count).map{_ in Variable(value: repeating)}
    }
    
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) {
        for i in 0 ..< count {
            self[i].value = Float.random(in: range)
        }
    }
}

extension Array where Element == [Variable] {
    public init(repeating: Float, rows: Int, columns: Int) {
        self = (0 ..< rows).map { _ in
            [Variable](repeating: repeating, count: columns)
        }
    }
    
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) {
        for i in 0 ..< count {
            self[i].fillRandomly(range)
        }
    }
}

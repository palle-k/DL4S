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

extension Variable: Hashable {
    public static func == (lhs: Variable, rhs: Variable) -> Bool {
        return lhs.value == rhs.value && lhs.gradient == rhs.gradient && lhs.context?.dependencies == rhs.context?.dependencies && lhs.context?.symbol == rhs.context?.symbol
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(value)
        hasher.combine(gradient)
        hasher.combine(context?.dependencies)
        hasher.combine(context?.symbol)
    }
}

extension Array where Element == Variable {
    public init(repeating: Float, count: Int) {
        self = (0 ..< count).map{_ in Variable(value: repeating)}
    }
    
    @discardableResult
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) -> [Variable] {
        for i in 0 ..< count {
            self[i].value = Float.random(in: range)
        }
        return self
    }
}

extension Array where Element == [Variable] {
    public init(repeating: Float, rows: Int, columns: Int) {
        self = (0 ..< rows).map { _ in
            [Variable](repeating: repeating, count: columns)
        }
    }
    
    @discardableResult
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) -> [[Variable]] {
        for i in 0 ..< count {
            self[i].fillRandomly(range)
        }
        return self
    }
}

extension Array where Element == [[Variable]] {
    public init(repeating: Float, depth: Int, rows: Int, columns: Int) {
        self = (0 ..< depth).map { _ in
            [[Variable]](repeating: repeating, rows: rows, columns: columns)
        }
    }
    
    @discardableResult
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) -> [[[Variable]]] {
        for i in 0 ..< count {
            self[i].fillRandomly(range)
        }
        return self
    }
}

extension Array where Element == [[[Variable]]] {
    public init(repeating: Float, count: Int, depth: Int, rows: Int, columns: Int) {
        self = (0 ..< count).map { _ in
            [[[Variable]]](repeating: repeating, depth: depth, rows: rows, columns: columns)
        }
    }
    
    @discardableResult
    public func fillRandomly(_ range: ClosedRange<Float> = 0 ... 1) -> [[[[Variable]]]] {
        for i in 0 ..< count {
            self[i].fillRandomly(range)
        }
        return self
    }
}

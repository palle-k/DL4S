//
//  Variable.swift
//  DL4S
//
//  Created by Palle Klewitz on 25.02.19.
//  Copyright (c) 2019 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

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

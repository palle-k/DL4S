//
//  VectorNumerics.swift
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

/// A type conforming to the ZeroableType protocol is required to have a zero value, which fulfills the following requirements for any other value x:
///
/// - x * .zero = .zero
/// - x + .zero = x
public protocol ZeroableType: Hashable, Codable, ExpressibleByIntegerLiteral {
    
    /// Zero value
    ///
    /// - x * .zero = .zero
    /// - x + .zero = x
    static var zero: Self { get }
}

/// A type that can be used as a number in a Tensor.
public protocol NumericType: ZeroableType, ExpressibleByFloatLiteral, CPUNumeric {
    
    /// Formats the number with the given amount of decimal places
    /// - Parameter maxDecimals: Maximum amount of decimal places
    func format(maxDecimals: Int) -> String
    
    init(_ floatValue: Double)
    init(_ integerValue: Int32)
    
    static var zero: Self { get }
    static var one: Self { get }
    
    var floatValue: Float { get }
    var doubleValue: Double { get }
    var intValue: Int32 { get }
    
    static prefix func - (value: Self) -> Self
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    
    func sqrt() -> Self
    func log() -> Self
    func exp() -> Self
    func sin() -> Self
    func cos() -> Self
    func tan() -> Self
    func sinh() -> Self
    func cosh() -> Self
    func tanh() -> Self
    
    var isFinite: Bool { get }
    var isNaN: Bool { get }
    
    init(_ float: Float)
    init(_ int: Int)
    init(_ uint: UInt)
    init(_ uint8: UInt8)
    
    func toUInt8() -> UInt8
    func toInt() -> Int
    
}

public extension UInt8 {
    init<Element: NumericType>(_ element: Element) {
        self = element.toUInt8()
    }
}

public extension Int32 {
    init<Element: NumericType>(element: Element) {
        self = element.intValue
    }
}

public extension Float {
    init<Element: NumericType>(element: Element) {
        self = element.floatValue
    }
}

public extension Double {
    init<Element: NumericType>(element: Element) {
        self = element.doubleValue
    }
}

public extension NumericType {
    @inline(__always)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    @_specialize(where Self == Int32)
    static func += (lhs: inout Self, rhs: Self) {
        lhs = lhs + rhs
    }
    
    @inline(__always)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    @_specialize(where Self == Int32)
    static func -= (lhs: inout Self, rhs: Self) {
        lhs = lhs - rhs
    }
    
    @inline(__always)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    @_specialize(where Self == Int32)
    static func *= (lhs: inout Self, rhs: Self) {
        lhs = lhs * rhs
    }
    
    @inline(__always)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    @_specialize(where Self == Int32)
    static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
    }
}

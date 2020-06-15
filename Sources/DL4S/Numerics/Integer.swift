//
//  File.swift
//  
//
//  Created by Palle Klewitz on 15.06.20.
//  Copyright (c) 2019 - 2020 - Palle Klewitz
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

public extension FixedWidthInteger where Self: NumericType {
    func format(maxDecimals: Int) -> String {
        return "\(self)"
    }
    
    var isFinite: Bool {
        return true
    }
    
    var isNaN: Bool {
        return false
    }
    
    func toUInt8() -> UInt8 {
        return UInt8(self)
    }
    
    func toInt() -> Int {
        return Int(self)
    }
    
    var floatValue: Float {
        return Float(self)
    }
    
    var doubleValue: Double {
        return Double(self)
    }
    
    var intValue: Int32 {
        return Int32(self)
    }
    
    static var one: Self {
        return 1
    }
    
    func sqrt() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func log() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func exp() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func sin() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func cos() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func tan() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func sinh() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func cosh() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    func tanh() -> Self {
        fatalError("\(#function) not available for integer types")
    }
    
    init(floatLiteral value: Double) {
        self = Self.init(value)
    }
}

public extension UnsignedInteger where Self: NumericType {
    static prefix func - (value: Self) -> Self {
        fatalError("Cannot negate unsigned integer")
    }
}

extension UInt8: NumericType {}
extension UInt16: NumericType {}
extension UInt32: NumericType {}
extension UInt64: NumericType {}

extension Int8: NumericType {}
extension Int16: NumericType {}
extension Int32: NumericType {}
extension Int64: NumericType {}

extension Int: NumericType {}
extension UInt: NumericType {}

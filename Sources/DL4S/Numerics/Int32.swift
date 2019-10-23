//
//  Int32.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
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


extension Int32: NumericType {
    public func format(maxDecimals: Int) -> String {
        return "\(self)"
    }
    
    public var isFinite: Bool {
        return true
    }
    
    public var isNaN: Bool {
        return false
    }
    
    public func toUInt8() -> UInt8 {
        return UInt8(self)
    }
    
    public func toInt() -> Int {
        return Int(self)
    }
    
    public var floatValue: Float {
        return Float(self)
    }
    
    public var doubleValue: Double {
        return Double(self)
    }
    
    public var intValue: Int32 {
        return self
    }
    
    public static var one: Int32 {
        return 1
    }
    
    public func sqrt() -> Int32 {
        return Int32(Foundation.sqrt(Float(self)))
    }
    
    public func log() -> Int32 {
        return Int32(Foundation.log(Float(self)))
    }
    
    public func exp() -> Int32 {
        return Int32(Foundation.exp(Float(self)))
    }
    
    public func sin() -> Int32 {
        return Int32(round(Float(Foundation.sin(Float(self)))))
    }
    
    public func cos() -> Int32 {
        return Int32(round(Float(Foundation.cos(Float(self)))))
    }
    
    public func tan() -> Int32 {
        return Int32(round(Float(Foundation.tan(Float(self)))))
    }
    
    public func sinh() -> Int32 {
        return Int32(round(Float(Foundation.sinh(Float(self)))))
    }
    
    public func cosh() -> Int32 {
        return Int32(round(Float(Foundation.cosh(Float(self)))))
    }
    
    public func tanh() -> Int32 {
        return Int32(round(Float(Foundation.tanh(Float(self)))))
    }
    
    public init(floatLiteral value: Double) {
        self = Int32(value)
    }
    
}

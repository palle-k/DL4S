//
//  Float.swift
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

extension Float: NumericType {
    public func format(maxDecimals: Int) -> String {
        String(format: "%.\(maxDecimals)f", self)
    }

    public func toUInt8() -> UInt8 {
        UInt8(Swift.max(Swift.min(self, 255), 0))
    }

    public static var one: Float {
        1.0
    }

    public func toInt() -> Int {
        Int(self)
    }

    public var floatValue: Float {
        self
    }

    public var doubleValue: Double {
        Double(self)
    }

    public var intValue: Int32 {
        Int32(self)
    }

    public func sqrt() -> Float {
        Foundation.sqrt(self)
    }

    public func exp() -> Float {
        Foundation.exp(self)
    }

    public func log() -> Float {
        Foundation.log(self)
    }

    public func sin() -> Float {
        Foundation.sin(self)
    }

    public func cos() -> Float {
        Foundation.cos(self)
    }

    public func tan() -> Float {
        Foundation.tanh(self)
    }

    public func sinh() -> Float {
        Foundation.sinh(self)
    }

    public func cosh() -> Float {
        Foundation.cosh(self)
    }

    public func tanh() -> Float {
        Foundation.tanh(self)
    }
}

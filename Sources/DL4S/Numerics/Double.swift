//
//  Double.swift
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

extension Double: NumericType {
    public func format(maxDecimals: Int) -> String {
        String(format: "%.\(maxDecimals)f", self)
    }

    public func toUInt8() -> UInt8 {
        UInt8(self)
    }

    public func toInt() -> Int {
        Int(self)
    }

    public var floatValue: Float {
        Float(self)
    }

    public var doubleValue: Double {
        self
    }

    public var intValue: Int32 {
        Int32(self)
    }

    public static var one: Double {
        1.0
    }

    public func sqrt() -> Double {
        Foundation.sqrt(self)
    }

    public func exp() -> Double {
        Foundation.exp(self)
    }

    public func log() -> Double {
        Foundation.log(self)
    }

    public func sin() -> Double {
        Foundation.sin(self)
    }

    public func cos() -> Double {
        Foundation.cos(self)
    }

    public func tan() -> Double {
        Foundation.tanh(self)
    }

    public func sinh() -> Double {
        Foundation.sinh(self)
    }

    public func cosh() -> Double {
        Foundation.cosh(self)
    }

    public func tanh() -> Double {
        Foundation.tanh(self)
    }
}

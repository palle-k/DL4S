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
import Accelerate

public protocol NumericType: Hashable, ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral, Codable, Comparable {
    init(_ floatValue: Double)
    init(_ integerValue: Int32)
    
    static var zero: Self { get }
    static var one: Self { get }
    
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
    
    static func pow(base: Self, exponent: Self) -> Self
    
    init(_ float: Float)
    init(_ int: Int)
    init(_ uint: UInt)
    init(_ uint8: UInt8)
    
    func toUInt8() -> UInt8
    
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, stride: Int, count: Int)
    
    static func transpose(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, srcRows: Int, srcCols: Int)
    
    static func vAdd(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vNeg(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vSub(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vMA(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, add: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMul(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vSquare(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func matMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    static func matMulAddInPlace(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool)
    static func dot(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, count: Int) -> Self
    
    static func vMulSA(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, add: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMulVAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, add: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func log(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func exp(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func relu(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tanh(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sqrt(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sum(val: UnsafeBufferPointer<Self>, count: Int) -> Self
    static func sum(val: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> Self
    
    static func copysign(values: UnsafeBufferPointer<Self>, signs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func argmax(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    static func argmin(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    
    static func conv2d(input: UnsafeBufferPointer<Self>, filter: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int)
    
    static func copy(values: UnsafeBufferPointer<Self>, srcStride: Int, result: UnsafeMutableBufferPointer<Self>, dstStride: Int, count: Int)
}

extension UInt8 {
    init<Element: NumericType>(_ element: Element) {
        self = element.toUInt8()
    }
}

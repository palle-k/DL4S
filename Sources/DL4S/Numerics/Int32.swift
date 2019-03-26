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
import Accelerate


extension Int32: NumericType {
    public var isFinite: Bool {
        return true
    }
    
    public var isNaN: Bool {
        return false
    }
    
    public func toUInt8() -> UInt8 {
        return UInt8(self)
    }
    
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, stride: Int, count: Int) {
        vDSP_vfilli([value], result.pointer(capacity: count), stride, UInt(count))
    }
    
    public static func vSquare(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = values[i] * values[i]
        }
    }
    
    public init(floatLiteral value: Double) {
        self = Int32(value)
    }
    
    public static func tanh(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("Tanh not implemented for Int32")
    }
    
    public static func relu(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = Swift.max(0, val[i])
        }
    }
    
    public static func transpose(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, srcRows: Int, srcCols: Int) {
        for x in 0 ..< srcCols {
            for y in 0 ..< srcRows {
                result[y + x * srcRows] = val[y * srcCols + x]
            }
        }
    }
    
    public static func pow(base: Int32, exponent: Int32) -> Int32 {
        return repeatElement(base, count: Int(exponent)).reduce(1, *)
    }
    
    public static var one: Int32 {
        return 1
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        vDSP_vsaddi(lhs.pointer(capacity: count), 1, [rhs], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs
        }
    }
    
    public static func svDiv(lhs: Int32, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs / rhs[i]
        }
    }
    
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        vDSP_vfilli([value], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func log(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = Int32(Foundation.log(Float(val[i])))
        }
    }
    
    public static func exp(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = Int32(Foundation.exp(Float(val[i])))
        }
    }
    
    public static func matMul(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        fatalError("Matrix multiplication not supported for type Int32")
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = Int32(Foundation.sqrt(Float(val[i])))
        }
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] - rhs[i]
        }
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i]
        }
    }
    
    public static func vMA(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, add: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i] + add[i]
        }
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i]
        }
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        vDSP_vaddi(lhs.pointer(capacity: count), 1, rhs.pointer(capacity: count), 1, result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = -val[i]
        }
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, count: Int) -> Int32 {
        var result: Int32 = 0
        for i in 0 ..< count {
            result += val[i]
        }
        return result
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> Int32 {
        var result: Int32 = 0
        for i in 0 ..< count {
            result += val[i * stride]
        }
        return result
    }
    
    public static func copysign(values: UnsafeBufferPointer<Int32>, signs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            if signs[i] < 0 {
                result[i] = -values[i]
            } else if signs[i] > 0 {
                result[i] = values[i]
            } else {
                result[i] = 0
            }
        }
    }
    
    public static func dot(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, count: Int) -> Int32 {
        var result: Int32 = 0
        for i in 0 ..< count {
            result += lhs[i] * rhs[i]
        }
        return result
    }
    
    public static func vMulSA(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, add: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs[i] + add
        }
    }
    
    public static func vsMulVAdd(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, add: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        for i in 0 ..< count {
            result[i] = lhs[i] * rhs + add[i]
        }
    }
    
    public static func matMulAddInPlace(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool = false, transposeSecond: Bool = false) {
        fatalError("Matrix multiplication not defined for Int32")
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
    
    public static func argmax(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        precondition(count > 0)
        var maxI = 0
        var maxV = values[0]
        
        for i in 0 ..< count {
            if maxV < values[i] {
                maxI = i
                maxV = values[i]
            }
        }
        
        return (maxI, maxV)
    }
    
    public static func conv2d(input: UnsafeBufferPointer<Int32>, filter: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) {
        fatalError("Conv2D is not available for data type Int32")
    }
    
    public static func argmin(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        precondition(count > 0)
        var minI = 0
        var minV = values[0]
        
        for i in 0 ..< count {
            if minV > values[i] {
                minI = i
                minV = values[i]
            }
        }
        
        return (minI, minV)
    }
}

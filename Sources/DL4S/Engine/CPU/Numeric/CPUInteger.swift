//
//  File.swift
//  
//
//  Created by Palle Klewitz on 15.06.20.
//  Copyright (c) 2020 - Palle Klewitz
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

// Extensions for integers that use unchecked operations for improved performance.
public extension CPUNumeric where Self: FixedWidthInteger & Numeric {
    static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))
        gemm_generic(
            transposeFirst,
            transposeSecond,
            resultShape.0, resultShape.1, transposeFirst ? lhsShape.0 : lhsShape.1,
            alpha,
            lhs.pointer(capacity: lhsShape.0 * lhsShape.1), lhsShape.1,
            rhs.pointer(capacity: rhsShape.0 * rhsShape.1), rhsShape.1,
            beta,
            result.pointer(capacity: resultShape.0 * resultShape.1), resultShape.1
        )
    }
    
    static func argmax(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self) {
        var maxI: Int = 0
        var maxV: Self = 0
        let src = values.pointer(capacity: count)
        maxV = src[0];
        for i in 0 ..< count {
            let v = src[i]
            if v > maxV {
                maxV = v
                maxI = i
            }
        }
        return (Int(maxI), maxV)
    }
    
    static func argmin(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self) {
        var minI: Int = 0
        var minV: Self = 0
        let src = values.pointer(capacity: count)
        minV = src[0];
        for i in 0 ..< count {
            let v = src[i]
            if v < minV {
                minV = v
                minI = i
            }
        }
        return (Int(minI), minV)
    }
    
    static func argmax(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self) {
        if stride == 1 {
            return argmax(values: values, count: count)
        }
        let src = values.pointer(capacity: stride * count)
        var maxI: Int = 0
        var maxV: Self = src[0]
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v > maxV {
                maxV = v
                maxI = i
            }
        }
        return (maxI, maxV)
    }
    
    static func argmin(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self) {
        if stride == 1 {
            return argmin(values: values, count: count)
        }
        var minI: Int = 0
        let src = values.pointer(capacity: stride * count)
        var minV: Self = src[0]
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v < minV {
                minV = v
                minI = i
            }
        }
        return (minI, minV)
    }
    
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let ptr = result.pointer(capacity: count)
        var i = 0
        while i < count {
            ptr[i] = value
            i &+= 1
        }
    }
    
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, stride: Int, count: Int) {
        let ptr = result.pointer(capacity: count &* stride)
        var i = 0
        while i < count {
            ptr[i &* stride] = value
            i &+= 1
        }
    }
    
    static func relu(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let s = src[i];
            dst[i] = s > 0 ? s : 0;
        }
    }
    
    static func transpose(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, srcRows: Int, srcCols: Int) {
        let src = val.pointer(capacity: srcRows * srcCols)
        let dst = result.pointer(capacity: srcRows * srcCols)
        for x in 0 ..< srcCols {
            for y in 0 ..< srcRows {
                dst[y &+ x &* srcRows] = src[y &* srcCols &+ x];
            }
        }
    }
    
    static func copy(values: UnsafeBufferPointer<Self>, srcStride: Int, result: UnsafeMutableBufferPointer<Self>, dstStride: Int, count: Int) {
        let src = values.pointer(capacity: count * srcStride)
        let dst = result.pointer(capacity: count * dstStride)
        for i in 0 ..< count {
            dst[i &* dstStride] = src[i &* srcStride]
        }
    }
    
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l >= r ? l : r
        }
    }
    
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l <= r ? l : r
        }
    }
    
    static func vAdd(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = lhs[i] &+ rhs[i]
            i &+= 1
        }
    }
    
    static func vsAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = src[i] &+ rhs;
            i &+= 1
        }
    }
    
    static func vNeg(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = 0 &- src[i]
            i &+= 1
        }
    }
    
    static func vSub(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = lhs[i] &- rhs[i]
            i &+= 1
        }
    }
    
    static func vMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = lhs[i] &* rhs[i]
            i &+= 1
        }
    }
    
    static func vsMul(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = src[i] &* rhs;
        }
    }
    
    static func vDiv(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        var i = 0
        while i < count {
            dst[i] = lhs[i] / rhs[i]
            i &+= 1
        }
    }
    
    static func svDiv(lhs: Self, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let src = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = lhs / src[i]
        }
    }
    
    static func arange(start: Self, end: Self, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        let dst = result.pointer(capacity: count)
        let increment = end / Self.init(count)
        var i = 0
        while i < count {
            dst[i] = start &+ Self.init(i) &* increment
            i &+= 1
        }
        
    }
    
    static func sum(val: UnsafeBufferPointer<Self>, count: Int) -> Self {
        let src = val.pointer(capacity: count)
        var dst: Self = 0
        for i in 0 ..< count {
            dst &+= src[i]
        }
        return dst
    }
    
    static func sum(val: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> Self {
        let src = val.pointer(capacity: count * stride)
        var dst: Self = 0
        for i in 0 ..< count {
            dst &+= src[i &* stride]
        }
        return dst
    }
    
    static func exp(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func log(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func tanh(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func sqrt(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func sin(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func cos(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
    
    static func tan(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int) {
        fatalError("\(#function) not available for integer types")
    }
}

extension UInt8: CPUNumeric {}
extension UInt16: CPUNumeric {}
extension UInt32: CPUNumeric {}
extension UInt64: CPUNumeric {}

extension Int8: CPUNumeric {}
extension Int16: CPUNumeric {}
extension Int64: CPUNumeric {}

extension Int: CPUNumeric {}
extension UInt: CPUNumeric {}

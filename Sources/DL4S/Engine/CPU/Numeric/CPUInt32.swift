//
//  CPUInt32.swift
//  DL4S
//
//  Created by Palle Klewitz on 20.10.19.
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

#if MKL_ENABLE
import MKL
#elseif canImport(Accelerate)
import Accelerate
#endif
#if os(Linux)
import Glibc
#else
import Foundation
#endif


extension Int32: CPUNumeric {
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, stride: Int, count: Int) {
        let dst = result.pointer(capacity: count &* stride)
        #if MKL_ENABLE
        if stride == 1 {
            ippsSet_32s(value, dst, Int32(count))
        } else {
            for i in 0 ..< count {
                dst[i &* stride] = value;
            }
        }
        #elseif canImport(Accelerate)
        vDSP_vfilli([value], dst, stride, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i &* stride] = value;
        }
        #endif
    }
    
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsSet_32s(value, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vfilli([value], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = value;
        }
        #endif
    }
    
    public static func relu(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let s = src[i];
            dst[i] = s > 0 ? s : 0;
        }
    }
    
    public static func transpose(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, srcRows: Int, srcCols: Int) {
        let src = val.pointer(capacity: srcRows * srcCols)
        let dst = result.pointer(capacity: srcRows * srcCols)
        for x in 0 ..< srcCols {
            for y in 0 ..< srcRows {
                dst[y &+ x &* srcRows] = src[y &* srcCols &+ x];
            }
        }
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = -src[i]
        }
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if !MKL_ENABLE && canImport(Accelerate)
        vDSP_vsaddi(src, 1, [rhs], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] &+ rhs;
        }
        #endif
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = src[i] &* rhs;
        }
    }
    
    public static func svDiv(lhs: Int32, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = lhs / src[i]
        }
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if !MKL_ENABLE && canImport(Accelerate)
        vDSP_vaddi(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] &+ rhs[i]
        }
        #endif
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = lhs[i] &- rhs[i]
        }
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = lhs[i] &* rhs[i]
        }
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if !MKL_ENABLE && canImport(Accelerate)
        vDSP_vdivi(rhs, 1, lhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] / rhs[i]
        }
        #endif
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, count: Int) -> Int32 {
        let src = val.pointer(capacity: count)
        var dst: Int32 = 0
        for i in 0 ..< count {
            dst &+= src[i]
        }
        return dst
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> Int32 {
        let src = val.pointer(capacity: count * stride)
        var dst: Int32 = 0
        for i in 0 ..< count {
            dst &+= src[i &* stride]
        }
        return dst
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
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
    
    public static func argmax(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        var maxI: Int = 0
        var maxV: Int32 = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var maxI32: Int32 = 0
        ippsMaxIndx_32s(src, Int32(count), &maxV, &maxI32)
        maxI = Int(maxI32)
        #else
        maxV = Int32.min;
        for i in 0 ..< count {
            let v = src[i]
            if v > maxV {
                maxV = v
                maxI = i
            }
        }
        #endif
        return (Int(maxI), maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        var minI: Int = 0
        var minV: Int32 = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var minI32: Int32 = 0
        ippsMinIndx_32s(src, Int32(count), &minV, &minI32)
        minI = Int(minI32)
        #else
        minV = Int32.max;
        for i in 0 ..< count {
            let v = src[i]
            if v < minV {
                minV = v
                minI = i
            }
        }
        #endif
        return (Int(minI), minV)
    }
    
    public static func argmax(values: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> (Int, Int32) {
        if stride == 1 {
            return argmax(values: values, count: count)
        }
        var maxI: Int = 0
        var maxV: Int32 = Int32.min
        let src = values.pointer(capacity: stride * count)
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v > maxV {
                maxV = v
                maxI = i
            }
        }
        return (maxI, maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> (Int, Int32) {
        if stride == 1 {
            return argmin(values: values, count: count)
        }
        var minI: Int = 0
        var minV: Int32 = Int32.max
        let src = values.pointer(capacity: stride * count)
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v < minV {
                minV = v
                minI = i
            }
        }
        return (minI, minV)
    }
    
    public static func copy(values: UnsafeBufferPointer<Int32>, srcStride: Int, result: UnsafeMutableBufferPointer<Int32>, dstStride: Int, count: Int) {
        let src = values.pointer(capacity: count * srcStride)
        let dst = result.pointer(capacity: count * dstStride)
        for i in 0 ..< count {
            dst[i &* dstStride] = src[i &* srcStride]
        }
    }
    
    public static func arange(start: Int32, end: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let dst = result.pointer(capacity: count)
        let increment = end / Int32(count)
        #if MKL_ENABLE
        ippsVectorSlope_32s(dst, Int32(count), Double(start), Double(increment))
        #else
        for i in 0 ..< count {
            dst[i] = start &+ Int32(i) &* increment
        }
        #endif
    }
    
    public static func max(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l >= r ? l : r
        }
    }
    
    public static func min(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l <= r ? l : r
        }
    }
    
    public static func exp(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func log(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func tanh(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = src[i] > 0 ? 1 : 0
        }
    }
    
    public static func sin(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func cos(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
    
    public static func tan(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) not available for Int32")
    }
}


//
//  CPUDouble.swift
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

extension Double: CPUNumeric {
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, stride: Int, count: Int) {
        let dst = result.pointer(capacity: count &* stride)
        #if MKL_ENABLE
        if stride == 1 {
            ippsSet_64f(value, dst, Int32(count))
        } else {
            for i in 0 ..< count {
                dst[i &* stride] = value;
            }
        }
        #elseif canImport(Accelerate)
        vDSP_vfillD([value], dst, stride, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i &* stride] = value;
        }
        #endif
    }
    
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsSet_64f(value, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vfillD([value], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = value;
        }
        #endif
    }
    
    public static func relu(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsThreshold_64f(src, dst, Int32(count), 0, ippCmpLess)
        #elseif canImport(Accelerate)
        vDSP_vthresD(src, 1, [0], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let s = src[i];
            dst[i] = s > 0 ? s : 0;
        }
        #endif
    }
    
    public static func transpose(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, srcRows: Int, srcCols: Int) {
        let src = val.pointer(capacity: srcRows * srcCols)
        let dst = result.pointer(capacity: srcRows * srcCols)
        #if MKL_ENABLE
        MKL_Domatcopy("R".utf8CString[0], "T".utf8CString[0], srcRows, srcCols, 1, src, srcCols, dst, srcRows)
        #elseif canImport(Accelerate)
        vDSP_mtransD(src, 1, dst, 1, UInt(srcCols), UInt(srcRows))
        #else
        for x in 0 ..< srcCols {
            for y in 0 ..< srcRows {
                dst[y &+ x &* srcRows] = src[y &* srcCols &+ x];
            }
        }
        #endif
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMulC_64f(src, -1, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vnegD(src, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = -src[i]
        }
        #endif
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsAddC_64f(src, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vsaddD(src, 1, [rhs], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] + rhs;
        }
        #endif
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMulC_64f(src, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vsmulD(src, 1, [rhs], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] * rhs;
        }
        #endif
    }
    
    public static func svDiv(lhs: Double, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if !MKL_ENABLE && canImport(Accelerate)
        vDSP_svdivD([lhs], src, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs / src[i]
        }
        #endif
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsAdd_64f(lhs, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vaddD(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] + rhs[i]
        }
        #endif
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        vdSub(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vsubD(rhs, 1, lhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] - rhs[i]
        }
        #endif
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        MKL.vdMul(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vmulD(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] * rhs[i]
        }
        #endif
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        MKL.vdDiv(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vdivD(rhs, 1, lhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] / rhs[i]
        }
        #endif
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, count: Int) -> Double {
        let src = val.pointer(capacity: count)
        var dst: Double = 0
        
        #if MKL_ENABLE
        ippsSum_64f(src, Int32(count), &dst)
        #elseif canImport(Accelerate)
        vDSP_sveD(src, 1, &dst, UInt(count))
        #else
        for i in 0 ..< count {
            dst += src[i]
        }
        #endif
        return dst
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> Double {
        let src = val.pointer(capacity: count * stride)
        var dst: Double = 0
        
        #if MKL_ENABLE
        if stride == 1 {
            ippsSum_64f(src, Int32(count), &dst)
        } else {
            for i in 0 ..< count {
                dst += src[i &* stride]
            }
        }
        return dst
        #elseif canImport(Accelerate)
        vDSP_sveD(src, stride, &dst, UInt(count))
        return dst
        #else
        for i in 0 ..< count {
            dst += src[i &* stride]
        }
        return dst
        #endif
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))

        #if MKL_ENABLE || canImport(Accelerate)
        cblas_dgemm(
            CblasRowMajor,
            transposeFirst ? CblasTrans : CblasNoTrans,
            transposeSecond ? CblasTrans : CblasNoTrans,
            Int32(resultShape.0), Int32(resultShape.1), Int32(transposeFirst ? lhsShape.0 : lhsShape.1),
            alpha,
            lhs.pointer(capacity: lhsShape.0 * lhsShape.1), Int32(lhsShape.1),
            rhs.pointer(capacity: rhsShape.0 * rhsShape.1), Int32(rhsShape.1),
            beta,
            result.pointer(capacity: resultShape.0 * resultShape.1), Int32(resultShape.1)
        )
        #else
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
        #endif
    }
    
    public static func argmax(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var maxI: Int = 0
        var maxV: Double = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var maxI32: Int32 = 0
        ippsMaxIndx_64f(src, Int32(count), &maxV, &maxI32)
        maxI = Int(maxI32)
        #elseif canImport(Accelerate)
        var maxIU: UInt = 0
        vDSP_maxviD(src, 1, &maxV, &maxIU, UInt(count))
        maxI = Int(maxIU)
        #else
        maxV = -Double.infinity;
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
    
    public static func argmin(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var minI: Int = 0
        var minV: Double = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var minI32: Int32 = 0
        ippsMinIndx_64f(src, Int32(count), &minV, &minI32)
        minI = Int(minI32)
        #elseif canImport(Accelerate)
        var minIU: UInt = 0
        vDSP_minviD(src, 1, &minV, &minIU, UInt(count))
        minI = Int(minIU)
        #else
        minV = Double.infinity;
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
    
    public static func argmax(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        if stride == 1 {
            return argmax(values: values, count: count)
        }
        
        #if canImport(Accelerate) && !MKL_ENABLE
        var maxI: UInt = 0
        var maxV: Double = 0
        vDSP_maxviD(values.pointer(capacity: count * stride), stride, &maxV, &maxI, UInt(count))
        return (Int(maxI) / stride, maxV)
        #else
        var maxI: Int = 0
        var maxV: Double = -Double.infinity
        let src = values.pointer(capacity: stride * count)
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v > maxV {
                maxV = v
                maxI = i
            }
        }
        return (maxI, maxV)
        #endif
    }
    
    public static func argmin(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        if stride == 1 {
            return argmin(values: values, count: count)
        }
        
        #if canImport(Accelerate) && !MKL_ENABLE
        var minI: UInt = 0
        var minV: Double = 0
        vDSP_minviD(values.pointer(capacity: count * stride), stride, &minV, &minI, UInt(count))
        return (Int(minI) / stride, minV)
        #else
        var minI: Int = 0
        var minV: Double = Double.infinity
        let src = values.pointer(capacity: stride * count)
        for i in 0 ..< count {
            let v = src[i &* stride]
            if v < minV {
                minV = v
                minI = i
            }
        }
        return (minI, minV)
        #endif
    }
    
    public static func copy(values: UnsafeBufferPointer<Double>, srcStride: Int, result: UnsafeMutableBufferPointer<Double>, dstStride: Int, count: Int) {
        let src = values.pointer(capacity: count * srcStride)
        let dst = result.pointer(capacity: count * dstStride)
        
        #if MKL_ENABLE || canImport(Accelerate)
        cblas_dcopy(Int32(count), src, Int32(srcStride), dst, Int32(dstStride))
        #else
        for i in 0 ..< count {
            dst[i &* dstStride] = src[i &* srcStride]
        }
        #endif
    }
    
    public static func arange(start: Double, end: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let dst = result.pointer(capacity: count)
        let increment = end / Double(count)
        #if MKL_ENABLE
        ippsVectorSlope_64f(dst, Int32(count), start, increment)
        #elseif canImport(Accelerate)
        vDSP_vrampD([start], [increment], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = start + Double(i) * increment
        }
        #endif
    }
    
    public static func max(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMaxEvery_64f(lhs, rhs, dst, UInt32(count))
        #elseif canImport(Accelerate)
        vDSP_vmaxD(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l >= r ? l : r
        }
        #endif
    }
    
    public static func min(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMinEvery_64f(lhs, rhs, dst, UInt32(count))
        #elseif canImport(Accelerate)
        vDSP_vminD(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l <= r ? l : r
        }
        #endif
    }
    
    public static func exp(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdExp(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvexp(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = exp(src[i])
        }
        #endif
    }
    
    public static func log(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdLn(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvlog(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = log(src[i])
        }
        #endif
    }
    
    public static func tanh(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdTanh(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvtanh(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = tanh(src[i])
        }
        #endif
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdSqrt(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvsqrt(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = sqrt(src[i])
        }
        #endif
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = src[i] > 0 ? 1 : 0
        }
    }
    
    public static func sin(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdSin(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvsin(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = sin(src[i])
        }
        #endif
    }
    
    public static func cos(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdCos(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvcos(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = cos(src[i])
        }
        #endif
    }
    
    public static func tan(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vdTan(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvtan(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = tan(src[i])
        }
        #endif
    }
}


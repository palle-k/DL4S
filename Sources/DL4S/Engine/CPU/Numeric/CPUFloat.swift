//
//  CPUFloat.swift
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
#else
#warning("Compiling DL4S without any accelerator library")
#warning("DL4S can be accelerated with Intel MKL, IPP and TBB. See README.md")
#endif
#if os(Linux)
import Glibc
#else
import Foundation
#endif

extension Float: CPUNumeric {
    public static func fill(value: Float, result: UnsafeMutableBufferPointer<Float>, stride: Int, count: Int) {
        let dst = result.pointer(capacity: count &* stride)
        #if MKL_ENABLE
        if stride == 1 {
            ippsSet_32f(value, dst, Int32(count))
        } else {
            for i in 0 ..< count {
                dst[i &* stride] = value;
            }
        }
        #elseif canImport(Accelerate)
        vDSP_vfill([value], dst, stride, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i &* stride] = value;
        }
        #endif
    }
    
    public static func fill(value: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsSet_32f(value, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vfill([value], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = value;
        }
        #endif
    }
    
    public static func relu(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsThreshold_32f(src, dst, Int32(count), 0, ippCmpLess)
        #elseif canImport(Accelerate)
        vDSP_vthres(src, 1, [0], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let s = src[i];
            dst[i] = s > 0 ? s : 0;
        }
        #endif
    }
    
    public static func transpose(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, srcRows: Int, srcCols: Int) {
        let src = val.pointer(capacity: srcRows * srcCols)
        let dst = result.pointer(capacity: srcRows * srcCols)
        #if MKL_ENABLE
        MKL_Somatcopy("R".utf8CString[0], "T".utf8CString[0], srcRows, srcCols, 1, src, srcCols, dst, srcRows)
        #elseif canImport(Accelerate)
        vDSP_mtrans(src, 1, dst, 1, UInt(srcCols), UInt(srcRows))
        #else
        for x in 0 ..< srcCols {
            for y in 0 ..< srcRows {
                dst[y &+ x &* srcRows] = src[y &* srcCols &+ x];
            }
        }
        #endif
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMulC_32f(src, -1, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vneg(src, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = -src[i]
        }
        #endif
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Float>, rhs: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsAddC_32f(src, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vsadd(src, 1, [rhs], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] + rhs;
        }
        #endif
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Float>, rhs: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMulC_32f(lhs, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vsmul(lhs, 1, [rhs], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] * rhs;
        }
        #endif
    }
    
    public static func svDiv(lhs: Float, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsDivCRev_32f(src, lhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_svdiv([lhs], src, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs / src[i]
        }
        #endif
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsAdd_32f(lhs, rhs, dst, Int32(count))
        #elseif canImport(Accelerate)
        vDSP_vadd(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] + rhs[i]
        }
        #endif
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        vsSub(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vsub(rhs, 1, lhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] - rhs[i]
        }
        #endif
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        MKL.vsMul(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vmul(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] * rhs[i]
        }
        #endif
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        
        #if MKL_ENABLE
        MKL.vsDiv(Int32(count), lhs, rhs, dst)
        #elseif canImport(Accelerate)
        vDSP_vdiv(rhs, 1, lhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = lhs[i] / rhs[i]
        }
        #endif
    }
    
    public static func sum(val: UnsafeBufferPointer<Float>, count: Int) -> Float {
        let src = val.pointer(capacity: count)
        var dst: Float = 0
        
        #if MKL_ENABLE
        ippsSum_32f(src, Int32(count), &dst, ippAlgHintFast)
        #elseif canImport(Accelerate)
        vDSP_sve(src, 1, &dst, UInt(count))
        #else
        for i in 0 ..< count {
            dst += src[i]
        }
        #endif
        return dst
    }
    
    public static func sum(val: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> Float {
        let src = val.pointer(capacity: (count - 1) * stride + 1)
        var dst: Float = 0
        
        #if MKL_ENABLE
        if stride == 1 {
            ippsSum_32f(src, Int32(count), &dst, ippAlgHintFast)
        } else {
            for i in 0 ..< count {
                dst += src[i &* stride]
            }
        }
        return dst
        #elseif canImport(Accelerate)
        vDSP_sve(src, stride, &dst, UInt(count))
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
        cblas_sgemm(
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
    
    public static func argmax(values: UnsafeBufferPointer<Float>, count: Int) -> (Int, Float) {
        var maxI: Int = 0
        var maxV: Float = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var maxI32: Int32 = 0
        ippsMaxIndx_32f(src, Int32(count), &maxV, &maxI32)
        maxI = Int(maxI32)
        #elseif canImport(Accelerate)
        var maxIU: UInt = 0
        vDSP_maxvi(src, 1, &maxV, &maxIU, UInt(count))
        maxI = Int(maxIU)
        #else
        maxV = -Float.infinity;
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
    
    public static func argmin(values: UnsafeBufferPointer<Float>, count: Int) -> (Int, Float) {
        var minI: Int = 0
        var minV: Float = 0
        let src = values.pointer(capacity: count)
        
        #if MKL_ENABLE
        var minI32: Int32 = 0
        ippsMinIndx_32f(src, Int32(count), &minV, &minI32)
        minI = Int(minI32)
        #elseif canImport(Accelerate)
        var minIU: UInt = 0
        vDSP_minvi(src, 1, &minV, &minIU, UInt(count))
        minI = Int(minIU)
        #else
        minV = Float.infinity;
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
    
    public static func argmax(values: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> (Int, Float) {
        if stride == 1 {
            return argmax(values: values, count: count)
        }
        
        #if canImport(Accelerate) && !MKL_ENABLE
        var maxI: UInt = 0
        var maxV: Float = 0
        vDSP_maxvi(values.pointer(capacity: (count - 1) * stride + 1), stride, &maxV, &maxI, UInt(count))
        return (Int(maxI) / stride, maxV)
        #else
        var maxI: Int = 0
        var maxV: Float = -Float.infinity
        let src = values.pointer(capacity: stride * (count - 1) + 1)
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
    
    public static func argmin(values: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> (Int, Float) {
        if stride == 1 {
            return argmin(values: values, count: count)
        }
        
        #if canImport(Accelerate) && !MKL_ENABLE
        var minI: UInt = 0
        var minV: Float = 0
        vDSP_minvi(values.pointer(capacity: (count - 1) * stride + 1), stride, &minV, &minI, UInt(count))
        return (Int(minI) / stride, minV)
        #else
        var minI: Int = 0
        var minV: Float = Float.infinity
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
    
    public static func copy(values: UnsafeBufferPointer<Float>, srcStride: Int, result: UnsafeMutableBufferPointer<Float>, dstStride: Int, count: Int) {
        let src = values.pointer(capacity: count * srcStride)
        let dst = result.pointer(capacity: count * dstStride)
        
        #if MKL_ENABLE || canImport(Accelerate)
        cblas_scopy(Int32(count), src, Int32(srcStride), dst, Int32(dstStride))
        #else
        for i in 0 ..< count {
            dst[i &* dstStride] = src[i &* srcStride]
        }
        #endif
    }
    
    public static func arange(start: Float, end: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let dst = result.pointer(capacity: count)
        let increment = end / Float(count)
        #if MKL_ENABLE
        ippsVectorSlope_32f(dst, Int32(count), start, increment)
        #elseif canImport(Accelerate)
        vDSP_vramp([start], [increment], dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            dst[i] = start + Float(i) * increment
        }
        #endif
    }
    
    public static func max(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMaxEvery_32f(lhs, rhs, dst, UInt32(count))
        #elseif canImport(Accelerate)
        vDSP_vmax(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l >= r ? l : r
        }
        #endif
    }
    
    public static func min(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let lhs = lhs.pointer(capacity: count)
        let rhs = rhs.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        ippsMinEvery_32f(lhs, rhs, dst, UInt32(count))
        #elseif canImport(Accelerate)
        vDSP_vmin(lhs, 1, rhs, 1, dst, 1, UInt(count))
        #else
        for i in 0 ..< count {
            let l = lhs[i]
            let r = rhs[i]
            dst[i] = l <= r ? l : r
        }
        #endif
    }
    
    public static func exp(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsExp(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvexpf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = expf(src[i])
        }
        #endif
    }
    
    public static func log(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsLn(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvlogf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = logf(src[i])
        }
        #endif
    }
    
    public static func tanh(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsTanh(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvtanhf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = tanhf(src[i])
        }
        #endif
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = val.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsSqrt(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvsqrtf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = sqrtf(src[i])
        }
        #endif
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        for i in 0 ..< count {
            dst[i] = src[i] > 0 ? 1 : 0
        }
    }
    
    public static func sin(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsSin(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvsinf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = sinf(src[i])
        }
        #endif
    }
    
    public static func cos(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsCos(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvcosf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = cosf(src[i])
        }
        #endif
    }
    
    public static func tan(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        let src = values.pointer(capacity: count)
        let dst = result.pointer(capacity: count)
        #if MKL_ENABLE
        vsTan(Int32(count), src, dst)
        #elseif canImport(Accelerate)
        vvtanf(dst, src, [Int32(count)])
        #else
        for i in 0 ..< count {
            dst[i] = tanf(src[i])
        }
        #endif
    }
}




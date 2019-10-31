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

import Foundation
import DL4SLib

extension Double: CPUNumeric {
    
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, stride: Int, count: Int) {
        d4lib_dfill([value], result.pointer(capacity: count), stride, UInt(count))
    }
    
    public static func fill(value: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dfill([value], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func relu(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dthreshold(val.pointer(capacity: count), [0.0], result.pointer(capacity: count), UInt(count))
    }
    
    public static func tanh(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dtanh(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func transpose(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, srcRows: Int, srcCols: Int) {
        d4lib_dtranspose(val.pointer(capacity: srcRows * srcCols), result.pointer(capacity: srcRows * srcCols), UInt(srcCols), UInt(srcRows))
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dsqrt(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_daddvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Double>, rhs: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dmulvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func svDiv(lhs: Double, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_ddivsv([lhs], rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func exp(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dexp(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func log(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dlog(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_daddv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dneg(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dsubv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dmulv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_ddivv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, count: Int) -> Double {
        var result: Double = 0
        d4lib_dsum(val.pointer(capacity: count), 1, &result, UInt(count))
        return result
    }
    
    public static func sum(val: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> Double {
        var result: Double = 0
        d4lib_dsum(val.pointer(capacity: count), stride, &result, UInt(count))
        return result
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))

        d4lib_dgemm(
            D4LIB_RowMajor,
            transposeFirst ? D4LIB_Trans : D4LIB_NoTrans,
            transposeSecond ? D4LIB_Trans : D4LIB_NoTrans,
            Int32(resultShape.0),
            Int32(resultShape.1),
            Int32(transposeFirst ? lhsShape.0 : lhsShape.1),
            alpha,
            lhs.pointer(capacity: lhsShape.0 * lhsShape.1),
            Int32(lhsShape.1),
            rhs.pointer(capacity: rhsShape.0 * rhsShape.1),
            Int32(rhsShape.1),
            beta,
            result.pointer(capacity: resultShape.0 * resultShape.1),
            Int32(resultShape.1)
        )
    }
    
    public static func argmax(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var maxI: UInt = 0
        var maxV: Double = 0
        
        d4lib_dmaxi(values.pointer(capacity: count), 1, &maxV, &maxI, UInt(count))
        
        return (Int(maxI), maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Double>, count: Int) -> (Int, Double) {
        var minI: UInt = 0
        var minV: Double = 0
        
        d4lib_dmini(values.pointer(capacity: count), 1, &minV, &minI, UInt(count))
        
        return (Int(minI), minV)
    }
    
    public static func argmax(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        var minI: UInt = 0
        var minV: Double = 0
        
        d4lib_dmaxi(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Double>, stride: Int, count: Int) -> (Int, Double) {
        var minI: UInt = 0
        var minV: Double = 0
        
        d4lib_dmini(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dheaviside(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func copy(values: UnsafeBufferPointer<Double>, srcStride: Int, result: UnsafeMutableBufferPointer<Double>, dstStride: Int, count: Int) {
        d4lib_dcopy_strided(values.pointer(capacity: count * srcStride), srcStride, result.pointer(capacity: dstStride * count), dstStride, UInt(count))
    }
    
    public static func arange(start: Double, end: Double, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dramp([start], [end / Double(count)], result.pointer(capacity: count), UInt(count))
    }
    
    public static func max(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dmax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func min(lhs: UnsafeBufferPointer<Double>, rhs: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dmax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sin(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dsin(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func cos(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dcos(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func tan(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, count: Int) {
        d4lib_dtan(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func img2col(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_dimg2col(
            values.baseAddress!,
            result.baseAddress!,
            D4LIB_Img2ColSetup(
                batch_size: batchSize,
                channels: channels,
                height: height,
                width: width,
                kernel_height: kernelHeight,
                kernel_width: kernelWidth,
                padding: padding,
                stride: stride
            )
        )
    }
    
    public static func col2img(values: UnsafeBufferPointer<Double>, result: UnsafeMutableBufferPointer<Double>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_dcol2img(
            values.baseAddress!,
            result.baseAddress!,
            D4LIB_Img2ColSetup(
                batch_size: batchSize,
                channels: channels,
                height: height,
                width: width,
                kernel_height: kernelHeight,
                kernel_width: kernelWidth,
                padding: padding,
                stride: stride
            )
        )
    }
    
    public static func scatter(values: UnsafeBufferPointer<Double>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Double>, dst_shape: [Int], axis: Int) {
        var src_shape = dst_shape
        src_shape.remove(at: axis)
        d4lib_dscatter(
            values.pointer(capacity: src_shape.reduce(1, *)),
            context.pointer(capacity: src_shape.reduce(1, *)),
            result.pointer(capacity: dst_shape.reduce(1, *)),
            Int32(dst_shape.count),
            dst_shape.map(Int32.init),
            Int32(axis)
        )
    }
    
    public static func gather(values: UnsafeBufferPointer<Double>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Double>, src_shape: [Int], axis: Int) {
        var dst_shape = src_shape
        dst_shape.remove(at: axis)
        d4lib_dgather(
            values.pointer(capacity: src_shape.reduce(1, *)),
            Int32(src_shape.count),
            src_shape.map(Int32.init),
            context.pointer(capacity: dst_shape.reduce(1, *)),
            result.pointer(capacity: dst_shape.reduce(1, *)),
            Int32(axis)
        )
    }
}


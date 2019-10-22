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

import Foundation
import DL4SLib

extension Float: CPUNumeric {
    
    public static func fill(value: Float, result: UnsafeMutableBufferPointer<Float>, stride: Int, count: Int) {
        d4lib_sfill([value], result.pointer(capacity: count), stride, UInt(count))
    }
    
    public static func fill(value: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sfill([value], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func relu(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sthreshold(val.pointer(capacity: count), [0.0], result.pointer(capacity: count), UInt(count))
    }
    
    public static func tanh(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_stanh(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func transpose(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, srcRows: Int, srcCols: Int) {
        d4lib_stranspose(val.pointer(capacity: srcRows * srcCols), result.pointer(capacity: srcRows * srcCols), UInt(srcCols), UInt(srcRows))
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_ssqrt(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Float>, rhs: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_saddvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Float>, rhs: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_smulvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func svDiv(lhs: Float, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sdivsv([lhs], rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func exp(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sexp(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func log(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_slog(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_saddv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sneg(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_ssubv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_smulv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sdivv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sum(val: UnsafeBufferPointer<Float>, count: Int) -> Float {
        var result: Float = 0
        d4lib_ssum(val.pointer(capacity: count), 1, &result, UInt(count))
        return result
    }
    
    public static func sum(val: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> Float {
        var result: Float = 0
        d4lib_ssum(val.pointer(capacity: count), stride, &result, UInt(count))
        return result
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))

        d4lib_sgemm(
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
    
    public static func argmax(values: UnsafeBufferPointer<Float>, count: Int) -> (Int, Float) {
        var maxI: UInt = 0
        var maxV: Float = 0
        
        d4lib_smaxi(values.pointer(capacity: count), 1, &maxV, &maxI, UInt(count))
        
        return (Int(maxI), maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Float>, count: Int) -> (Int, Float) {
        var minI: UInt = 0
        var minV: Float = 0
        
        d4lib_smini(values.pointer(capacity: count), 1, &minV, &minI, UInt(count))
        
        return (Int(minI), minV)
    }
    
    public static func argmax(values: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> (Int, Float) {
        var minI: UInt = 0
        var minV: Float = 0
        
        d4lib_smaxi(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Float>, stride: Int, count: Int) -> (Int, Float) {
        var minI: UInt = 0
        var minV: Float = 0
        
        d4lib_smini(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sheaviside(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func copy(values: UnsafeBufferPointer<Float>, srcStride: Int, result: UnsafeMutableBufferPointer<Float>, dstStride: Int, count: Int) {
        d4lib_scopy_strided(values.pointer(capacity: count * srcStride), srcStride, result.pointer(capacity: dstStride * count), dstStride, UInt(count))
    }
    
    public static func arange(start: Float, end: Float, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_sramp([start], [end / Float(count)], result.pointer(capacity: count), UInt(count))
    }
    
    public static func max(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_smax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func min(lhs: UnsafeBufferPointer<Float>, rhs: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_smax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sin(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_ssin(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func cos(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_scos(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func tan(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, count: Int) {
        d4lib_stan(values.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func img2col(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_simg2col(
            values.baseAddress!,
            result.baseAddress!,
            D4LIB_Img2ColSetup(
                batch_size: Int32(batchSize),
                channels: Int32(channels),
                height: Int32(height),
                width: Int32(width),
                kernel_height: Int32(kernelHeight),
                kernel_width: Int32(kernelWidth),
                padding: Int32(padding),
                stride: Int32(stride)
            )
        )
    }
    
    public static func col2img(values: UnsafeBufferPointer<Float>, result: UnsafeMutableBufferPointer<Float>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_scol2img(
            values.baseAddress!,
            result.baseAddress!,
            D4LIB_Img2ColSetup(
                batch_size: Int32(batchSize),
                channels: Int32(channels),
                height: Int32(height),
                width: Int32(width),
                kernel_height: Int32(kernelHeight),
                kernel_width: Int32(kernelWidth),
                padding: Int32(padding),
                stride: Int32(stride)
            )
        )
    }
}

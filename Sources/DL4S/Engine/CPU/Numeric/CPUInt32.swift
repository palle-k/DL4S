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

import Foundation
import DL4SLib


extension Int32: CPUNumeric {
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, stride: Int, count: Int) {
        d4lib_ifill([value], result.pointer(capacity: count), stride, UInt(count))
    }
    
    public static func fill(value: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_ifill([value], result.pointer(capacity: count), 1, UInt(count))
    }
    
    public static func relu(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_ithreshold(val.pointer(capacity: count), [0.0], result.pointer(capacity: count), UInt(count))
    }
    
    public static func tanh(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func transpose(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, srcRows: Int, srcCols: Int) {
        d4lib_itranspose(val.pointer(capacity: srcRows * srcCols), result.pointer(capacity: srcRows * srcCols), UInt(srcCols), UInt(srcRows))
    }
    
    public static func sqrt(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func vsAdd(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_iaddvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func vsMul(lhs: UnsafeBufferPointer<Int32>, rhs: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_imulvs(lhs.pointer(capacity: count), [rhs], result.pointer(capacity: count), UInt(count))
    }
    
    public static func svDiv(lhs: Int32, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_idivsv([lhs], rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func exp(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func log(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func vAdd(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_iaddv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vNeg(val: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_ineg(val.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vSub(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_isubv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vMul(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_imulv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func vDiv(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_idivv(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, count: Int) -> Int32 {
        var result: Int32 = 0
        d4lib_isum(val.pointer(capacity: count), 1, &result, UInt(count))
        return result
    }
    
    public static func sum(val: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> Int32 {
        var result: Int32 = 0
        d4lib_isum(val.pointer(capacity: count), stride, &result, UInt(count))
        return result
    }
    
    public static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool) {
        precondition((transposeFirst ? lhsShape.1 : lhsShape.0) == resultShape.0)
        precondition((transposeSecond ? rhsShape.0 : rhsShape.1) == resultShape.1)
        precondition((transposeFirst ? lhsShape.0 : lhsShape.1) == (transposeSecond ? rhsShape.1 : rhsShape.0))

        d4lib_igemm(
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
    
    public static func argmax(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        var maxI: UInt = 0
        var maxV: Int32 = 0
        
        d4lib_imaxi(values.pointer(capacity: count), 1, &maxV, &maxI, UInt(count))
        
        return (Int(maxI), maxV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Int32>, count: Int) -> (Int, Int32) {
        var minI: UInt = 0
        var minV: Int32 = 0
        
        d4lib_imini(values.pointer(capacity: count), 1, &minV, &minI, UInt(count))
        
        return (Int(minI), minV)
    }
    
    public static func argmax(values: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> (Int, Int32) {
        var minI: UInt = 0
        var minV: Int32 = 0
        
        d4lib_imaxi(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func argmin(values: UnsafeBufferPointer<Int32>, stride: Int, count: Int) -> (Int, Int32) {
        var minI: UInt = 0
        var minV: Int32 = 0
        
        d4lib_imini(values.pointer(capacity: count), stride, &minV, &minI, UInt(count))
        
        return (Int(minI) / stride, minV)
    }
    
    public static func heaviside(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func copy(values: UnsafeBufferPointer<Int32>, srcStride: Int, result: UnsafeMutableBufferPointer<Int32>, dstStride: Int, count: Int) {
        d4lib_icopy_strided(values.pointer(capacity: count * srcStride), srcStride, result.pointer(capacity: dstStride * count), dstStride, UInt(count))
    }
    
    public static func arange(start: Int32, end: Int32, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_iramp([start], [end / Int32(count)], result.pointer(capacity: count), UInt(count))
    }
    
    public static func max(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_imax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func min(lhs: UnsafeBufferPointer<Int32>, rhs: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        d4lib_imax(lhs.pointer(capacity: count), rhs.pointer(capacity: count), result.pointer(capacity: count), UInt(count))
    }
    
    public static func sin(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func cos(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func tan(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, count: Int) {
        fatalError("\(#function) is unavailable for \(Self.self)")
    }
    
    public static func img2col(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_iimg2col(
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
    
    public static func col2img(values: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Int32>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        d4lib_icol2img(
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


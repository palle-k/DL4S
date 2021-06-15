//
//  CPUNumeric.swift
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

public protocol CPUNumeric: Numeric, Comparable {
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func fill(value: Self, result: UnsafeMutableBufferPointer<Self>, stride: Int, count: Int)
    
    static func transpose(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, srcRows: Int, srcCols: Int)
    
    static func vAdd(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsAdd(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vNeg(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vSub(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vMul(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func vsMul(lhs: UnsafeBufferPointer<Self>, rhs: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func vDiv(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func svDiv(lhs: Self, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)

    static func gemm(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), alpha: Self, beta: Self, transposeFirst: Bool, transposeSecond: Bool)
    
    static func log(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func exp(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func relu(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tanh(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func sqrt(val: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sum(val: UnsafeBufferPointer<Self>, count: Int) -> Self
    static func sum(val: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> Self
    
    static func argmax(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    static func argmin(values: UnsafeBufferPointer<Self>, count: Int) -> (Int, Self)
    
    static func argmax(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self)
    static func argmin(values: UnsafeBufferPointer<Self>, stride: Int, count: Int) -> (Int, Self)
    
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int)
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func copy(values: UnsafeBufferPointer<Self>, srcStride: Int, result: UnsafeMutableBufferPointer<Self>, dstStride: Int, count: Int)
    static func arange(start: Self, end: Self, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func sin(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func cos(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    static func tan(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, count: Int)
    
    static func img2col(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int)
    static func col2img(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int)
    
    static func scatter(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, dst_shape: [Int], axis: Int, ignoreIndex: Int32)
    static func gather(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, src_shape: [Int], axis: Int, ignoreIndex: Int32)
}

//
//  EngineV2.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.03.19.
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


public protocol EngineType {
    associatedtype Device: DeviceType where Device.Engine == Self
    
    static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int)
    static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N)

    
    // MARK: New Engine API
    static func gemm<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool)
    static func broadcastGemm<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool)
    
    static func broadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func reduceSum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    static func reduceMax<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    static func reduceMin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    static func reduceMean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    static func reduceSum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axes: [Int])
    static func reduceMax<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axes: [Int])
    static func reduceMin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axes: [Int])
    static func reduceMean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axes: [Int])
    
    static func expandContext<N>(reduced: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    static func sum<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func mean<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    @discardableResult static func max<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    @discardableResult static func min<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    
    static func exp<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func log<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func sqrt<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func square<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func relu<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func heaviside<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func sin<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func cos<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func tan<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func sinh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func cosh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func tanh<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func permuteAxes<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    static func permuteAxesAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    
    static func subscriptRead<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWrite<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptReadAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWriteAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    static func reverse<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func reverseAdd<N: NumericType>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func stack<N>(buffers: [ShapedBuffer<N, Device>], result: ShapedBuffer<N, Device>, axis: Int)
    static func unstackAdd<N: NumericType>(stacked: ShapedBuffer<N, Device>, add: [ShapedBuffer<N, Device>], result: [ShapedBuffer<N, Device>], axis: Int)
    static func unstack<N: NumericType>(stacked: ShapedBuffer<N, Device>, result: [ShapedBuffer<N, Device>], axis: Int)
    
    static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, Device>)
    
    static func img2col<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
    static func col2img<N: NumericType>(matrix: ShapedBuffer<N, Device>, image: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
}

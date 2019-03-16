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


public protocol EngineTypeV2 {
    associatedtype Device: DeviceType
    // MARK: New Engine API
    
    static func matMul<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func matMulAdd<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func broadcastAdd<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastSub<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastMul<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastDiv<N>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func reduceSum <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    static func reduceMax <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    static func reduceMin <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, axis: Int)
    static func reduceMean<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, axis: Int)
    
    static func sum<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func mean<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    @discardableResult static func max<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    @discardableResult static func min<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>) -> Int
    
    static func exp   <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func log   <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func sqrt  <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func square<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func relu<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func heaviside<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func sin <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func cos <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func tan <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func sinh<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func cosh<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func tanh<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func permuteAxes<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    static func permuteAxesAdd<N>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, arangement: [Int])
    
    static func subscriptRead    <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWrite   <N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptReadAdd <N>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    static func subscriptWriteAdd<N>(values: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, index: [Int?])
    
    static func stack<N>(buffers: [ShapedBuffer<N, Device>], result: ShapedBuffer<N, Device>, axis: Int)
    
    static func conv2d<N>(values: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int))
    static func revConv2d<N>(values: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int))
    static func kernelGradConv2d<N>(values: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int))
    
    static func maxPool2D<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>?, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int))
    static func avgPool2D<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int))
    
    static func revMaxPool2D<N>(values: ShapedBuffer<N, Device>, context: ShapedBuffer<Int32, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int))
    static func revAvgPool2D<N>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int))
}

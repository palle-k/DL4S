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
    
    static func matMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func matMulAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, add: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func broadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    static func broadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>)
    
    static func unbroadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>)
    static func unbroadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>)
    static func unbroadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, rhsGradient: ShapedBuffer<N, Device>)
    static func unbroadcastMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>)
    static func unbroadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>)
    static func unbroadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, rhsGradient: ShapedBuffer<N, Device>)
    
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
    
    static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, Device>)
    
    static func img2col<N: NumericType>(values: ShapedBuffer<N, Device>, result: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
    static func col2img<N: NumericType>(matrix: ShapedBuffer<N, Device>, image: ShapedBuffer<N, Device>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int)
}

extension EngineTypeV2 {
    public static func unbroadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>) {
        if lhs.shape == rhs.shape {
            // Fast path
            Device.Engine.broadcastAdd(lhs: resultGradient, rhs: lhsGradient, result: lhsGradient)
            return
        }
        
        let tmp = Device.Memory.allocateBuffer(withShape: lhs.shape, type: N.self)
        defer {
            Device.Memory.free(tmp)
        }
        
        let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
        let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
            .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
        
        var tmpReducedShape = lhsPadded
        
        for a in lhsReducedAxes.reversed() {
            tmpReducedShape.remove(at: a)
        }
        
        Device.Engine.reduceSum(values: resultGradient, result: tmp.reshaped(to: tmpReducedShape), axes: lhsReducedAxes)
        Device.Engine.broadcastAdd(lhs: tmp, rhs: lhsGradient, result: lhsGradient)
    }
    
    public static func unbroadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>) {
        if lhs.shape == rhs.shape {
            Device.Engine.broadcastAdd(lhs: resultGradient, rhs: lhsGradient, result: lhsGradient)
        } else {
            let tmp = Device.Memory.allocateBuffer(withShape: lhs.shape, type: N.self)
            defer {
                Device.Memory.free(tmp)
            }
            
            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
            
            var tmpReducedShape = lhsPadded
            
            for a in lhsReducedAxes.reversed() {
                tmpReducedShape.remove(at: a)
            }
            
            Device.Engine.reduceSum(values: resultGradient, result: tmp.reshaped(to: tmpReducedShape), axes: lhsReducedAxes)
            Device.Engine.broadcastAdd(lhs: tmp, rhs: lhsGradient, result: lhsGradient)
        }
    }
    
    public static func unbroadcastSub<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, rhsGradient: ShapedBuffer<N, Device>) {
        if lhs.shape == rhs.shape {
            Device.Engine.broadcastSub(lhs: rhsGradient, rhs: resultGradient, result: rhsGradient)
        } else {
            let tmp = Device.Memory.allocateBuffer(withShape: rhs.shape, type: N.self)
            defer {
                Device.Memory.free(tmp)
            }
            
            let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
            let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
            
            var tmpReducedShape = rhsPadded
            
            for a in rhsReducedAxes.reversed() {
                tmpReducedShape.remove(at: a)
            }
            
            Device.Engine.reduceSum(values: resultGradient, result: tmp.reshaped(to: tmpReducedShape), axes: rhsReducedAxes)
            Device.Engine.broadcastSub(lhs: rhsGradient, rhs: tmp, result: rhsGradient)
        }
    }
    
    public static func unbroadcastMul<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>) {
        if lhs.shape == rhs.shape {
            Device.Engine.vMA(lhs: rhs.values, rhs: resultGradient.values, add: lhsGradient.values, result: lhsGradient.values, count: lhsGradient.count)
        } else {
            let tmp1 = Device.Memory.allocateBuffer(withShape: lhs.shape, type: N.self)
            let tmp2 = Device.Memory.allocateBuffer(withShape: resultGradient.shape, type: N.self)
            defer {
                Device.Memory.free(tmp1)
                Device.Memory.free(tmp2)
            }
            
            let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
            let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
                .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
            
            var tmp1reducedShape = lhsPadded
            
            for a in lhsReducedAxes.reversed() {
                tmp1reducedShape.remove(at: a)
            }
            
            Device.Engine.broadcastMul(lhs: rhs, rhs: resultGradient, result: tmp2)
            Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: lhsReducedAxes)
            Device.Engine.broadcastAdd(lhs: tmp1, rhs: lhsGradient, result: lhsGradient)
        }
    }
    
    public static func unbroadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, lhsGradient: ShapedBuffer<N, Device>) {
        let tmp1 = Device.Memory.allocateBuffer(withShape: lhs.shape, type: N.self)
        let tmp2 = Device.Memory.allocateBuffer(withShape: resultGradient.shape, type: N.self)
        defer {
            Device.Memory.free(tmp1)
            Device.Memory.free(tmp2)
        }
        
        let lhsPadded = Array(repeating: 1, count: resultGradient.dim - lhs.dim) + lhs.shape
        let lhsReducedAxes = zip(lhsPadded, resultGradient.shape).enumerated()
            .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
        
        var tmp1reducedShape = lhsPadded
        
        for a in lhsReducedAxes.reversed() {
            tmp1reducedShape.remove(at: a)
        }
        
        Device.Engine.broadcastDiv(lhs: resultGradient, rhs: rhs, result: tmp2)
        Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: lhsReducedAxes)
        Device.Engine.broadcastAdd(lhs: tmp1, rhs: lhsGradient, result: lhsGradient)
    }
    
    public static func unbroadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, Device>, rhs: ShapedBuffer<N, Device>, resultGradient: ShapedBuffer<N, Device>, rhsGradient: ShapedBuffer<N, Device>) {
        let tmp1 = Device.Memory.allocateBuffer(withShape: rhs.shape, type: N.self)
        let tmp2 = Device.Memory.allocateBuffer(withShape: resultGradient.shape, type: N.self)
        defer {
            Device.Memory.free(tmp1)
            Device.Memory.free(tmp2)
        }
        
        let rhsPadded = Array(repeating: 1, count: resultGradient.dim - rhs.dim) + rhs.shape
        let rhsReducedAxes = zip(rhsPadded, resultGradient.shape).enumerated()
            .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
        
        var tmp1reducedShape = rhsPadded
        
        for a in rhsReducedAxes.reversed() {
            tmp1reducedShape.remove(at: a)
        }
        
        Device.Engine.broadcastMul(lhs: resultGradient, rhs: lhs, result: tmp2)
        Device.Engine.broadcastDiv(lhs: tmp2, rhs: rhs, result: tmp2)
        Device.Engine.broadcastDiv(lhs: tmp2, rhs: rhs, result: tmp2)
        Device.Engine.reduceSum(values: tmp2, result: tmp1.reshaped(to: tmp1reducedShape), axes: rhsReducedAxes)
        Device.Engine.broadcastSub(lhs: rhsGradient, rhs: tmp1, result: rhsGradient)
    }
    
}

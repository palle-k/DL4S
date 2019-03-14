//
//  Engine.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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


public protocol DeviceType {
    associatedtype Memory: MemoryOperatorsType where Memory.Device == Self
    associatedtype Engine: EngineType where Engine.Device == Self
}


public protocol MemoryOperatorsType {
    associatedtype Device: DeviceType where Device.Memory == Self
    associatedtype RawBuffer: Hashable
    
    static func allocateBuffer<Element>(withCapacity: Int, type: Element.Type) -> Buffer<Element, Device>
    static func free<Element>(_ buffer: Buffer<Element, Device>)
    
    static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, Device>, count: Int)
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: Buffer<Element, Device>, count: Int)
    static func assign<Element>(from source: Buffer<Element, Device>, to destination: UnsafeMutableBufferPointer<Element>, count: Int)
    
    static func getValue<Element>(from source: Buffer<Element, Device>) -> Element
    static func getSize<Element>(of buffer: Buffer<Element, Device>) -> Int
    
    static func get<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, Device>, with shape: [Int]) -> (Buffer<Element, Device>, Bool, [Int])
    static func set<Element>(slice: [Int?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, Device>, with dstShape: [Int], from source: Buffer<Element, Device>, with sourceShape: [Int])
    
    static func advance<Element>(buffer: Buffer<Element, Device>, by advancement: Int) -> Buffer<Element, Device>
}


public protocol EngineType {
    associatedtype Device: DeviceType where Device.Engine == Self
    
    static func sqrt<N: NumericType>(_ value: N) -> N
    static func log<N: NumericType>(_ value: N) -> N
    static func exp<N: NumericType>(_ value: N) -> N
    static func sin<N: NumericType>(_ value: N) -> N
    static func cos<N: NumericType>(_ value: N) -> N
    static func tan<N: NumericType>(_ value: N) -> N
    static func sinh<N: NumericType>(_ value: N) -> N
    static func cosh<N: NumericType>(_ value: N) -> N
    static func tanh<N: NumericType>(_ value: N) -> N
    static func pow<N: NumericType>(base: N, exponent: N) -> N
    
    static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int)
    static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, stride: Int, count: Int)
    static func transpose<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, srcRows: Int, srcCols: Int)
    static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vsAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int)
    static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vMA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vsMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int)
    static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func vSquare<N: NumericType>(values: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func matMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsRows: Int, lhsCols: Int, rhsCols: Int)
    static func dot<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, count: Int) -> N
    static func vMulSA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: N, result: Buffer<N, Device>, count: Int)
    static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func log<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func exp<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func relu<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func isPositive<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func tanh<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func sqrt<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int)
    static func sum<N: NumericType>(val: Buffer<N, Device>, count: Int) -> N
    static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N)
    static func conv2d<N: NumericType>(input: Buffer<N, Device>, filter: Buffer<N, Device>, result: Buffer<N, Device>, width: Int, height: Int, batchSize: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int)
    static func permuteAxes<N: NumericType>(input: Buffer<N, Device>, arangement: [Int], shape: [Int], destination: Buffer<N, Device>)
    static func permuteAxesAdd<N: NumericType>(input: Buffer<N, Device>, arangement: [Int], shape: [Int], add: Buffer<N, Device>, destination: Buffer<N, Device>)
    
    static func maxPool2d<N>(input: Buffer<N, Device>, result: Buffer<N, Device>, resultContext: Buffer<Int32, Device>, inputSize: (batchSize: Int, depth: Int, height: Int, width: Int), kernelSize: (height: Int, width: Int), stride: (vertical: Int, horizontal: Int)) where N : NumericType
    static func maxPool2DRevAdd<N>(pooled: Buffer<N, Device>, poolCtx: Buffer<Int32, Device>, add: Buffer<Int32, Device>, target: Buffer<Int32, Device>, inputSize: (batchSize: Int, depth: Int, height: Int, width: Int), kernelSize: (height: Int, width: Int), stride: (vertical: Int, horizontal: Int))
}


extension EngineType {
    public static func sqrt<N: NumericType>(_ value: N) -> N {
        return value.sqrt()
    }
    
    public static func log<N: NumericType>(_ value: N) -> N {
        return value.log()
    }
    
    public static func exp<N: NumericType>(_ value: N) -> N {
        return value.exp()
    }
    
    public static func sin<N: NumericType>(_ value: N) -> N {
        return value.sin()
    }
    
    public static func cos<N: NumericType>(_ value: N) -> N {
        return value.cos()
    }
    
    public static func tan<N: NumericType>(_ value: N) -> N {
        return value.tan()
    }
    
    public static func sinh<N: NumericType>(_ value: N) -> N {
        return value.sinh()
    }
    
    public static func cosh<N: NumericType>(_ value: N) -> N {
        return value.cosh()
    }
    
    public static func tanh<N: NumericType>(_ value: N) -> N {
        return value.tanh()
    }
    
    public static func pow<N: NumericType>(base: N, exponent: N) -> N {
        return N.pow(base: base, exponent: exponent)
    }
}

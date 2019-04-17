//
//  CPUEngine.swift
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
import Metal


public struct GPU: DeviceType {
    public typealias Memory = VRAMAllocator
    public typealias Engine = GPUEngine
    
    fileprivate static var device: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not available.")
        }
        print("Running on \(device.name)")
        return device
    }()
}

public struct VRAMAllocator: MemoryOperatorsType {
    public typealias RawBuffer = VRAMBuffer
    public typealias Device = GPU
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, GPU> where Element : NumericType {
        let stride = MemoryLayout<Element>.stride
        guard let buffer = Device.device.makeBuffer(length: stride * capacity, options: .storageModePrivate) else {
            fatalError("Could not allocate memory")
        }
        
        return Buffer<Element, GPU>(memory: VRAMBuffer(buffer: buffer, offset: 0))
    }
    
    public static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, GPU> where Element : NumericType {
        let count = shape.reduce(1, *)
        return ShapedBuffer(values: allocateBuffer(withCapacity: count, type: Element.self), shape: shape)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, GPU>) where Element : NumericType {
        // Noop, MTLBuffer is reference counted
    }
    
    public static func free<Element>(_ buffer: ShapedBuffer<Element, GPU>) where Element : NumericType {
        // Noop, MTLBuffer is reference counted
    }
    
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, GPU>, count: Int) where Element : NumericType {
        // TODO
        fatalError("TODO")
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: Buffer<Element, GPU>, count: Int) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func getValue<Element>(from source: Buffer<Element, GPU>) -> Element where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, GPU>) -> Int where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) where Element : NumericType {
        fatalError("TODO")
    }
    
    public static func advance<Element>(buffer: Buffer<Element, GPU>, by advancement: Int) -> Buffer<Element, GPU> where Element : NumericType {
        fatalError("TODO")
    }
}

public struct GPUEngine: EngineType {
    public typealias Device = GPU
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, stride: Int, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func transpose<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, srcRows: Int, srcCols: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vSquare<N: NumericType>(values: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func matMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func dot<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, count: Int) -> N {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMulSA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: N, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func log<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func exp<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func relu<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func isPositive<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func tanh<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, Device>, count: Int) -> N {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func copysign<N: NumericType>(values: Buffer<N, Device>, signs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func conv2d<N>(input: Buffer<N, GPU>, filter: Buffer<N, GPU>, result: Buffer<N, GPU>, width: Int, height: Int, batchSize: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxes<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], destination: Buffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxesAdd<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], add: Buffer<N, GPU>, destination: Buffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func maxPool2d<N>(input: Buffer<N, GPU>, result: Buffer<N, GPU>, resultContext: Buffer<Int32, GPU>, inputSize: (batchSize: Int, depth: Int, height: Int, width: Int), kernelSize: (height: Int, width: Int), stride: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func maxPool2DRevAdd<N>(pooled: Buffer<N, GPU>, poolCtx: Buffer<Int32, GPU>, add: Buffer<Int32, GPU>, target: Buffer<Int32, GPU>, inputSize: (batchSize: Int, depth: Int, height: Int, width: Int), kernelSize: (height: Int, width: Int), stride: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
}


public struct VRAMBuffer: Hashable {
    public static func == (lhs: VRAMBuffer, rhs: VRAMBuffer) -> Bool {
        return lhs.buffer.hash == rhs.buffer.hash && lhs.offset == rhs.offset
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(buffer.hash)
        hasher.combine(offset)
    }
    
    var buffer: MTLBuffer
    var offset: Int
}

extension GPUEngine: EngineTypeV2 {
    
    public static func matMul<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func matMulAdd<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastAdd<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastMul<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastDiv<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func mean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func max<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func min<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func square<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func relu<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func heaviside<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptReadAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptWriteAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, GPU>], result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func arange<N>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func img2col<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func col2img<N>(matrix: ShapedBuffer<N, GPU>, image: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func conv2d<N>(values: ShapedBuffer<N, GPU>, filters: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, padding: Int, stride: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func revConv2d<N>(values: ShapedBuffer<N, GPU>, filters: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, strides: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func kernelGradConv2d<N>(values: ShapedBuffer<N, GPU>, filters: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, strides: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func maxPool2D<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func avgPool2D<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func revMaxPool2D<N>(values: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: ShapedBuffer<N, GPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func revAvgPool2D<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func exp<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func log<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }

    public static func sqrt<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func cos<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func tan<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sinh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func cosh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func tanh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
}

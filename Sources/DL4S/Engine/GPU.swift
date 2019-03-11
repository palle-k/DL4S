//
//  CPUEngine.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation
import Metal


public struct GPU: Device {
    public typealias MemoryOperatorType = VRAMAllocator
    public typealias EngineType = GPUEngine
    
    fileprivate static var device: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not available.")
        }
        print("Running on \(device.name)")
        return device
    }()
}

public struct VRAMAllocator: MemoryOperators {
    public typealias RawBufferType = MTLBuffer
    public typealias DeviceType = GPU
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, GPU> where Element : NumericType {
        let stride = MemoryLayout<Element>.stride
        guard let buffer = DeviceType.device.makeBuffer(length: stride * capacity, options: .storageModePrivate) else {
            fatalError("Could not allocate memory")
        }
        
        return Buffer<Element, GPU>(memory: buffer)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, GPU>) where Element : NumericType {
        // Noop, MTLBuffer is reference counted
    }
    
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, GPU>, count: Int) where Element : NumericType {
        // TODO
        fatalError("TODO")
    }
}

public struct GPUEngine: Engine {
    public typealias DeviceType = GPU
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, stride: Int, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func transpose<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, srcRows: Int, srcCols: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vNeg<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vSub<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vSquare<N: NumericType>(values: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func matMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func dot<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, count: Int) -> N {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vMulSA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: N, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func log<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func exp<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func relu<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func isPositive<N: NumericType>(val: Buffer<N, DeviceType>, result: UnsafeMutablePointer<N>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func tanh<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, DeviceType>, count: Int) -> N {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func copysign<N: NumericType>(values: Buffer<N, DeviceType>, signs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, DeviceType>, count: Int) -> (Int, N) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func conv2d<N: NumericType>(input: Buffer<N, DeviceType>, filter: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) {
        fatalError("\(#function) not available for GPU")
    }
}

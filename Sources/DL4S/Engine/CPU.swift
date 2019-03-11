//
//  CPU.swift
//  DL4S
//
//  Created by Palle Klewitz on 11.03.19.
//

import Foundation
import Accelerate


struct CPU: Device {
    typealias AllocatorType = _CPUAllocator
    typealias EngineType = CPUEngine
}

struct _CPUAllocator: Allocator {
    typealias RawBufferType = UnsafeMutableRawBufferPointer
    typealias DeviceType = CPU
    
    static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, CPU> where Element : NumericType {
        let stride = MemoryLayout<Element>.stride
        let alignment = max(MemoryLayout<Element>.alignment, 16)
        
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: stride * capacity, alignment: alignment)
        
        return Buffer<Element, CPU>(memory: buffer)
    }
    
    static func free<Element>(_ buffer: Buffer<Element, CPU>) where Element : NumericType {
        buffer.memory.deallocate()
    }
    
}

struct CPUEngine: Engine {
    typealias DeviceType = CPU
    
    static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, stride: Int, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func transpose<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, srcRows: Int, srcCols: Int) {
        N.transpose(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), srcRows: srcRows, srcCols: srcCols)
    }
    
    static func vAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vsAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vsAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vNeg<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vNeg(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vSub<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vSub(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vMA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vMA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vsMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vsMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vDiv<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vDiv(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.svDiv(lhs: lhs, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vSquare<N: NumericType>(values: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vSquare(values: values.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func matMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        N.matMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsRows: lhsRows, lhsCols: lhsCols, rhsCols: rhsCols)
    }
    
    static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        N.matMulAddInPlace(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsShape: lhsShape, rhsShape: rhsShape, resultShape: resultShape, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
    }
    
    static func dot<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, count: Int) -> N {
        return N.dot(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    static func vMulSA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vMulSA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vsMulVAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func log<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.log(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func exp<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.exp(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func relu<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.relu(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func isPositive<N: NumericType>(val: Buffer<N, DeviceType>, result: UnsafeMutablePointer<N>, count: Int) {
        fatalError("isPositive operator not available for CPU device")
    }
    
    static func tanh<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.tanh(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func sqrt<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.sqrt(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func sum<N: NumericType>(val: Buffer<N, DeviceType>, count: Int) -> N {
        return N.sum(val: val.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    static func copysign<N: NumericType>(values: Buffer<N, DeviceType>, signs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.copysign(values: values.memory.bindMemory(to: N.self).immutable, signs: signs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    static func argmax<N: NumericType>(values: Buffer<N, DeviceType>, count: Int) -> (Int, N) {
        return N.argmax(values: values.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    static func conv2d<N: NumericType>(input: Buffer<N, DeviceType>, filter: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) {
        N.conv2d(input: input.memory.bindMemory(to: N.self).immutable, filter: filter.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), width: width, height: height, kernelWidth: kernelWidth, kernelHeight: kernelHeight, kernelDepth: kernelDepth, kernelCount: kernelCount)
    }
    
}

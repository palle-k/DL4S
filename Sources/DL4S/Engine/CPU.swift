//
//  CPU.swift
//  DL4S
//
//  Created by Palle Klewitz on 11.03.19.
//

import Foundation
import Accelerate


public struct CPU: Device {
    public typealias MemoryOperatorType = CPUMemoryOperators
    public typealias EngineType = CPUEngine
}

public struct CPUMemoryOperators: MemoryOperators {
    public typealias RawBufferType = UnsafeMutableRawBufferPointer
    public typealias DeviceType = CPU
    
    static func strides(from shape: [Int]) -> [Int] {
        let dim = shape.count
        
        if dim == 0 {
            return []
        }
        
        var str = [Int](repeating: 1, count: dim)
        for i in (0 ..< dim - 1).reversed() {
            str[i] = str[i + 1] * shape[i + 1]
        }
        return str
    }
    
    static func linearIndex(from index: [Int], shape: [Int]) -> Int {
        let strides = MemoryOps.strides(from: shape)
        return zip(index, strides).map(*).reduce(0, +)
    }
    
    static func index(from linearIndex: Int, shape: [Int]) -> [Int] {
        let strides = MemoryOps.strides(from: shape)
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim}
    }
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, CPU> where Element : NumericType {
        let stride = MemoryLayout<Element>.stride
        let alignment = max(MemoryLayout<Element>.alignment, 16)
        
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: stride * capacity, alignment: alignment)
        
        return Buffer<Element, CPU>(memory: buffer)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, CPU>) where Element : NumericType {
        buffer.memory.deallocate()
    }
    
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, CPU>, count: Int) where Element : NumericType {
        destination.memory.bindMemory(to: Element.self).assign(from: source, count: count)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: Buffer<Element, CPU>, count: Int) where Element : NumericType {
        destination.memory.bindMemory(to: Element.self).assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) where Element : NumericType {
        destination.assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
    }
    
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) where Element : NumericType {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = CPUMemoryOperators.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            let advanced = buffer.memory.bindMemory(to: Element.self).advanced(by: offset)
            let advancedRaw = UnsafeMutableRawBufferPointer(advanced)
            return (Buffer<Element, CPU>(memory: advancedRaw), false, Array(shape.dropFirst(nonNilIndices.count)))
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
            
            recursiveRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) where Element : NumericType {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let strides = CPUMemoryOperators.strides(from: shape)
        
        let padded = slice + [Range<Int>?](repeating: nil, count: shape.count - slice.count)
        
        let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        let flattenedResultShape = resultShape.compactMap {$0}
        
        let resultCount = flattenedResultShape.reduce(1, *)
        let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
        
        recursiveRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
        
        return (resultBuffer, true, flattenedResultShape)
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) where Element : NumericType {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        recursiveWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) where Element : NumericType {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Range<Int>?](repeating: nil, count: dstShape.count - slice.count)
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        
        recursiveWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func getValue<Element>(from source: Buffer<Element, CPU>) -> Element where Element : NumericType {
        return source.memory.bindMemory(to: Element.self).pointee
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, CPU>) -> Int where Element : NumericType {
        return buffer.memory.bindMemory(to: Element.self).count
    }
}

public struct CPUEngine: Engine {
    public typealias DeviceType = CPU
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, DeviceType>, stride: Int, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func transpose<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, srcRows: Int, srcCols: Int) {
        N.transpose(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), srcRows: srcRows, srcCols: srcCols)
    }
    
    public static func vAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vsAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vNeg<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vNeg(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vSub<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vSub(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vMA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vMA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vsMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vDiv(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.svDiv(lhs: lhs, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vSquare<N: NumericType>(values: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vSquare(values: values.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func matMul<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        N.matMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsRows: lhsRows, lhsCols: lhsCols, rhsCols: rhsCols)
    }
    
    public static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        N.matMulAddInPlace(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsShape: lhsShape, rhsShape: rhsShape, resultShape: resultShape, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
    }
    
    public static func dot<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, count: Int) -> N {
        return N.dot(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func vMulSA<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: Buffer<N, DeviceType>, add: N, result: Buffer<N, DeviceType>, count: Int) {
        N.vMulSA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, DeviceType>, rhs: N, add: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.vsMulVAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func log<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.log(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func exp<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.exp(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func relu<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.relu(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func isPositive<N: NumericType>(val: Buffer<N, DeviceType>, result: UnsafeMutablePointer<N>, count: Int) {
        fatalError("isPositive operator not available for CPU device")
    }
    
    public static func tanh<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.tanh(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.sqrt(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, DeviceType>, count: Int) -> N {
        return N.sum(val: val.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func copysign<N: NumericType>(values: Buffer<N, DeviceType>, signs: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, count: Int) {
        N.copysign(values: values.memory.bindMemory(to: N.self).immutable, signs: signs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, DeviceType>, count: Int) -> (Int, N) {
        return N.argmax(values: values.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func conv2d<N: NumericType>(input: Buffer<N, DeviceType>, filter: Buffer<N, DeviceType>, result: Buffer<N, DeviceType>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) {
        N.conv2d(input: input.memory.bindMemory(to: N.self).immutable, filter: filter.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), width: width, height: height, kernelWidth: kernelWidth, kernelHeight: kernelHeight, kernelDepth: kernelDepth, kernelCount: kernelCount)
    }
    
}

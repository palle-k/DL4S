//
//  CPU.swift
//  DL4S
//
//  Created by Palle Klewitz on 11.03.19.
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


public struct CPU: DeviceType {
    public typealias Memory = CPUMemoryOperators
    public typealias Engine = CPUEngine
}

public struct CPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = UnsafeMutableRawBufferPointer
    public typealias Device = CPU
    
    static var traceAllocations: Bool = false {
        didSet {
            allocations.removeAll()
        }
    }
    private static var allocations: [UnsafeMutableRawBufferPointer: [String]] = [:]
    private static let sema = DispatchSemaphore(value: 1)
    
    @inline(__always)
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
        let strides = CPUMemoryOperators.strides(from: shape)
        return zip(index, strides).map(*).reduce(0, +)
    }
    
    static func index(from linearIndex: Int, shape: [Int]) -> [Int] {
        let strides = CPUMemoryOperators.strides(from: shape)
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim}
    }
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, CPU> {
        let stride = MemoryLayout<Element>.stride
        let alignment = max(MemoryLayout<Element>.alignment, 16)
        
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: stride * capacity, alignment: alignment)
        
        if traceAllocations {
            sema.wait()
            let trace = Thread.callStackSymbols
            allocations[buffer] = trace
            sema.signal()
            
            DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(5)) {
                sema.wait()
                if let trace = allocations[buffer] {
                    print("[ALLOC TRACE]: buffer of size \(capacity) not freed after 3 seconds.")
                    print("[ALLOC TRACE] [begin callstack]")
                    print(trace.joined(separator: "\n"))
                    print("[ALLOC TRACE] [end callstack]")
                }
                sema.signal()
            }
        }
        
        return Buffer<Element, CPU>(memory: buffer)
    }
    
    public static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, CPU> {
        let count = shape.reduce(1, *)
        return ShapedBuffer(values: allocateBuffer(withCapacity: count, type: Element.self), shape: shape)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, CPU>) {
        if traceAllocations {
            sema.wait()
            allocations.removeValue(forKey: buffer.memory)
            sema.signal()
        }
        DispatchQueue.global().async {
            buffer.memory.deallocate()
        }
    }
    
    public static func free<Element>(_ buffer: ShapedBuffer<Element, CPU>) {
        free(buffer.values)
    }
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, CPU>, count: Int) {
        // destination.memory.bindMemory(to: Element.self).assign(from: source, count: count)
        memcpy(destination.memory.baseAddress!, source.baseAddress!, count * MemoryLayout<Element>.stride)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: Buffer<Element, CPU>, count: Int) {
        // destination.memory.bindMemory(to: Element.self).assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
        memcpy(destination.memory.baseAddress!, source.memory.baseAddress!, count * MemoryLayout<Element>.stride)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        // destination.assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
        memcpy(destination.baseAddress!, source.memory.baseAddress!, count * MemoryLayout<Element>.stride)
    }
    
    @inline(__always)
    @_specialize(where Element == Float)
    @_specialize(where Element == Int32)
    @_specialize(where Element == Double)
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        // Prevent unneccessary copies when index ends with nil
        let slice = slice.reversed().drop(while: {$0 == nil}).reversed()
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = CPUMemoryOperators.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            let resultShape = Array(shape.dropFirst(nonNilIndices.count))
            
            let bound = buffer.memory
                .bindMemory(to: Element.self)
            let advanced = UnsafeMutableBufferPointer(
                rebasing: bound.advanced(by: offset).prefix(resultShape.reduce(1, *))
            )
            let advancedRaw = UnsafeMutableRawBufferPointer(advanced)
            return (Buffer<Element, CPU>(memory: advancedRaw), false, resultShape)
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
            
            iterativeRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let strides = CPUMemoryOperators.strides(from: shape)
        
        let padded = slice + [Range<Int>?](repeating: nil, count: shape.count - slice.count)
        
        let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        
        let resultCount = resultShape.reduce(1, *)
        let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
        
        recursiveRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
        
        return (resultBuffer, true, resultShape)
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        let countDelta = dstShape.count - slice.filter {$0 != nil}.count
        precondition(sourceShape.count == countDelta, "Dimensionality of source must be equal to dimensionality of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        iterativeWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count, "Dimensionality of source must be equal to dimensionality of destination")
        
        let padded = slice + [Range<Int>?](repeating: nil, count: dstShape.count - slice.count)
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        
        recursiveWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func getValue<Element>(from source: Buffer<Element, CPU>) -> Element {
        return source.memory.bindMemory(to: Element.self).pointee
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, CPU>) -> Int {
        return buffer.memory.bindMemory(to: Element.self).count
    }
    
    public static func advance<Element>(buffer: Buffer<Element, CPU>, by advancement: Int) -> Buffer<Element, CPU> {
        return Buffer<Element, CPU>(
            memory: UnsafeMutableRawBufferPointer(
                buffer.memory
                    .bindMemory(to: Element.self)
                    .advanced(by: advancement)
            )
        )
    }
    
    public static func setPointee<Element>(of buffer: Buffer<Element, CPU>, to newValue: Element) {
        buffer.pointer.pointee = newValue
    }
}

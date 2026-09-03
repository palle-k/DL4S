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
#if DL4S_TRACE_ALLOCATIONS
import Synchronization
#endif


public struct CPU: DeviceType {
    public typealias Memory = CPUMemoryOperators
    public typealias Engine = CPUEngine
}

public struct CPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = UnsafeMutableRawBufferPointer
    public typealias Device = CPU
    
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
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> MutableBuffer<Element, CPU> {
        let stride = MemoryLayout<Element>.stride
        let alignment = max(MemoryLayout<Element>.alignment, 16)
        
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: stride * capacity, alignment: alignment)
        #if DL4S_TRACE_ALLOCATIONS
        recordAllocation(of: buffer, capacity: capacity)
        #endif
        return MutableBuffer<Element, CPU>(memory: buffer)
    }
    
    public static func free<Element>(_ buffer: MutableBuffer<Element, CPU>) {
        #if DL4S_TRACE_ALLOCATIONS
        recordFree(of: buffer.memory)
        #endif
        buffer.memory.deallocate()
    }
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: MutableBuffer<Element, CPU>, count: Int) {
        // destination.memory.bindMemory(to: Element.self).assign(from: source, count: count)
        memcpy(destination.memory.baseAddress!, source.baseAddress!, count * MemoryLayout<Element>.stride)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: MutableBuffer<Element, CPU>, count: Int) {
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
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (MutableBuffer<Element, CPU>, Bool, [Int]) {
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
            return (MutableBuffer<Element, CPU>(memory: advancedRaw), false, resultShape)
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
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (MutableBuffer<Element, CPU>, Bool, [Int]) {
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
    
    public static func set<Element>(slice: [Int?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        let countDelta = dstShape.count - slice.filter {$0 != nil}.count
        precondition(sourceShape.count == countDelta, "Dimensionality of source must be equal to dimensionality of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        iterativeWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
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
        return Buffer<Element, CPU>(memory: advance(memory: buffer.memory, by: advancement, type: Element.self))
    }
    
    public static func advance<Element>(buffer: MutableBuffer<Element, CPU>, by advancement: Int) -> MutableBuffer<Element, CPU> {
        return MutableBuffer<Element, CPU>(memory: advance(memory: buffer.memory, by: advancement, type: Element.self))
    }
    
    private static func advance<Element>(memory: UnsafeMutableRawBufferPointer, by advancement: Int, type: Element.Type) -> UnsafeMutableRawBufferPointer {
        return UnsafeMutableRawBufferPointer(
            memory
                .bindMemory(to: Element.self)
                .advanced(by: advancement)
        )
    }
    
    public static func setPointee<Element>(of buffer: MutableBuffer<Element, CPU>, to newValue: Element) {
        buffer.pointer.pointee = newValue
    }
}

#if DL4S_TRACE_ALLOCATIONS
// MARK: Allocation tracing
//
// Compile with `-Xswiftc -DDL4S_TRACE_ALLOCATIONS` to enable this feature.

struct AllocationTraceState: Sendable {
    /// Whether allocate and free record call stacks.
    var isEnabled = false
    
    /// Call stacks of live allocations, by the address of the buffer.
    var callStacks: [UInt: [String]] = [:]
}

public extension CPUMemoryOperators {
    /// Time in seconds after which a live allocation is reported as a possible leak.
    static let allocationTraceReportDelaySeconds = 5
    
    internal static let allocationTraceState = Mutex(AllocationTraceState())
    
    /// Switches allocation tracing on or off.
    ///
    /// While tracing is on, every allocation records its call stack. A buffer that is not freed within
    /// `allocationTraceReportDelaySeconds` is printed with the call stack of its allocation.
    /// Switching tracing on or off discards the recorded call stacks.
    ///
    /// - Parameter enabled: Whether to trace allocations.
    static func setAllocationTracing(_ enabled: Bool) {
        allocationTraceState.withLock { state in
            state.isEnabled = enabled
            state.callStacks.removeAll()
        }
    }
    
    /// Number of allocations that are traced and not yet freed.
    static var tracedAllocationCount: Int {
        allocationTraceState.withLock { $0.callStacks.count }
    }
    
    internal static func recordAllocation(of buffer: UnsafeMutableRawBufferPointer, capacity: Int) {
        let address = UInt(bitPattern: buffer.baseAddress!)
        let isEnabled = allocationTraceState.withLock { state in
            guard state.isEnabled else {
                return false
            }
            state.callStacks[address] = Thread.callStackSymbols
            return true
        }
        guard isEnabled else {
            return
        }
        
        DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(allocationTraceReportDelaySeconds)) {
            let callStack = allocationTraceState.withLock { $0.callStacks[address] }
            guard let callStack else {
                return
            }
            print("[ALLOC TRACE]: buffer of size \(capacity) not freed after \(allocationTraceReportDelaySeconds) seconds.")
            print("[ALLOC TRACE] [begin callstack]")
            print(callStack.joined(separator: "\n"))
            print("[ALLOC TRACE] [end callstack]")
        }
    }
    
    internal static func recordFree(of buffer: UnsafeMutableRawBufferPointer) {
        let address = UInt(bitPattern: buffer.baseAddress!)
        allocationTraceState.withLock { state in
            _ = state.callStacks.removeValue(forKey: address)
        }
    }
}
#endif

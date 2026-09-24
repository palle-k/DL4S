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
    public typealias FusedOperations = CPUFusedOperations
}

/// Fused operations of the CPU.
///
/// The CPU has fused kernels for the activations, softmax, normalization, convolution and pooling, attention, and dropout.
/// The kernels process the data in blocks or rows that fit into the cache and use the accelerated primitives of ``CPUNumeric``.
/// The other operations, and shapes that the kernels do not support, use the default implementations.
public struct CPUFusedOperations: FusedOperationsType {
    public typealias Device = CPU
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
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim }
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
        var sliceCount = slice.count
        while sliceCount > 0, slice[sliceCount - 1] == nil {
            sliceCount -= 1
        }
        let strides = CPUMemoryOperators.strides(from: shape)

        if slice.prefix(sliceCount).allSatisfy({ $0 != nil }) {
            // Simple offset into storage
            var offset = 0
            for axis in 0 ..< sliceCount {
                offset += slice[axis]! * strides[axis]
            }
            let resultShape = Array(shape.dropFirst(sliceCount))

            let bound = buffer.memory
                .bindMemory(to: Element.self)
            let advanced = UnsafeMutableBufferPointer(
                rebasing: bound.advanced(by: offset).prefix(resultShape.reduce(1, *)),
            )
            let advancedRaw = UnsafeMutableRawBufferPointer(advanced)
            return (MutableBuffer<Element, CPU>(memory: advancedRaw), false, resultShape)
        } else {
            let padded = Array(slice.prefix(sliceCount)) + [Int?](repeating: nil, count: shape.count - sliceCount)

            let resultShape = zip(padded, shape).map { el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap(\.self)

            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)

            iterativeRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)

            return (resultBuffer, true, flattenedResultShape)
        }
    }

    public static func get<Element>(slice: [CountableRange<Int>?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (MutableBuffer<Element, CPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")

        let strides = CPUMemoryOperators.strides(from: shape)
        let ranges = shape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< shape[axis] }
        let resultShape = ranges.map(\.count)
        let offset = zip(ranges, strides).map { $0.lowerBound * $1 }.reduce(0, +)

        let resultBuffer = allocateBuffer(withCapacity: resultShape.reduce(1, *), type: Element.self)
        copyRegion(
            from: buffer.memory.bindMemory(to: Element.self).baseAddress! + offset,
            strides: strides,
            to: resultBuffer.memory.bindMemory(to: Element.self).baseAddress!,
            strides: CPUMemoryOperators.strides(from: resultShape),
            shape: resultShape,
        )
        return (resultBuffer, true, resultShape)
    }

    /// Copies the elements of a region between two buffers, one row of the region at a time.
    ///
    /// - Parameters:
    ///   - source: First element of the region in the source.
    ///   - sourceStrides: Strides of the source.
    ///   - destination: First element of the region in the destination.
    ///   - destinationStrides: Strides of the destination.
    ///   - shape: Shape of the region. The last axis must be contiguous in the source and the destination.
    private static func copyRegion<Element>(from source: UnsafeMutablePointer<Element>, strides sourceStrides: [Int], to destination: UnsafeMutablePointer<Element>, strides destinationStrides: [Int], shape: [Int]) {
        guard let rowLength = shape.last else {
            destination.pointee = source.pointee
            return
        }
        guard shape.allSatisfy({ $0 > 0 }) else {
            return
        }
        StridedIteration.forEachOffset(shape: Array(shape.dropLast()), strides: Array(sourceStrides.dropLast()), Array(destinationStrides.dropLast())) { sourceOffset, destinationOffset in
            (destination + destinationOffset).update(from: source + sourceOffset, count: rowLength)
        }
    }

    public static func set<Element>(slice: [Int?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        let countDelta = dstShape.count - slice.filter { $0 != nil }.count
        precondition(sourceShape.count == countDelta, "Dimensionality of source must be equal to dimensionality of destination minus number of knowns in slice")

        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)

        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        iterativeWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }

    public static func set<Element>(slice: [Range<Int>?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count, "Dimensionality of source must be equal to dimensionality of destination")

        let strides = CPUMemoryOperators.strides(from: dstShape)
        let ranges = dstShape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< dstShape[axis] }
        let regionShape = ranges.map(\.count)
        precondition(regionShape == sourceShape, "Shape of source must be equal to the shape of the slice")
        let offset = zip(ranges, strides).map { $0.lowerBound * $1 }.reduce(0, +)

        copyRegion(
            from: UnsafeMutablePointer(mutating: source.memory.bindMemory(to: Element.self).baseAddress!),
            strides: CPUMemoryOperators.strides(from: regionShape),
            to: buffer.memory.bindMemory(to: Element.self).baseAddress! + offset,
            strides: strides,
            shape: regionShape,
        )
    }

    public static func getValue<Element>(from source: Buffer<Element, CPU>) -> Element {
        source.memory.bindMemory(to: Element.self).pointee
    }

    public static func getSize<Element>(of buffer: Buffer<Element, CPU>) -> Int {
        buffer.memory.bindMemory(to: Element.self).count
    }

    public static func advance<Element>(buffer: Buffer<Element, CPU>, by advancement: Int) -> Buffer<Element, CPU> {
        Buffer<Element, CPU>(memory: advance(memory: buffer.memory, by: advancement, type: Element.self))
    }

    public static func advance<Element>(buffer: MutableBuffer<Element, CPU>, by advancement: Int) -> MutableBuffer<Element, CPU> {
        MutableBuffer<Element, CPU>(memory: advance(memory: buffer.memory, by: advancement, type: Element.self))
    }

    private static func advance<Element>(memory: UnsafeMutableRawBufferPointer, by advancement: Int, type: Element.Type) -> UnsafeMutableRawBufferPointer {
        UnsafeMutableRawBufferPointer(
            memory
                .bindMemory(to: Element.self)
                .advanced(by: advancement),
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

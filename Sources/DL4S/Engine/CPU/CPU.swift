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

/// Memory of the CPU, which is freed when the last buffer that refers to it is released.
final class CPUStorage {
    let bytes: UnsafeMutableRawBufferPointer

    init(byteCount: Int, alignment: Int) {
        bytes = UnsafeMutableRawBufferPointer.allocate(byteCount: byteCount, alignment: alignment)
        #if DL4S_TRACE_ALLOCATIONS
        CPUMemoryOperators.recordAllocation(of: bytes)
        #endif
    }

    deinit {
        #if DL4S_TRACE_ALLOCATIONS
        CPUMemoryOperators.recordFree(of: bytes)
        #endif
        bytes.deallocate()
    }
}

// `@unchecked Sendable`: Tensors write their storage only through `Tensor.mutableValues`, which copies it first when
// another buffer refers to it, so no two threads write or read and write the same memory.
/// A region of memory of the CPU.
///
/// Views of a tensor refer to the storage of the tensor with an offset.
public struct CPUBuffer: @unchecked Sendable {
    /// The storage that owns the memory, or nil for memory that another object keeps alive, such as the host memory
    /// of a GPU buffer.
    var storage: CPUStorage?

    /// The region in memory.
    let bytes: UnsafeMutableRawBufferPointer
}

public struct CPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = CPUBuffer
    public typealias Device = CPU

    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> MutableBuffer<Element, CPU> {
        let storage = CPUStorage(byteCount: MemoryLayout<Element>.stride * capacity, alignment: max(MemoryLayout<Element>.alignment, 16))
        return MutableBuffer(memory: CPUBuffer(storage: storage, bytes: storage.bytes))
    }

    public static func isUniquelyReferenced(_ buffer: inout CPUBuffer) -> Bool {
        isKnownUniquelyReferenced(&buffer.storage)
    }

    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: MutableBuffer<Element, CPU>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        precondition(count <= source.count && byteCount <= destination.bytes.count, "The copy exceeds a buffer.")
        guard byteCount > 0 else {
            return
        }
        memcpy(destination.bytes.baseAddress!, source.baseAddress!, byteCount)
    }

    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: MutableBuffer<Element, CPU>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        precondition(byteCount <= source.bytes.count && byteCount <= destination.bytes.count, "The copy exceeds a buffer.")
        guard byteCount > 0 else {
            return
        }
        memcpy(destination.bytes.baseAddress!, source.bytes.baseAddress!, byteCount)
    }

    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        precondition(byteCount <= source.bytes.count && count <= destination.count, "The copy exceeds a buffer.")
        guard byteCount > 0 else {
            return
        }
        memcpy(destination.baseAddress!, source.bytes.baseAddress!, byteCount)
    }

    @inline(__always)
    @_specialize(where Element == Float)
    @_specialize(where Element == Int32)
    @_specialize(where Element == Double)
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> MutableShapedBuffer<Element, CPU> {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")

        // An index that ends with nil selects the same elements as the index without the trailing nils.
        var sliceCount = slice.count
        while sliceCount > 0, slice[sliceCount - 1] == nil {
            sliceCount -= 1
        }
        let strides = ShapeUtil.contiguousStrides(of: shape)
        let prefix = slice.prefix(sliceCount)

        if prefix.allSatisfy({ $0 != nil }) {
            // The elements are contiguous, so the result is a view of the storage.
            let offset = zip(prefix, strides).map { $0! * $1 }.reduce(0, +)
            let resultShape = Array(shape.dropFirst(sliceCount))
            let view = advance(buffer: MutableBuffer<Element, CPU>(memory: buffer.memory), by: offset, count: resultShape.reduce(1, *))
            return MutableShapedBuffer(values: view, shape: resultShape)
        }
        let ranges = shape.indices.map { axis in (axis < slice.count ? slice[axis] : nil).map { $0 ..< $0 + 1 } ?? 0 ..< shape[axis] }
        let resultShape = zip(slice + [Int?](repeating: nil, count: shape.count - slice.count), shape).compactMap { index, size in index == nil ? size : nil }
        let result = allocateBuffer(withCapacity: resultShape.reduce(1, *), type: Element.self)
        copyRegion(of: buffer, shape: shape, ranges: ranges, to: result, inside: false)
        return MutableShapedBuffer(values: result, shape: resultShape)
    }

    public static func get<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> MutableShapedBuffer<Element, CPU> {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        let ranges = shape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< shape[axis] }
        let result = allocateBuffer(withCapacity: ranges.map(\.count).reduce(1, *), type: Element.self)
        copyRegion(of: buffer, shape: shape, ranges: ranges, to: result, inside: false)
        return MutableShapedBuffer(values: result, shape: ranges.map(\.count))
    }

    public static func set<Element>(slice: [Int?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        let countDelta = dstShape.count - slice.filter { $0 != nil }.count
        precondition(sourceShape.count == countDelta, "Dimensionality of source must be equal to dimensionality of destination minus number of knowns in slice")
        let ranges = dstShape.indices.map { axis in (axis < slice.count ? slice[axis] : nil).map { $0 ..< $0 + 1 } ?? 0 ..< dstShape[axis] }
        copyRegion(of: Buffer(buffer), shape: dstShape, ranges: ranges, to: MutableBuffer(memory: source.memory), inside: true)
    }

    public static func set<Element>(slice: [Range<Int>?], of buffer: MutableBuffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count, "Dimensionality of source must be equal to dimensionality of destination")
        let ranges = dstShape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< dstShape[axis] }
        precondition(ranges.map(\.count) == sourceShape, "Shape of source must be equal to the shape of the slice")
        copyRegion(of: Buffer(buffer), shape: dstShape, ranges: ranges, to: MutableBuffer(memory: source.memory), inside: true)
    }

    /// Copies the elements of a region of a buffer from or into a contiguous buffer, one row of the region at a time.
    ///
    /// - Parameters:
    ///   - buffer: Buffer that contains the region.
    ///   - shape: Shape of `buffer`.
    ///   - ranges: Range of the region along every axis.
    ///   - contiguous: Buffer with the elements of the region in a contiguous layout.
    ///   - inside: Whether the elements of `contiguous` are copied into the region (true), or the elements of the region
    ///     into `contiguous` (false).
    private static func copyRegion<Element>(of buffer: Buffer<Element, CPU>, shape: [Int], ranges: [Range<Int>], to contiguous: MutableBuffer<Element, CPU>, inside: Bool) {
        let regionShape = ranges.map(\.count)
        guard regionShape.allSatisfy({ $0 > 0 }) else {
            return
        }
        let strides = ShapeUtil.contiguousStrides(of: shape)
        let offset = zip(ranges, strides).map { $0.lowerBound * $1 }.reduce(0, +)
        let region = UnsafeMutablePointer(mutating: buffer.immutable.baseAddress!) + offset
        let elements = contiguous.pointer.baseAddress!
        let merged = StridedIteration.mergingAxes(shape: regionShape, strides: [strides, ShapeUtil.contiguousStrides(of: regionShape)])
        var (outerShape, regionStrides, elementStrides) = (merged.shape, merged.strides[0], merged.strides[1])
        var rowLength = 1
        if let length = outerShape.last, regionStrides.last == 1, elementStrides.last == 1 {
            rowLength = length
            outerShape.removeLast()
            regionStrides.removeLast()
            elementStrides.removeLast()
        }
        StridedIteration.forEachOffset(shape: outerShape, strides: regionStrides, elementStrides) { regionOffset, elementOffset in
            if inside {
                (region + regionOffset).update(from: elements + elementOffset, count: rowLength)
            } else {
                (elements + elementOffset).update(from: region + regionOffset, count: rowLength)
            }
        }
    }

    public static func getValue<Element>(from source: Buffer<Element, CPU>) -> Element {
        source.immutable[0]
    }

    public static func getSize<Element>(of buffer: Buffer<Element, CPU>) -> Int {
        buffer.bytes.count / MemoryLayout<Element>.stride
    }

    public static func advance<Element>(buffer: Buffer<Element, CPU>, by advancement: Int) -> Buffer<Element, CPU> {
        Buffer(advance(buffer: MutableBuffer(memory: buffer.memory), by: advancement, count: buffer.count - advancement))
    }

    public static func advance<Element>(buffer: MutableBuffer<Element, CPU>, by advancement: Int) -> MutableBuffer<Element, CPU> {
        advance(buffer: buffer, by: advancement, count: buffer.count - advancement)
    }

    /// Returns the region of `count` elements that starts `advancement` elements after the start of the buffer.
    private static func advance<Element>(buffer: MutableBuffer<Element, CPU>, by advancement: Int, count: Int) -> MutableBuffer<Element, CPU> {
        let stride = MemoryLayout<Element>.stride
        let bytes = UnsafeMutableRawBufferPointer(rebasing: buffer.bytes[advancement * stride ..< (advancement + count) * stride])
        return MutableBuffer(memory: CPUBuffer(storage: buffer.memory.storage, bytes: bytes))
    }

    public static func setPointee<Element>(of buffer: MutableBuffer<Element, CPU>, to newValue: Element) {
        buffer.pointer[0] = newValue
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

    internal static func recordAllocation(of buffer: UnsafeMutableRawBufferPointer) {
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
            print("[ALLOC TRACE]: buffer of \(buffer.count) bytes not freed after \(allocationTraceReportDelaySeconds) seconds.")
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

//
//  GPUMemory.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation
import Metal

/// A Metal buffer in memory that the host and the GPU share.
///
/// The buffer returns to the pool of the context when the storage is released.
/// The GPU can still use the buffer at that time: the queue runs the command buffers in order, so later commands
/// that reuse the buffer run after the commands that used it before.
final class GPUStorage: @unchecked Sendable {
    // `@unchecked Sendable`: The mutable state is only accessed while the stream lock of the context is held,
    // or by the host after a wait for the GPU, when no command can use the storage.

    /// The Metal buffer. A storage that no command used yet can replace it, see ``GPUContext/waitUntilWritable(_:)``.
    var buffer: any MTLBuffer
    /// Sequence number of the last command buffer that reads or writes the buffer.
    var lastUse: UInt64
    /// Sequence number of the last command buffer that writes the buffer.
    var lastWrite: UInt64 = 0
    /// Whether no command used the storage yet. The values of such a storage are undefined.
    var isFresh = true

    init(byteCount: Int) {
        let pooled = GPUContext.current.makeBuffer(byteCount: byteCount)
        buffer = pooled.buffer
        lastUse = pooled.lastUse
    }

    deinit {
        GPUContext.shared?.recycle(buffer, lastUse: lastUse)
    }
}

/// A region of a GPU storage.
///
/// Views of a tensor refer to the storage of the tensor with an offset.
public struct GPUBuffer {
    let storage: GPUStorage
    let byteOffset: Int
    let byteCount: Int

    /// The region in host memory.
    ///
    /// The host can access it only after the GPU completed the work that uses it, see ``GPUContext/waitUntilReadable(_:)``.
    var hostMemory: UnsafeMutableRawBufferPointer {
        // `contents()` autoreleases the buffer, see ``GPUContext``. The storage keeps the buffer alive while the pointer is used.
        let contents = autoreleasepool { storage.buffer.contents() }
        return UnsafeMutableRawBufferPointer(start: contents + byteOffset, count: byteCount)
    }
}

public struct GPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = GPUBuffer
    public typealias Device = GPU

    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> MutableBuffer<Element, GPU> {
        let byteCount = capacity * MemoryLayout<Element>.stride
        return MutableBuffer(memory: GPUBuffer(storage: GPUStorage(byteCount: byteCount), byteOffset: 0, byteCount: byteCount))
    }

    public static func free<Element>(_ buffer: MutableBuffer<Element, GPU>) {
        // The storage returns its buffer to the pool when the last reference to it is released.
    }

    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: MutableBuffer<Element, GPU>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        guard byteCount > 0 else {
            return
        }
        let context = GPUContext.current
        if context.isHostWritableWithoutReplacement(destination.memory) {
            memcpy(destination.memory.hostMemory.baseAddress!, source.baseAddress!, byteCount)
            return
        }
        // The GPU still uses the buffer of the destination. Small values are written by a GPU command that contains them,
        // larger values go through a staging buffer and a copy on the GPU. The host does not wait for the GPU.
        if byteCount <= GPUKernels.maximumImmediateByteCount {
            GPUKernels.write(UnsafeRawBufferPointer(source), to: destination.memory)
            return
        }
        let staging = GPUBuffer(storage: GPUStorage(byteCount: byteCount), byteOffset: 0, byteCount: byteCount)
        context.waitUntilWritable(staging.storage)
        memcpy(staging.hostMemory.baseAddress!, source.baseAddress!, byteCount)
        GPUKernels.copyWords(from: staging, to: destination.memory, byteCount: byteCount)
    }

    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: MutableBuffer<Element, GPU>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        guard byteCount > 0 else {
            return
        }
        if GPUPlacement.runsOnHost(elements: count, reading: [source.memory], writing: [destination.memory]) {
            memcpy(destination.host.memory.baseAddress!, source.host.memory.baseAddress!, byteCount)
        } else {
            GPUKernels.copyWords(from: source.memory, to: destination.memory, byteCount: byteCount)
        }
    }

    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        guard count > 0 else {
            return
        }
        GPUContext.current.waitUntilReadable(source.memory.storage)
        memcpy(destination.baseAddress!, source.memory.hostMemory.baseAddress!, count * MemoryLayout<Element>.stride)
    }

    public static func getValue<Element>(from source: Buffer<Element, GPU>) -> Element {
        GPUContext.current.waitUntilReadable(source.memory.storage)
        return source.memory.hostMemory.load(as: Element.self)
    }

    public static func getSize<Element>(of buffer: Buffer<Element, GPU>) -> Int {
        buffer.memory.byteCount / MemoryLayout<Element>.stride
    }

    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (MutableBuffer<Element, GPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        var sliceCount = slice.count
        while sliceCount > 0, slice[sliceCount - 1] == nil {
            sliceCount -= 1
        }
        let strides = CPUMemoryOperators.strides(from: shape)
        let prefix = slice.prefix(sliceCount)
        if prefix.allSatisfy({ $0 != nil }) {
            let offset = zip(prefix, strides).map { $0! * $1 }.reduce(0, +)
            let resultShape = Array(shape.dropFirst(sliceCount))
            let view = GPUBuffer(
                storage: buffer.memory.storage,
                byteOffset: buffer.memory.byteOffset + offset * MemoryLayout<Element>.stride,
                byteCount: resultShape.reduce(1, *) * MemoryLayout<Element>.stride,
            )
            return (MutableBuffer(memory: view), false, resultShape)
        }
        let ranges = shape.indices.map { axis in (axis < slice.count ? slice[axis] : nil).map { $0 ..< $0 + 1 } ?? 0 ..< shape[axis] }
        let resultShape = zip(slice + [Int?](repeating: nil, count: shape.count - slice.count), shape).compactMap { index, size in index == nil ? size : nil }
        let result = allocateBuffer(withCapacity: resultShape.reduce(1, *), type: Element.self)
        GPUKernels.copyRegion(of: buffer, shape: shape, ranges: ranges, to: result)
        return (result, true, resultShape)
    }

    public static func get<Element>(slice: [CountableRange<Int>?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (MutableBuffer<Element, GPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        let ranges = shape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< shape[axis] }
        let result = allocateBuffer(withCapacity: ranges.map(\.count).reduce(1, *), type: Element.self)
        GPUKernels.copyRegion(of: buffer, shape: shape, ranges: ranges, to: result)
        return (result, true, ranges.map(\.count))
    }

    public static func set<Element>(slice: [Int?], of buffer: MutableBuffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        let countDelta = dstShape.count - slice.filter { $0 != nil }.count
        precondition(sourceShape.count == countDelta, "Dimensionality of source must be equal to dimensionality of destination minus number of knowns in slice")
        let ranges = dstShape.indices.map { axis in (axis < slice.count ? slice[axis] : nil).map { $0 ..< $0 + 1 } ?? 0 ..< dstShape[axis] }
        GPUKernels.writeRegion(of: buffer, shape: dstShape, ranges: ranges, from: source)
    }

    public static func set<Element>(slice: [Range<Int>?], of buffer: MutableBuffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        precondition(sourceShape.count == dstShape.count, "Dimensionality of source must be equal to dimensionality of destination")
        let ranges = dstShape.indices.map { axis in (axis < slice.count ? slice[axis] : nil) ?? 0 ..< dstShape[axis] }
        GPUKernels.writeRegion(of: buffer, shape: dstShape, ranges: ranges, from: source)
    }

    public static func setPointee<Element>(of buffer: MutableBuffer<Element, GPU>, to newValue: Element) {
        GPUContext.current.waitUntilWritable(buffer.memory.storage)
        buffer.memory.hostMemory.storeBytes(of: newValue, as: Element.self)
    }

    public static func advance<Element>(buffer: Buffer<Element, GPU>, by advancement: Int) -> Buffer<Element, GPU> {
        Buffer(memory: advance(buffer.memory, by: advancement * MemoryLayout<Element>.stride))
    }

    public static func advance<Element>(buffer: MutableBuffer<Element, GPU>, by advancement: Int) -> MutableBuffer<Element, GPU> {
        MutableBuffer(memory: advance(buffer.memory, by: advancement * MemoryLayout<Element>.stride))
    }

    private static func advance(_ buffer: GPUBuffer, by bytes: Int) -> GPUBuffer {
        GPUBuffer(storage: buffer.storage, byteOffset: buffer.byteOffset + bytes, byteCount: buffer.byteCount - bytes)
    }
}

// MARK: Host views

extension Buffer where Device == GPU {
    /// The values in host memory, after the GPU completed the work that writes them.
    var host: Buffer<Element, CPU> {
        GPUContext.current.waitUntilReadable(memory.storage)
        return Buffer<Element, CPU>(memory: memory.hostMemory)
    }
}

extension MutableBuffer where Device == GPU {
    /// The values in host memory, after the GPU completed the work that uses them.
    var host: MutableBuffer<Element, CPU> {
        GPUContext.current.waitUntilWritable(memory.storage)
        return MutableBuffer<Element, CPU>(memory: memory.hostMemory)
    }
}

extension ShapedBuffer where Device == GPU {
    /// The values in host memory, after the GPU completed the work that writes them.
    var host: ShapedBuffer<Element, CPU> {
        ShapedBuffer<Element, CPU>(values: values.host, shape: shape)
    }
}

extension MutableShapedBuffer where Device == GPU {
    /// The values in host memory, after the GPU completed the work that uses them.
    var host: MutableShapedBuffer<Element, CPU> {
        MutableShapedBuffer<Element, CPU>(values: values.host, shape: shape)
    }
}
#endif

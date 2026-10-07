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
import Synchronization

// The queue runs the command buffers in order, so later commands that reuse the buffer run after the commands that used it before.
/// A Metal buffer in memory that the host and the GPU share, which belongs to a context.
///
/// The buffer returns to the pool of the context when the storage is released, also when the GPU still uses it.
final class GPUStorage: Sendable {
    // The context changes the state only while it holds the lock of its stream, so the values of the fields agree with each
    // other. The atomics and the mutex only make the single accesses safe.

    /// The context that records the commands that use the storage, and whose pool receives the buffer.
    let context: GPUContext
    private let metalBuffer: Mutex<GPUMetalBuffer>
    private let lastUseSequence: Atomic<UInt64>
    private let lastWriteSequence = Atomic<UInt64>(0)
    private let fresh = Atomic<Bool>(true)

    init(byteCount: Int, context: GPUContext) {
        let pooled = context.makeBuffer(byteCount: byteCount)
        self.context = context
        metalBuffer = Mutex(pooled.buffer)
        lastUseSequence = Atomic(pooled.lastUse)
    }

    deinit {
        context.recycle(metalBuffer.withLock { $0 }, lastUse: lastUse)
    }

    /// The Metal buffer. A storage that no command used yet can replace it, see ``GPUContext/waitUntilWritable(_:)``.
    var buffer: any MTLBuffer {
        metalBuffer.withLock { $0 }.buffer
    }

    /// Replaces the Metal buffer of a storage that no command used yet.
    func replaceBuffer(with replacement: GPUPooledBuffer) -> GPUMetalBuffer {
        let previous = metalBuffer.withLock { buffer in
            let previous = buffer
            buffer = replacement.buffer
            return previous
        }
        lastUse = replacement.lastUse
        return previous
    }

    /// Sequence number of the last command buffer that reads or writes the buffer.
    var lastUse: UInt64 {
        get {
            lastUseSequence.load(ordering: .relaxed)
        }
        set {
            lastUseSequence.store(newValue, ordering: .relaxed)
        }
    }

    /// Sequence number of the last command buffer that writes the buffer.
    var lastWrite: UInt64 {
        get {
            lastWriteSequence.load(ordering: .relaxed)
        }
        set {
            lastWriteSequence.store(newValue, ordering: .relaxed)
        }
    }

    /// Whether no command used the storage yet. The values of such a storage are undefined.
    var isFresh: Bool {
        get {
            fresh.load(ordering: .relaxed)
        }
        set {
            fresh.store(newValue, ordering: .relaxed)
        }
    }

    /// Waits until the GPU completed all work that writes the storage.
    func waitUntilReadable() {
        context.waitUntilReadable(self)
    }

    /// Waits until the GPU completed all work that uses the storage, see ``GPUContext/waitUntilWritable(_:)``.
    func waitUntilWritable() {
        context.waitUntilWritable(self)
    }
}

/// A region of a GPU storage.
///
/// Views of a tensor refer to the storage of the tensor with an offset.
public struct GPUBuffer: Sendable {
    let storage: GPUStorage
    let byteOffset: Int
    let byteCount: Int

    /// The region in host memory.
    ///
    /// The host can access it only after the GPU completed the work that uses it, see ``GPUContext/waitUntilReadable(_:)``.
    var hostMemory: UnsafeMutableRawBufferPointer {
        // `contents()` autoreleases the buffer, see ``GPUContext``. The storage keeps the buffer alive while the pointer is used.
        let buffer = storage.buffer
        let contents = autoreleasepool { buffer.contents() }
        return UnsafeMutableRawBufferPointer(start: contents + byteOffset, count: byteCount)
    }
}

public struct GPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = GPUBuffer
    public typealias Device = GPU

    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> MutableBuffer<Element, GPU> {
        let byteCount = capacity * MemoryLayout<Element>.stride
        return MutableBuffer(memory: GPUBuffer(storage: GPUStorage(byteCount: byteCount, context: .current), byteOffset: 0, byteCount: byteCount))
    }

    public static func free<Element>(_ buffer: MutableBuffer<Element, GPU>) {
        // The storage returns its buffer to the pool when the last reference to it is released.
    }

    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: MutableBuffer<Element, GPU>, count: Int) {
        let byteCount = count * MemoryLayout<Element>.stride
        guard byteCount > 0 else {
            return
        }
        let context = destination.memory.storage.context
        if context.makeHostWritableWithoutWait(destination.memory) {
            memcpy(destination.memory.hostMemory.baseAddress!, source.baseAddress!, byteCount)
            return
        }
        // The GPU still uses the buffer of the destination. Small values are written by a GPU command that contains them,
        // larger values go through a staging buffer and a copy on the GPU. The host does not wait for the GPU.
        if byteCount <= GPUKernels.maximumImmediateByteCount {
            GPUKernels.write(UnsafeRawBufferPointer(source), to: destination.memory)
            return
        }
        let staging = GPUBuffer(storage: GPUStorage(byteCount: byteCount, context: context), byteOffset: 0, byteCount: byteCount)
        staging.storage.waitUntilWritable()
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
        source.memory.storage.waitUntilReadable()
        memcpy(destination.baseAddress!, source.memory.hostMemory.baseAddress!, count * MemoryLayout<Element>.stride)
    }

    public static func getValue<Element>(from source: Buffer<Element, GPU>) -> Element {
        source.memory.storage.waitUntilReadable()
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
        let strides = MemoryOps.strides(from: shape)
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
        buffer.memory.storage.waitUntilWritable()
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
        memory.storage.waitUntilReadable()
        return Buffer<Element, CPU>(memory: memory.hostMemory)
    }
}

extension MutableBuffer where Device == GPU {
    /// The values in host memory, after the GPU completed the work that uses them.
    var host: MutableBuffer<Element, CPU> {
        memory.storage.waitUntilWritable()
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

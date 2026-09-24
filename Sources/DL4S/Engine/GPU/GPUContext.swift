//
//  GPUContext.swift
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
import MetalPerformanceShaders
import Synchronization

/// The command buffer that records GPU work, and the command buffers that the GPU runs.
///
/// Every command buffer has a sequence number. The numbers increase in the order in which the command buffers are committed,
/// and the queue runs them in that order. A storage records the sequence numbers of the last command buffers that use it and that write it,
/// so that a host access waits only for the command buffers that it depends on.
struct GPUStream: ~Copyable, @unchecked Sendable {
    // `@unchecked Sendable`: The stream is only accessed while the lock of the context is held.

    /// Command buffer that records work, or nil when no work was recorded since the last commit.
    var commandBuffer: (any MTLCommandBuffer)?
    /// Open compute encoder of the command buffer.
    var encoder: (any MTLComputeCommandEncoder)?
    /// Sequence number of the command buffer that records work.
    var sequence: UInt64 = 1
    /// Number of commands in the command buffer that records work.
    var commandCount = 0
    /// Committed command buffers that can still run.
    var inFlight: [(sequence: UInt64, commandBuffer: any MTLCommandBuffer)] = []

    mutating func endEncoding() {
        encoder?.endEncoding()
        encoder = nil
    }

    mutating func openCommandBuffer(queue: any MTLCommandQueue) -> any MTLCommandBuffer {
        if let commandBuffer {
            return commandBuffer
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            preconditionFailure("DL4S: The Metal command queue could not create a command buffer.")
        }
        self.commandBuffer = commandBuffer
        return commandBuffer
    }

    mutating func openEncoder(queue: any MTLCommandQueue) -> any MTLComputeCommandEncoder {
        if let encoder {
            return encoder
        }
        // A serial encoder runs the dispatches in order and makes the writes of a dispatch visible to the next one.
        // Most commands of a training step depend on the command before them, and on Apple GPUs a serial encoder costs
        // less per dependent dispatch than a concurrent encoder with memory barriers.
        guard let encoder = openCommandBuffer(queue: queue).makeComputeCommandEncoder(dispatchType: .serial) else {
            preconditionFailure("DL4S: The Metal command buffer could not create a compute encoder.")
        }
        self.encoder = encoder
        return encoder
    }

    /// Records that the command buffer that records work reads and writes the given buffers.
    func record(reading: [GPUBuffer], writing: [GPUBuffer]) {
        for buffer in reading {
            buffer.storage.lastUse = sequence
            buffer.storage.isFresh = false
        }
        for buffer in writing {
            buffer.storage.lastUse = sequence
            buffer.storage.lastWrite = sequence
            buffer.storage.isFresh = false
        }
    }
}

/// A buffer that the pool keeps for reuse.
struct GPUPooledBuffer {
    let buffer: any MTLBuffer
    /// Sequence number of the last command buffer that used the buffer.
    let lastUse: UInt64
}

/// Buffers of released storages, grouped by capacity.
struct GPUBufferPool: ~Copyable, @unchecked Sendable {
    // `@unchecked Sendable`: The pool is only accessed while the lock of the context is held.

    var buckets: [Int: [GPUPooledBuffer]] = [:]
    var cachedBytes = 0
}

/// The Metal device, its command queue, and the state of the GPU work.
final class GPUContext: @unchecked Sendable {
    // `@unchecked Sendable`: The mutable state is in mutexes and atomics. Metal devices and queues can be used from any thread.

    /// The context of the system default Metal device, or nil when the system has none.
    static let shared: GPUContext? = GPUContext()

    /// The shared context. Traps when the system has no Metal device.
    static var current: GPUContext {
        guard let shared else {
            preconditionFailure("DL4S: The system has no Metal device. Check GPU.isAvailable before you create GPU tensors.")
        }
        return shared
    }

    /// Number of commands after which a command buffer is committed, so that the GPU runs while the host records more work.
    static let commandsPerCommandBuffer = 16

    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let kernels: GPUKernelLibrary
    /// Whether the matrix kernels of the package compute products. They need SIMD group matrices, so on other devices,
    /// Metal Performance Shaders compute all products. Tests switch the kernels off to check that path.
    var supportsMatrixKernels: Bool {
        get {
            matrixKernels.load(ordering: .relaxed)
        }
        set {
            matrixKernels.store(newValue && device.supportsFamily(.apple7), ordering: .relaxed)
        }
    }

    private let matrixKernels: Atomic<Bool>
    /// Maximum number of bytes that the pool keeps.
    let poolLimit: Int

    private let stream = Mutex(GPUStream())
    private let pool = Mutex(GPUBufferPool())
    /// Sequence number up to which all command buffers completed.
    private let completed = Atomic<UInt64>(0)
    /// Sequence numbers of completed command buffers after the first command buffer that did not complete yet.
    ///
    /// The queue can run neighboring command buffers at the same time, so a command buffer can complete before an earlier one.
    /// `completed` only advances over a contiguous run of completed command buffers.
    private let completions = Mutex<Set<UInt64>>([])
    /// Description of the first command buffer that failed.
    private let failure = Mutex<String?>(nil)
    private let hostLimit = Atomic<Int>(4096)
    private let commandCounter = Atomic<Int>(0)
    private let commitCounter = Atomic<Int>(0)
    private let waitCounter = Atomic<Int>(0)

    private init?() {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.queue = queue
        kernels = GPUKernelLibrary(device: device)
        matrixKernels = Atomic(device.supportsFamily(.apple7))
        poolLimit = Int(min(device.recommendedMaxWorkingSetSize / 4, 4 << 30))
    }

    var hostExecutionLimit: Int {
        get {
            hostLimit.load(ordering: .relaxed)
        }
        set {
            hostLimit.store(max(newValue, 0), ordering: .relaxed)
        }
    }

    // MARK: Recording work

    /// Records a compute command.
    ///
    /// - Parameters:
    ///   - pipeline: Pipeline state of the kernel.
    ///   - reading: Buffers that the command reads.
    ///   - writing: Buffers that the command writes.
    ///   - encode: Sets the arguments and dispatches the threads.
    func compute(_ pipeline: any MTLComputePipelineState, reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (inout GPUArguments) -> Void) {
        stream.withLock { stream in
            let encoder = stream.openEncoder(queue: queue)
            encoder.setComputePipelineState(pipeline)
            var arguments = GPUArguments(encoder: encoder, pipeline: pipeline)
            encode(&arguments)
            stream.record(reading: reading, writing: writing)
            finishCommand(&stream)
        }
    }

    /// Records commands that use the command buffer directly, such as Metal Performance Shaders kernels.
    ///
    /// - Parameters:
    ///   - reading: Buffers that the commands read.
    ///   - writing: Buffers that the commands write.
    ///   - encode: Encodes the commands into the command buffer.
    func commands(reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (any MTLCommandBuffer) -> Void) {
        stream.withLock { stream in
            stream.endEncoding()
            encode(stream.openCommandBuffer(queue: queue))
            stream.record(reading: reading, writing: writing)
            finishCommand(&stream)
        }
    }

    /// Records the commands of a Metal Performance Shaders graph.
    ///
    /// - Parameters:
    ///   - reading: Buffers that the graph reads.
    ///   - writing: Buffers that the graph writes.
    ///   - encode: Encodes the graph into the command buffer.
    func graph(reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (MPSCommandBuffer) -> Void) {
        stream.withLock { stream in
            stream.endEncoding()
            let commandBuffer = MPSCommandBuffer(commandBuffer: stream.openCommandBuffer(queue: queue))
            encode(commandBuffer)
            // A graph can commit the command buffer and continue in a new one. The queue runs the new one after the
            // committed one, so the new one takes the place of the old one in the stream.
            stream.commandBuffer = commandBuffer.commandBuffer
            stream.record(reading: reading, writing: writing)
            finishCommand(&stream)
        }
    }

    private func finishCommand(_ stream: inout GPUStream) {
        commandCounter.add(1, ordering: .relaxed)
        stream.commandCount += 1
        if stream.commandCount >= Self.commandsPerCommandBuffer {
            commit(&stream)
        }
    }

    private func commit(_ stream: inout GPUStream) {
        guard let commandBuffer = stream.commandBuffer else {
            return
        }
        stream.endEncoding()
        let sequence = stream.sequence
        commandBuffer.addCompletedHandler { [self] commandBuffer in
            if let error = commandBuffer.error {
                failure.withLock { failure in
                    failure = failure ?? error.localizedDescription
                }
            }
            markCompleted(sequence)
        }
        commandBuffer.commit()
        commitCounter.add(1, ordering: .relaxed)

        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        stream.inFlight.removeAll { $0.sequence <= completedSequence }
        stream.inFlight.append((sequence, commandBuffer))
        stream.commandBuffer = nil
        stream.commandCount = 0
        stream.sequence += 1
    }

    /// Numbers of recorded commands, committed command buffers, and host accesses that waited for the GPU since the process started.
    var statistics: (commands: Int, commits: Int, waits: Int) {
        (commandCounter.load(ordering: .relaxed), commitCounter.load(ordering: .relaxed), waitCounter.load(ordering: .relaxed))
    }

    // MARK: Host access

    /// Whether the host can read and write the given buffers without waiting for the GPU.
    func isHostAccessible(reading: [GPUBuffer], writing: [GPUBuffer]) -> Bool {
        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        return stream.withLock { _ in
            reading.allSatisfy { $0.storage.lastWrite <= completedSequence }
                && writing.allSatisfy { $0.storage.isFresh || $0.storage.lastUse <= completedSequence }
        }
    }

    /// Whether the GPU completed all work that uses the buffer of the given region, so that the host can write it now.
    func isHostWritableWithoutReplacement(_ buffer: GPUBuffer) -> Bool {
        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        return stream.withLock { _ in buffer.storage.lastUse <= completedSequence }
    }

    /// Waits until the GPU completed all work that writes the storage.
    func waitUntilReadable(_ storage: GPUStorage) {
        wait(forSequence: stream.withLock { _ in storage.lastWrite })
    }

    /// Waits until the GPU completed all work that uses the storage.
    ///
    /// A storage that no command used yet does not wait: when the GPU still uses its buffer for earlier work,
    /// the storage takes a buffer that the host can write at once, and the old buffer returns to the pool.
    func waitUntilWritable(_ storage: GPUStorage) {
        let target = stream.withLock { _ -> UInt64 in
            let completedSequence = completed.load(ordering: .sequentiallyConsistent)
            guard storage.isFresh else {
                return storage.lastUse
            }
            if storage.lastUse > completedSequence {
                let replacement = makeBuffer(byteCount: storage.buffer.length, hostWritable: true)
                recycle(storage.buffer, lastUse: storage.lastUse)
                storage.buffer = replacement.buffer
                storage.lastUse = replacement.lastUse
            }
            return 0
        }
        wait(forSequence: target)
    }

    /// Submits all recorded work and waits until the GPU completes it.
    func synchronize() {
        wait(forSequence: stream.withLock { $0.commandBuffer == nil ? $0.sequence - 1 : $0.sequence })
    }

    private func wait(forSequence target: UInt64) {
        guard completed.load(ordering: .sequentiallyConsistent) < target else {
            return
        }
        waitCounter.add(1, ordering: .relaxed)
        if let trace = ProcessInfo.processInfo.environment["DL4S_GPU_TRACE_WAITS"], trace == "1" {
            print("[DL4S GPU wait]", Thread.callStackSymbols.dropFirst(2).prefix(12).joined(separator: "\n"))
        }
        let commandBuffers = stream.withLock { stream in
            if stream.commandBuffer != nil, stream.sequence <= target {
                commit(&stream)
            }
            return stream.inFlight.filter { $0.sequence <= target }.map(\.commandBuffer)
        }
        for commandBuffer in commandBuffers {
            commandBuffer.waitUntilCompleted()
        }
        if let failure = failure.withLock({ $0 }) {
            preconditionFailure("DL4S: A GPU command buffer failed: \(failure)")
        }
        markCompleted(upTo: target)
    }

    /// Records that the command buffer with the given sequence number completed.
    private func markCompleted(_ sequence: UInt64) {
        completions.withLock { finished in
            var contiguous = completed.load(ordering: .sequentiallyConsistent)
            guard sequence > contiguous else {
                return
            }
            finished.insert(sequence)
            while finished.remove(contiguous + 1) != nil {
                contiguous += 1
            }
            completed.store(contiguous, ordering: .sequentiallyConsistent)
        }
    }

    /// Records that all command buffers up to the given sequence number completed.
    private func markCompleted(upTo target: UInt64) {
        completions.withLock { finished in
            var contiguous = max(completed.load(ordering: .sequentiallyConsistent), target)
            finished = finished.filter { $0 > contiguous }
            while finished.remove(contiguous + 1) != nil {
                contiguous += 1
            }
            completed.store(contiguous, ordering: .sequentiallyConsistent)
        }
    }

    // MARK: Buffer pool

    /// Returns a buffer with at least the given number of bytes, from the pool when possible.
    ///
    /// - Parameters:
    ///   - byteCount: Minimum number of bytes.
    ///   - hostWritable: Whether the GPU must have completed all work that uses the buffer.
    /// - Returns: The buffer and the sequence number of the last command buffer that used it.
    func makeBuffer(byteCount: Int, hostWritable: Bool = false) -> GPUPooledBuffer {
        let capacity = Self.capacity(forByteCount: byteCount)
        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        let reused = pool.withLock { pool -> GPUPooledBuffer? in
            guard var bucket = pool.buckets[capacity], !bucket.isEmpty else {
                return nil
            }
            // Buffers are appended when they are released, so the first buffers of a bucket are the oldest ones.
            // The GPU reuses the newest buffer, which is likely still in its cache. The host needs a buffer that the GPU no longer uses,
            // which is most likely one of the oldest ones.
            let index: Int
            if hostWritable {
                guard let oldest = bucket.indices.prefix(16).first(where: { bucket[$0].lastUse <= completedSequence }) else {
                    return nil
                }
                index = oldest
            } else {
                index = bucket.count - 1
            }
            let buffer = bucket.remove(at: index)
            pool.buckets[capacity] = bucket
            pool.cachedBytes -= capacity
            return buffer
        }
        if let reused {
            return reused
        }
        if let buffer = device.makeBuffer(length: capacity, options: .storageModeShared) {
            return GPUPooledBuffer(buffer: buffer, lastUse: 0)
        }
        clearCache()
        guard let buffer = device.makeBuffer(length: capacity, options: .storageModeShared) else {
            preconditionFailure("DL4S: The GPU could not allocate a buffer of \(capacity) bytes.")
        }
        return GPUPooledBuffer(buffer: buffer, lastUse: 0)
    }

    /// Returns the buffer of a released storage to the pool.
    func recycle(_ buffer: any MTLBuffer, lastUse: UInt64) {
        let capacity = buffer.length
        pool.withLock { pool in
            guard pool.cachedBytes + capacity <= poolLimit else {
                return
            }
            pool.buckets[capacity, default: []].append(GPUPooledBuffer(buffer: buffer, lastUse: lastUse))
            pool.cachedBytes += capacity
        }
    }

    func clearCache() {
        // The buffers are released after the lock, so that no deinitializer runs while the lock is held.
        let buffers = pool.withLock { pool in
            let buffers = pool.buckets
            pool.buckets.removeAll()
            pool.cachedBytes = 0
            return buffers
        }
        _ = consume buffers
    }

    /// Size class of an allocation: a power of two up to 1 MiB, above that a multiple of an eighth of a power of two.
    ///
    /// Size classes let released buffers serve later allocations of a similar size, with at most 12.5% unused memory for large buffers.
    static func capacity(forByteCount byteCount: Int) -> Int {
        let byteCount = max(byteCount, 256)
        let power = Int.bitWidth - (byteCount - 1).leadingZeroBitCount
        if power <= 20 {
            return 1 << power
        }
        let step = 1 << (power - 4)
        return (byteCount + step - 1) / step * step
    }
}

/// Sets the arguments of a compute command in the order of the buffer indices of the kernel.
struct GPUArguments: ~Copyable {
    let encoder: any MTLComputeCommandEncoder
    let pipeline: any MTLComputePipelineState
    private var index = 0

    init(encoder: any MTLComputeCommandEncoder, pipeline: any MTLComputePipelineState) {
        self.encoder = encoder
        self.pipeline = pipeline
    }

    mutating func buffer(_ buffer: GPUBuffer) {
        encoder.setBuffer(buffer.storage.buffer, offset: buffer.byteOffset, index: index)
        index += 1
    }

    /// Sets a buffer that starts the given number of bytes after the start of the given buffer.
    mutating func buffer(_ buffer: GPUBuffer, byteOffset: Int) {
        encoder.setBuffer(buffer.storage.buffer, offset: buffer.byteOffset + byteOffset, index: index)
        index += 1
    }

    mutating func value<Value: BitwiseCopyable>(_ value: Value) {
        withUnsafeBytes(of: value) { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        }
        index += 1
    }

    mutating func bytes(_ bytes: UnsafeRawBufferPointer) {
        encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        index += 1
    }

    mutating func values<Value: BitwiseCopyable>(_ values: [Value]) {
        values.withUnsafeBytes { bytes in
            encoder.setBytes(bytes.baseAddress!, length: max(bytes.count, 4), index: index)
        }
        index += 1
    }

    /// Dispatches one thread per element.
    func dispatch(count: Int) {
        guard count > 0 else {
            return
        }
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    /// Dispatches a grid of threads with the given threadgroup size.
    func dispatch(threads: MTLSize, threadgroup: MTLSize) {
        guard threads.width > 0, threads.height > 0, threads.depth > 0 else {
            return
        }
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threadgroup)
    }

    /// Dispatches the given number of threadgroups.
    func dispatch(threadgroups: MTLSize, threadgroup: MTLSize) {
        guard threadgroups.width > 0, threadgroups.height > 0, threadgroups.depth > 0 else {
            return
        }
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroup)
    }
}
#endif

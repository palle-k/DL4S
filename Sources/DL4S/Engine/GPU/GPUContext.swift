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
struct GPUStream: ~Copyable {
    /// Command buffer that records work, or nil when no work was recorded since the last commit.
    var commandBuffer: (any MTLCommandBuffer)?
    /// Open compute encoder of the command buffer.
    var encoder: (any MTLComputeCommandEncoder)?
    /// Sequence number of the command buffer that records work.
    var sequence: UInt64 = 1
    /// Number of commands in the command buffer that records work.
    var commandCount = 0
    /// Whether the completion of the command buffer that records work is tracked.
    var tracksCompletion = false
    /// Committed command buffers that can still run.
    var inFlight: [GPUCommittedCommandBuffer] = []

    mutating func endEncoding() {
        encoder?.endEncoding()
        encoder = nil
    }

    mutating func openCommandBuffer(queue: any MTLCommandQueue) -> any MTLCommandBuffer {
        if let commandBuffer {
            return commandBuffer
        }
        // The command buffer is autoreleased, see ``GPUContext``.
        guard let commandBuffer = autoreleasepool(invoking: { queue.makeCommandBuffer() }) else {
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
        let commandBuffer = openCommandBuffer(queue: queue)
        guard let encoder = autoreleasepool(invoking: { commandBuffer.makeComputeCommandEncoder(dispatchType: .serial) }) else {
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

/// Numbers of recorded commands, committed command buffers, and host accesses that waited for the GPU.
struct GPUStatistics {
    let commands: Int
    let commits: Int
    let waits: Int
}

/// A committed command buffer with its sequence number.
struct GPUCommittedCommandBuffer {
    let sequence: UInt64
    let commandBuffer: any MTLCommandBuffer
}

// `@unchecked Sendable`: Metal does not mark buffers as `Sendable`, but the object of a buffer can be used from any thread.
// The context orders the accesses to the contents of a buffer with the sequence numbers of its command buffers.
/// A Metal buffer.
struct GPUMetalBuffer: @unchecked Sendable {
    let buffer: any MTLBuffer
}

/// A buffer of the pool, or a new buffer for a storage.
struct GPUPooledBuffer {
    let buffer: GPUMetalBuffer
    /// Sequence number of the last command buffer that used the buffer.
    let lastUse: UInt64
    /// Position of the buffer in the order in which the pool received the buffers.
    var age: UInt64 = 0
}

/// Buffers of released storages, grouped by capacity.
struct GPUBufferPool: ~Copyable {
    var buckets: [Int: [GPUPooledBuffer]] = [:]
    var cachedBytes = 0
    /// Number of buffers that the pool received.
    var receivedCount: UInt64 = 0

    /// Removes the buffer that the pool received first.
    mutating func removeOldest() -> GPUPooledBuffer? {
        // The first buffer of every bucket is the oldest buffer of the bucket.
        var oldestCapacity: Int?
        var oldestAge = UInt64.max
        for (capacity, bucket) in buckets {
            if let first = bucket.first, first.age < oldestAge {
                (oldestCapacity, oldestAge) = (capacity, first.age)
            }
        }
        guard let capacity = oldestCapacity else {
            return nil
        }
        let buffer = buckets[capacity]!.removeFirst()
        cachedBytes -= capacity
        return buffer
    }

    /// Position of the buffer in the bucket of the given capacity that serves a request: the newest buffer for the GPU, and
    /// for the host the oldest of the first 16 buffers that the GPU no longer uses, or nil when the bucket has none.
    func index(in capacity: Int, hostWritable: Bool, completedSequence: UInt64) -> Int? {
        guard let bucket = buckets[capacity], !bucket.isEmpty else {
            return nil
        }
        guard hostWritable else {
            return bucket.count - 1
        }
        return bucket.indices.prefix(16).first { bucket[$0].lastUse <= completedSequence }
    }
}

/// A Metal device, its command queue, the state of the GPU work, and the caches of the device. Every function that calls
/// Metal drains an autorelease pool.
///
/// Every storage belongs to one context, and the kernels record their commands in the context of their operands.
final class GPUContext: Sendable {
    // Metal returns command buffers, encoders, and objects of Metal Performance Shaders autoreleased, a command buffer keeps
    // all buffers that it uses alive, and `contents()` of a buffer autoreleases the buffer. A thread without a run loop, such
    // as the main thread of a command line tool, never drains its autorelease pool, so without the local pools every buffer
    // that a command or the host used would stay allocated.

    // The kernels use SIMD group matrices and assume SIMD groups of 32 threads, which Apple GPUs of the family Apple7 and
    // later have.
    /// The context of the system default Metal device, or nil when the system has no supported device.
    static let systemDefault: GPUContext? = MTLCreateSystemDefaultDevice().flatMap(GPUContext.init(device:))

    /// The context in which new storages are created. Traps when the system has no supported Metal device.
    static var current: GPUContext {
        guard let systemDefault else {
            preconditionFailure("DL4S: The system has no supported Metal device. Check GPU.isAvailable before you create GPU tensors.")
        }
        return systemDefault
    }

    // The GPU runs the committed commands while the host records more work.
    /// Number of commands after which a command buffer is committed.
    static let commandsPerCommandBuffer = 16

    let device: any MTLDevice
    let queue: any MTLCommandQueue
    let kernels: GPUKernelLibrary
    // Tests switch the kernels off to check the other path.
    /// Whether the matrix kernels of the package compute products. Otherwise, Metal Performance Shaders compute all products.
    var supportsMatrixKernels: Bool {
        get {
            matrixKernels.load(ordering: .relaxed)
        }
        set {
            matrixKernels.store(newValue, ordering: .relaxed)
        }
    }

    /// Compiled Metal Performance Shaders graphs of convolutions.
    let graphs = Mutex(GPUCache<GPUConvolutionKey, GPUGraph>(capacity: 256))
    /// Metal Performance Shaders kernels of matrix products.
    let matrixProducts = Mutex(GPUCache<GPUMatrixProductKey, GPUMatrixProductKernel>(capacity: 256))
    // The tables do not depend on the batch size, but on the size of the images, which can change in every step.
    /// Offset tables of the implicit convolutions, see ``clearCache()``.
    let convolutionTables = Mutex(GPUCache<GPUConvolutionTableKey, GPUConvolutionTables>(capacity: 64))

    private let matrixKernels = Atomic(true)
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
    private let hostLimit = Atomic<Int>(GPU.defaultHostExecutionLimit)
    private let commandCounter = Atomic<Int>(0)
    private let commitCounter = Atomic<Int>(0)
    private let waitCounter = Atomic<Int>(0)

    /// Creates the context of a device, or returns nil when the kernels do not support the device.
    init?(device: any MTLDevice) {
        guard device.supportsFamily(.apple7), let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.queue = queue
        kernels = GPUKernelLibrary(device: device)
        // The pool keeps at most a quarter of the memory that the GPU can use well, and at most 4 GB, so that released
        // tensors do not hold memory that other allocations of the process need.
        poolLimit = Int(min(device.recommendedMaxWorkingSetSize / 4, 4 << 30))
    }

    /// The context of the storages of a command. Traps when they belong to different contexts.
    static func of(reading: [GPUBuffer], writing: [GPUBuffer]) -> GPUContext {
        guard let context = (writing.first ?? reading.first)?.storage.context else {
            return current
        }
        precondition(
            reading.allSatisfy { $0.storage.context === context } && writing.allSatisfy { $0.storage.context === context },
            "DL4S: The operands of a GPU operation belong to different devices.",
        )
        return context
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
    ///   - kernel: The kernel.
    ///   - reading: Buffers that the command reads.
    ///   - writing: Buffers that the command writes.
    ///   - encode: Sets the arguments and dispatches the threads.
    func compute(_ kernel: GPUKernel, reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (inout GPUArguments) -> Void) {
        let pipeline = kernels.pipeline(kernel)
        // Encoding a command autoreleases nothing, so only the creation of a command buffer and a commit need a pool.
        stream.withLock { stream in
            let encoder = stream.openEncoder(queue: queue)
            encoder.setComputePipelineState(pipeline)
            var arguments = GPUArguments(encoder: encoder, pipeline: pipeline)
            encode(&arguments)
            stream.record(reading: reading, writing: writing)
            finishCommand(&stream)
        }
    }

    /// Records a compute command in the context of its buffers.
    ///
    /// - Parameters:
    ///   - kernel: The kernel.
    ///   - reading: Buffers that the command reads.
    ///   - writing: Buffers that the command writes.
    ///   - encode: Sets the arguments and dispatches the threads.
    static func compute(_ kernel: GPUKernel, reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (inout GPUArguments) -> Void) {
        of(reading: reading, writing: writing).compute(kernel, reading: reading, writing: writing, encode)
    }

    /// Records commands that use the command buffer directly, such as Metal Performance Shaders kernels.
    ///
    /// - Parameters:
    ///   - reading: Buffers that the commands read.
    ///   - writing: Buffers that the commands write.
    ///   - encode: Encodes the commands into the command buffer.
    func commands(reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (any MTLCommandBuffer) -> Void) {
        autoreleasepool {
            stream.withLock { stream in
                stream.endEncoding()
                encode(stream.openCommandBuffer(queue: queue))
                stream.record(reading: reading, writing: writing)
                finishCommand(&stream)
            }
        }
    }

    /// Records the commands of a Metal Performance Shaders graph.
    ///
    /// - Parameters:
    ///   - reading: Buffers that the graph reads.
    ///   - writing: Buffers that the graph writes.
    ///   - encode: Encodes the graph into the command buffer.
    func graph(reading: [GPUBuffer], writing: [GPUBuffer], _ encode: (MPSCommandBuffer) -> Void) {
        autoreleasepool {
            stream.withLock { stream in
                stream.endEncoding()
                let original = stream.openCommandBuffer(queue: queue)
                // A graph can commit the command buffer and continue in a new one, so the completion of the command buffer is
                // tracked before the graph encodes.
                trackCompletion(&stream)
                let commandBuffer = MPSCommandBuffer(commandBuffer: original)
                encode(commandBuffer)
                if commandBuffer.commandBuffer !== original {
                    // The commands of the committed command buffer keep its sequence number. The new command buffer, which the
                    // queue runs after it, takes the next one.
                    commitCounter.add(1, ordering: .relaxed)
                    stream.inFlight.append(GPUCommittedCommandBuffer(sequence: stream.sequence, commandBuffer: original))
                    stream.commandBuffer = commandBuffer.commandBuffer
                    stream.tracksCompletion = false
                    stream.commandCount = 0
                    stream.sequence += 1
                }
                stream.record(reading: reading, writing: writing)
                finishCommand(&stream)
            }
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
        autoreleasepool {
            stream.endEncoding()
            trackCompletion(&stream)
            commandBuffer.commit()
        }
        commitCounter.add(1, ordering: .relaxed)

        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        stream.inFlight.removeAll { $0.sequence <= completedSequence }
        stream.inFlight.append(GPUCommittedCommandBuffer(sequence: stream.sequence, commandBuffer: commandBuffer))
        stream.commandBuffer = nil
        stream.tracksCompletion = false
        stream.commandCount = 0
        stream.sequence += 1
    }

    /// Adds the handler that records the completion of the command buffer that records work, unless it has one.
    private func trackCompletion(_ stream: inout GPUStream) {
        guard let commandBuffer = stream.commandBuffer, !stream.tracksCompletion else {
            return
        }
        let sequence = stream.sequence
        commandBuffer.addCompletedHandler { [self] commandBuffer in
            if let error = commandBuffer.error {
                failure.withLock { failure in
                    failure = failure ?? error.localizedDescription
                }
            }
            markCompleted(sequence)
        }
        stream.tracksCompletion = true
    }

    /// Numbers of recorded commands, committed command buffers, and host accesses that waited for the GPU since the process started.
    var statistics: GPUStatistics {
        GPUStatistics(commands: commandCounter.load(ordering: .relaxed), commits: commitCounter.load(ordering: .relaxed), waits: waitCounter.load(ordering: .relaxed))
    }

    /// Number of bytes of the buffers that the pool keeps for reuse.
    var cachedByteCount: Int {
        pool.withLock { $0.cachedBytes }
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

    /// Whether the host can write the given region now, without a wait: when the GPU completed all work that uses its buffer,
    /// or when no command used the storage yet. Such a storage takes a buffer that the GPU no longer uses.
    func makeHostWritableWithoutWait(_ buffer: GPUBuffer) -> Bool {
        let completedSequence = completed.load(ordering: .sequentiallyConsistent)
        return stream.withLock { _ in
            let storage = buffer.storage
            guard storage.lastUse > completedSequence else {
                return true
            }
            guard storage.isFresh else {
                return false
            }
            replaceBuffer(of: storage)
            return true
        }
    }

    /// Gives a storage that no command used yet a buffer that the host can write at once, and returns its old buffer to the pool.
    /// The caller holds the lock of the stream.
    private func replaceBuffer(of storage: GPUStorage) {
        let lastUse = storage.lastUse
        let previous = storage.replaceBuffer(with: makeBuffer(byteCount: storage.buffer.length, hostWritable: true))
        recycle(previous, lastUse: lastUse)
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
                replaceBuffer(of: storage)
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
        #if DL4S_TRACE_WAITS
        print("[DL4S GPU wait]", Thread.callStackSymbols.dropFirst(2).prefix(12).joined(separator: "\n"))
        #endif
        autoreleasepool {
            let commandBuffers = stream.withLock { stream in
                if stream.commandBuffer != nil, stream.sequence <= target {
                    commit(&stream)
                }
                return stream.inFlight.filter { $0.sequence <= target }.map(\.commandBuffer)
            }
            for commandBuffer in commandBuffers {
                commandBuffer.waitUntilCompleted()
            }
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
            // A buffer of up to twice the size serves the request, so that tensors whose shapes change in every step,
            // such as batches of padded sequences, find buffers in the pool.
            var bucketCapacity = capacity
            while bucketCapacity <= 2 * capacity {
                defer {
                    bucketCapacity = Self.capacity(forByteCount: bucketCapacity + 1)
                }
                // Buffers are appended when they are released, so the first buffers of a bucket are the oldest ones.
                // The GPU reuses the newest buffer, which is likely still in its cache. The host needs a buffer that the GPU
                // no longer uses, which is most likely one of the oldest ones. It checks the 16 oldest, so that a search under the
                // lock stays short.
                guard let index = pool.index(in: bucketCapacity, hostWritable: hostWritable, completedSequence: completedSequence) else {
                    continue
                }
                // The bucket is changed in place: a copy of the bucket would copy all of its buffers for every allocation.
                let buffer = pool.buckets[bucketCapacity]!.remove(at: index)
                pool.cachedBytes -= bucketCapacity
                return buffer
            }
            return nil
        }
        if let reused {
            return reused
        }
        // The allocation autoreleases the device, see ``GPUContext``.
        if let buffer = autoreleasepool(invoking: { device.makeBuffer(length: capacity, options: .storageModeShared) }) {
            return GPUPooledBuffer(buffer: GPUMetalBuffer(buffer: buffer), lastUse: 0)
        }
        clearCache()
        guard let buffer = autoreleasepool(invoking: { device.makeBuffer(length: capacity, options: .storageModeShared) }) else {
            preconditionFailure("DL4S: The GPU could not allocate a buffer of \(capacity) bytes.")
        }
        return GPUPooledBuffer(buffer: GPUMetalBuffer(buffer: buffer), lastUse: 0)
    }

    /// Returns the buffer of a released storage to the pool.
    func recycle(_ buffer: GPUMetalBuffer, lastUse: UInt64) {
        let capacity = buffer.buffer.length
        pool.withLock { pool in
            guard capacity <= poolLimit else {
                return
            }
            // A full pool releases its oldest buffers, which were not reused for the longest time.
            while pool.cachedBytes + capacity > poolLimit, pool.removeOldest() != nil {}
            pool.receivedCount += 1
            pool.buckets[capacity, default: []].append(GPUPooledBuffer(buffer: buffer, lastUse: lastUse, age: pool.receivedCount))
            pool.cachedBytes += capacity
        }
    }

    /// Releases the buffers of the pool and the offset tables of the implicit convolutions.
    func clearCache() {
        // The buffers are released after the lock, so that no deinitializer runs while the lock is held. The storages of the
        // tables return their buffers to the pool, so the tables are released first.
        let tables = convolutionTables.withLock { $0.removeAll() }
        _ = consume tables
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
        precondition(!values.isEmpty, "A kernel argument has at least one value.")
        values.withUnsafeBytes { bytes in
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
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

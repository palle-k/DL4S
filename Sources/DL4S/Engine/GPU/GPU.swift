//
//  GPU.swift
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

/// The GPU of the system, used through Metal.
///
/// Tensors on the GPU are stored in memory that the CPU and the GPU share. Operations are recorded in command buffers
/// and run on the GPU after the operation returns. The values of a tensor are available on the host when they are read,
/// for example with `elements` or `item`: a read waits for the GPU work that writes the tensor.
///
/// Operations on small tensors whose values are available on the host run on the CPU, because a GPU command costs more
/// than the computation. Operations on `Double` tensors always run on the CPU, because Metal has no double precision arithmetic.
///
/// Check ``isAvailable`` before you create tensors on the GPU. Without a Metal device, the creation of a GPU tensor traps.
public struct GPU: DeviceType {
    public typealias Memory = GPUMemoryOperators
    public typealias Engine = GPUEngine
    public typealias FusedOperations = GPUFusedOperations

    /// Whether the system has a Metal device.
    public static var isAvailable: Bool {
        GPUContext.shared != nil
    }

    /// Name of the Metal device, or nil when the system has none.
    public static var deviceName: String? {
        GPUContext.shared?.device.name
    }

    /// Submits all recorded work to the GPU and waits until the GPU completes it.
    ///
    /// Reads of tensor values wait for the work that they need, so a call is only necessary to measure time.
    public static func synchronize() {
        GPUContext.shared?.synchronize()
    }

    /// Releases the buffers that the GPU keeps for reuse.
    ///
    /// Tensors that are released return their buffers to a cache, so that new tensors do not allocate memory.
    /// Call this function to return the memory of the cache to the system.
    public static func clearCache() {
        GPUContext.shared?.clearCache()
    }

    /// Number of elements up to which an operation runs on the CPU when all of its operands are available on the host.
    ///
    /// Set it to 0 to run all supported operations on the GPU.
    public static var hostExecutionLimit: Int {
        get {
            GPUContext.shared?.hostExecutionLimit ?? 0
        }
        set {
            GPUContext.shared?.hostExecutionLimit = newValue
        }
    }
}

/// Fused operations of the GPU.
///
/// The GPU has fused kernels for the activations, softmax, normalization, and the optimizer step.
/// The other operations use the default implementations, which run on the GPU through the basic operations of ``GPUEngine``.
public struct GPUFusedOperations: FusedOperationsType {
    public typealias Device = GPU
}
#endif

//
//  GPUFusedSupport.swift
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

extension Tensor where Device == GPU {
    /// Region of the storage that holds the elements of the tensor.
    var gpuBuffer: GPUBuffer {
        handle.values.memory
    }

    /// Writable region of the storage of the tensor. The storage is copied first when another tensor shares it.
    var mutableGPUBuffer: GPUBuffer {
        mutating get {
            mutableValues.values.memory
        }
    }
}

/// Helpers of the fused operations of the GPU.
enum GPUFused {
    /// Whether a fused operation runs its GPU kernel.
    ///
    /// The kernels exist for floats. A small operation whose operands are available on the host uses the default implementation,
    /// whose basic operations then run on the host.
    static func runsKernel<N>(_: N.Type, elements: Int, reading tensors: [Tensor<N, GPU>]) -> Bool {
        guard N.self == Float.self, elements > 0 else {
            return false
        }
        let context = GPUContext.current
        guard elements <= context.hostExecutionLimit else {
            return true
        }
        return !context.isHostAccessible(reading: tensors.map(\.gpuBuffer), writing: [])
    }

    /// Creates a tensor with undefined elements.
    static func makeTensor<N>(shape: [Int]) -> Tensor<N, GPU> {
        Tensor(using: GPU.Memory.allocateBuffer(withShape: shape, type: N.self), context: nil)
    }

    /// Number of threads of a threadgroup that processes one row of the given length.
    static func rowThreadgroupWidth(length: Int) -> Int {
        length <= 64 ? 32 : length <= 512 ? 128 : 256
    }

    /// Length of the parameter of an element-wise kernel, or nil when the kernel does not support the shape of the parameter.
    ///
    /// The kernels support a parameter with one element and a parameter with the shape of the last axes of the input.
    static func parameterLength(_ parameter: Tensor<some Any, GPU>, input: Tensor<some Any, GPU>) -> Int? {
        if parameter.count == 1 {
            return 1
        }
        guard parameter.dim <= input.dim, Array(input.shape.suffix(parameter.dim)) == parameter.shape else {
            return nil
        }
        return parameter.count
    }

    private struct ElementwiseParameters {
        var count: UInt32
        var parameterLength: UInt32
        var accumulate: UInt32
    }

    /// Records the forward kernel of an activation and returns its result.
    static func activation<N>(_ name: String, input: Tensor<N, GPU>, parameter: Tensor<N, GPU>? = nil, parameterLength: Int = 1) -> Tensor<N, GPU> {
        let result: Tensor<N, GPU> = makeTensor(shape: input.shape)
        let (x, y, a) = (input.gpuBuffer, result.gpuBuffer, parameter?.gpuBuffer ?? input.gpuBuffer)
        let parameters = ElementwiseParameters(count: UInt32(input.count), parameterLength: UInt32(parameterLength), accumulate: 0)
        let pipeline = GPUKernels.pipeline("\(name)_forward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, a], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.buffer(a)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
        return result
    }

    /// Records the backward kernel of an activation, which adds the gradient to the accumulated gradient or stores it.
    ///
    /// - Parameter input: Input of the forward operation, or its result for activations whose gradient uses the result.
    static func activationBackward<N>(_ name: String, input: Tensor<N, GPU>, outputGradient: Tensor<N, GPU>, parameter: Tensor<N, GPU>? = nil, parameterLength: Int = 1, accumulating gradient: inout Tensor<N, GPU>?) {
        let target = GPUGradientTarget(taking: &gradient, shape: input.shape)
        let (x, g, dx, a) = (input.gpuBuffer, outputGradient.gpuBuffer, target.buffer, parameter?.gpuBuffer ?? input.gpuBuffer)
        let parameters = ElementwiseParameters(count: UInt32(input.count), parameterLength: UInt32(parameterLength), accumulate: target.accumulates ? 1 : 0)
        let pipeline = GPUKernels.pipeline("\(name)_backward", in: .fused)
        GPUContext.current.compute(pipeline, reading: [x, g, a, dx], writing: [dx]) { arguments in
            arguments.buffer(x)
            arguments.buffer(g)
            arguments.buffer(dx)
            arguments.buffer(a)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
        target.finish(into: &gradient)
    }

    struct RowParameters {
        var length: UInt32
        var accumulate: UInt32
        var epsilon: Float
    }
}

/// Receives the gradient of a GPU kernel: the accumulated gradient itself, or a new tensor.
///
/// The accumulated gradient is used when it has no gradient graph and the shape of the gradient. The kernel then adds to it.
struct GPUGradientTarget<N: NumericType>: ~Copyable {
    private var tensor: Tensor<N, GPU>
    /// Whether the kernel adds to the elements of the buffer. It stores into them otherwise.
    let accumulates: Bool
    /// Buffer that the kernel writes.
    let buffer: GPUBuffer
    /// Accumulated gradient with a gradient graph, which gets the sum without an in-place write.
    private let graphAccumulator: Tensor<N, GPU>?

    /// Takes the accumulated gradient, or creates a new tensor when there is none or when it cannot be written in place.
    /// - Parameters:
    ///   - accumulator: Accumulated gradient, which is nil until ``finish(into:)``
    ///   - shape: Shape of the gradient
    init(taking accumulator: inout Tensor<N, GPU>?, shape: [Int]) {
        // The accumulated gradient is taken out of the optional, so that it is the only reference to its storage.
        if var existing = accumulator.take() {
            if !existing.requiresGradient, existing.shape == shape {
                buffer = existing.mutableGPUBuffer
                tensor = existing
                accumulates = true
                graphAccumulator = nil
                return
            }
            graphAccumulator = existing
        } else {
            graphAccumulator = nil
        }
        tensor = GPUFused.makeTensor(shape: shape)
        buffer = tensor.gpuBuffer
        accumulates = false
    }

    /// Stores the gradient in the accumulated gradient.
    consuming func finish(into accumulator: inout Tensor<N, GPU>?) {
        accumulator = graphAccumulator
        Tensor.accumulate(tensor, into: &accumulator)
    }
}
#endif

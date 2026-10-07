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

// The fused operations do not accept empty buffers, see `FusedOperationsType`. Every kernel reads its buffers through
// `gpuBuffer`, so this is the one place that checks it.

extension ShapedBuffer where Device == GPU {
    /// Region of the storage that holds the elements of the buffer. The buffer must not be empty.
    var gpuBuffer: GPUBuffer {
        precondition(count > 0, "The fused operations do not accept empty buffers.")
        return values.memory
    }
}

extension MutableShapedBuffer where Device == GPU {
    /// Region of the storage that holds the elements of the buffer. The buffer must not be empty.
    var gpuBuffer: GPUBuffer {
        precondition(count > 0, "The fused operations do not accept empty buffers.")
        return values.memory
    }
}

extension GradientBuffer where Device == GPU {
    /// Region of the storage that holds the elements of the gradient.
    var gpuBuffer: GPUBuffer {
        values.gpuBuffer
    }

    /// Value of the `accumulate` parameter of a kernel: 1 when the kernel adds to the elements, 0 when it stores into them.
    var accumulateFlag: UInt32 {
        adds ? 1 : 0
    }
}

/// Helpers of the fused operations of the GPU.
enum GPUFused {
    // A small operation whose operands are available on the host uses the default implementation, whose basic operations then
    // run on the host.
    /// Whether a fused operation runs its GPU kernel: for floats, unless the operation runs on the host, see ``GPUPlacement``.
    static func runsKernel<N>(_: N.Type, elements: Int, reading: [GPUBuffer], writing: [GPUBuffer]) -> Bool {
        N.self == Float.self && !GPUPlacement.runsOnHost(elements: elements, reading: reading, writing: writing)
    }

    // Short rows get one SIMD group, so that no threadgroup synchronization is necessary. Longer rows get more threads, so that
    // every thread reads a few elements, up to 256 threads, which keeps several threadgroups resident on a GPU core.
    /// Number of threads of a threadgroup that processes one row of the given length.
    static func rowThreadgroupWidth(length: Int) -> Int {
        length <= 64 ? 32 : length <= 512 ? 128 : 256
    }

    /// Length of the parameter of an element-wise kernel, or nil when the kernel does not support the shape of the parameter.
    ///
    /// The kernels support a parameter with one element and a parameter with the shape of the last axes of the input.
    static func parameterLength(_ parameter: ShapedBuffer<some Any, GPU>, input: ShapedBuffer<some Any, GPU>) -> Int? {
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

    /// Records the forward kernel of an activation, or calls `fallback` when the activation runs on the host or the kernel
    /// does not support the shape of the parameter.
    static func activation<N>(_ name: String, input: ShapedBuffer<N, GPU>, parameter: ShapedBuffer<N, GPU>? = nil, result: MutableShapedBuffer<N, GPU>, fallback: () -> Void) {
        let (x, y, a) = (input.gpuBuffer, result.gpuBuffer, parameter?.gpuBuffer ?? input.gpuBuffer)
        let length = parameter.map { parameterLength($0, input: input) } ?? 1
        guard let length, runsKernel(N.self, elements: input.count, reading: [x, a], writing: [y]) else {
            fallback()
            return
        }
        let parameters = ElementwiseParameters(count: UInt32(input.count), parameterLength: UInt32(length), accumulate: 0)
        GPUContext.compute(GPUKernels.kernel("\(name)_forward", in: .fused), reading: [x, a], writing: [y]) { arguments in
            arguments.buffer(x)
            arguments.buffer(y)
            arguments.buffer(a)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
    }

    /// Records the backward kernel of an activation, which adds the gradient to the accumulated gradient or stores it, or calls
    /// `fallback` when the activation runs on the host or the kernel does not support the shape of the parameter.
    ///
    /// - Parameter input: Input of the forward operation, or its result for activations whose gradient uses the result.
    static func activationBackward<N>(
        _ name: String,
        input: ShapedBuffer<N, GPU>,
        outputGradient: ShapedBuffer<N, GPU>,
        parameter: ShapedBuffer<N, GPU>? = nil,
        inputGradient: GradientBuffer<N, GPU>?,
        fallback: () -> Void,
    ) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g, dx, a) = (input.gpuBuffer, outputGradient.gpuBuffer, inputGradient.gpuBuffer, parameter?.gpuBuffer ?? input.gpuBuffer)
        let length = parameter.map { parameterLength($0, input: input) } ?? 1
        guard let length, runsKernel(N.self, elements: input.count, reading: [x, g, a], writing: [dx]) else {
            fallback()
            return
        }
        let parameters = ElementwiseParameters(count: UInt32(input.count), parameterLength: UInt32(length), accumulate: inputGradient.accumulateFlag)
        GPUContext.compute(GPUKernels.kernel("\(name)_backward", in: .fused), reading: [x, g, a, dx], writing: [dx]) { arguments in
            arguments.buffer(x)
            arguments.buffer(g)
            arguments.buffer(dx)
            arguments.buffer(a)
            arguments.value(parameters)
            arguments.dispatch(count: input.count)
        }
    }

    struct RowParameters {
        var length: UInt32
        var accumulate: UInt32
        var epsilon: Float
    }
}
#endif

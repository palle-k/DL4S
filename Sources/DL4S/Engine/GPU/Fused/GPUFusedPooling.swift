//
//  GPUFusedPooling.swift
//  DL4S
//
//  Created by Palle Klewitz on 29.09.26.
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

// Pooling runs as one kernel per pass on planes of the images, with the semantics of the CPU: padding elements are zeros
// that take part in the maximum and count for the mean. The backward kernels run one thread per input element and add
// the gradients of the windows that contain it, so they need no atomics.

public extension GPUFusedOperations {
    static func maxPooling2d<N: NumericType>(input: ShapedBuffer<N, GPU>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, GPU>) {
        let pooling = GPUPooling(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(result.shape == pooling.outputShape, "The result must have the shape of the pooled images.")
        guard GPUFused.runsKernel(N.self, elements: result.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.maxPooling2d(input: input, windowSize: windowSize, padding: padding, stride: stride, result: result)
            return
        }
        pooling.forward("max_pool_forward", input: input.gpuBuffer, result: result.gpuBuffer)
    }

    static func maxPooling2dBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        let pooling = GPUPooling(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(outputGradient.shape == pooling.outputShape, "The gradient of the result must have the shape of the result.")
        // The kernels store the position of a maximum in its window in one byte, which supports windows of up to 15 x 15 elements.
        guard windowSize <= 15, GPUFused.runsKernel(N.self, elements: input.count, reading: [input.gpuBuffer, outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.maxPooling2dBackward(input: input, outputGradient: outputGradient, windowSize: windowSize, padding: padding, stride: stride, inputGradient: inputGradient)
            return
        }
        // The positions of the maxima of the windows go through a temporary buffer, so that the backward kernel does not scan
        // every window once for every element that it contains.
        let positions = GPUKernels.temporary(byteCount: outputGradient.count)
        pooling.forward("max_pool_positions", input: input.gpuBuffer, result: positions)
        pooling.backward("max_pool_backward", windows: positions, outputGradient: outputGradient.gpuBuffer, inputGradient: inputGradient)
    }

    static func averagePooling2d<N: NumericType>(input: ShapedBuffer<N, GPU>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, GPU>) {
        let pooling = GPUPooling(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(result.shape == pooling.outputShape, "The result must have the shape of the pooled images.")
        guard GPUFused.runsKernel(N.self, elements: result.count, reading: [input.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.averagePooling2d(input: input, windowSize: windowSize, padding: padding, stride: stride, result: result)
            return
        }
        pooling.forward("average_pool_forward", input: input.gpuBuffer, result: result.gpuBuffer)
    }

    static func averagePooling2dBackward<N: NumericType>(input: ShapedBuffer<N, GPU>, outputGradient: ShapedBuffer<N, GPU>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, GPU>?) {
        guard let inputGradient else {
            return
        }
        let pooling = GPUPooling(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(outputGradient.shape == pooling.outputShape, "The gradient of the result must have the shape of the result.")
        guard GPUFused.runsKernel(N.self, elements: input.count, reading: [outputGradient.gpuBuffer]) else {
            DefaultFusedOperations<GPU>.averagePooling2dBackward(input: input, outputGradient: outputGradient, windowSize: windowSize, padding: padding, stride: stride, inputGradient: inputGradient)
            return
        }
        pooling.backward("average_pool_backward", windows: nil, outputGradient: outputGradient.gpuBuffer, inputGradient: inputGradient)
    }
}

/// Shapes of a pooling operation on the GPU.
private struct GPUPooling {
    let planes: Int
    let parameters: PoolingParameters

    let outputShape: [Int]

    /// The shapes of a pooling operation. The arguments must have the shapes that ``FusedOperationsType/maxPooling2d(input:windowSize:padding:stride:result:)`` states.
    init(input: ShapedBuffer<some Any, GPU>, windowSize: Int, padding: Int, stride: Int) {
        precondition(input.dim == 4, "The images must have 4 axes.")
        precondition(windowSize > 0 && stride > 0 && padding >= 0, "The window size and the stride must be positive, and the padding must not be negative.")
        precondition(input.shape[2] + 2 * padding >= windowSize && input.shape[3] + 2 * padding >= windowSize, "The windows must fit into the padded images.")
        let outputHeight = ConvUtil.outputSize(inputSize: input.shape[2], kernelSize: windowSize, padding: padding, stride: stride)
        let outputWidth = ConvUtil.outputSize(inputSize: input.shape[3], kernelSize: windowSize, padding: padding, stride: stride)
        planes = input.shape[0] * input.shape[1]
        outputShape = [input.shape[0], input.shape[1], outputHeight, outputWidth]
        parameters = PoolingParameters(
            height: UInt32(input.shape[2]), width: UInt32(input.shape[3]), outputHeight: UInt32(outputHeight), outputWidth: UInt32(outputWidth),
            windowSize: UInt32(windowSize), padding: Int32(padding), stride: UInt32(stride), accumulate: 0, scale: 1 / Float(windowSize * windowSize), inverseStride: 1 / Float(stride),
        )
    }

    /// Records a kernel with one thread per output element, which writes the result or the positions of the maxima.
    func forward(_ name: String, input: GPUBuffer, result: GPUBuffer) {
        let threads = MTLSize(width: Int(parameters.outputWidth), height: Int(parameters.outputHeight), depth: planes)
        GPUContext.current.compute(GPUKernels.pipeline(name, in: .fused), reading: [input], writing: [result]) { arguments in
            arguments.buffer(input)
            arguments.buffer(result)
            arguments.value(parameters)
            arguments.dispatch(threads: threads, threadgroup: Self.threadgroup)
        }
    }

    /// Records a backward kernel with one thread per input element. `windows` is a buffer per window that the kernel reads,
    /// the positions of the maxima for max pooling, and nil for average pooling.
    func backward<N>(_ name: String, windows: GPUBuffer?, outputGradient: GPUBuffer, inputGradient: GradientBuffer<N, GPU>) {
        var parameters = parameters
        parameters.accumulate = inputGradient.accumulateFlag
        let dx = inputGradient.gpuBuffer
        let threads = MTLSize(width: Int(parameters.width), height: Int(parameters.height), depth: planes)
        GPUContext.current.compute(GPUKernels.pipeline(name, in: .fused), reading: [windows, outputGradient, dx].compactMap(\.self), writing: [dx]) { arguments in
            if let windows {
                arguments.buffer(windows)
            }
            arguments.buffer(outputGradient)
            arguments.buffer(dx)
            arguments.value(parameters)
            arguments.dispatch(threads: threads, threadgroup: Self.threadgroup)
        }
    }

    /// Threads along the rows of the planes, so that neighboring threads read neighboring elements.
    private static let threadgroup = MTLSize(width: 32, height: 8, depth: 1)
}

private struct PoolingParameters {
    var height: UInt32
    var width: UInt32
    var outputHeight: UInt32
    var outputWidth: UInt32
    var windowSize: UInt32
    var padding: Int32
    var stride: UInt32
    var accumulate: UInt32
    var scale: Float
    var inverseStride: Float
}
#endif

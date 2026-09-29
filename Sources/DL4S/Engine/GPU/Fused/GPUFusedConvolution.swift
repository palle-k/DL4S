//
//  GPUFusedConvolution.swift
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

#if canImport(Metal) && canImport(MetalPerformanceShaders) && canImport(MetalPerformanceShadersGraph)
import Foundation
import MetalPerformanceShadersGraph

// Convolutions with the stride 1 run as Metal Performance Shaders graphs. The graphs select the algorithm for the shapes,
// such as Winograd convolutions for 3 x 3 kernels, and do not write the window matrix of the input to memory.
// A graph is compiled once for every combination of shapes, padding, and computed gradients.
// Strided convolutions use the default implementation, with the window matrix and the matrix kernels.

public extension GPUFusedOperations {
    static func convolution2d<N: NumericType>(input: ShapedBuffer<N, GPU>, filters: ShapedBuffer<N, GPU>, bias: ShapedBuffer<N, GPU>?, padding: Int, stride: Int, result: MutableShapedBuffer<N, GPU>) {
        guard let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride),
              GPUFused.runsKernel(N.self, elements: result.count, reading: [input.gpuBuffer, filters.gpuBuffer] + (bias.map { [$0.gpuBuffer] } ?? []))
        else {
            DefaultFusedOperations<GPU>.convolution2d(input: input, filters: filters, bias: bias, padding: padding, stride: stride, result: result)
            return
        }
        let graph = GPUGraphCache.graph(for: geometry.key(outputs: [])) {
            GPUGraph(inputShapes: [input.shape, filters.shape] + (bias.map { [$0.shape] } ?? [])) { graph, inputs in
                let convolved = graph.convolution2D(inputs[0], weights: inputs[1], descriptor: geometry.descriptor, name: nil)
                guard inputs.count == 3 else {
                    return [convolved]
                }
                let bias = graph.reshape(inputs[2], shape: [1, NSNumber(value: filters.shape[0]), 1, 1], name: nil)
                return [graph.addition(convolved, bias, name: nil)]
            }
        }
        graph.encode(
            inputs: [(input.gpuBuffer, input.shape), (filters.gpuBuffer, filters.shape)] + (bias.map { [($0.gpuBuffer, $0.shape)] } ?? []),
            results: [(result.gpuBuffer, result.shape)],
        )
    }

    static func convolution2dBackward<N: NumericType>(
        input: ShapedBuffer<N, GPU>,
        filters: ShapedBuffer<N, GPU>,
        bias: ShapedBuffer<N, GPU>?,
        outputGradient: ShapedBuffer<N, GPU>,
        padding: Int,
        stride: Int,
        inputGradient: GradientBuffer<N, GPU>?,
        filterGradient: GradientBuffer<N, GPU>?,
        biasGradient: GradientBuffer<N, GPU>?,
    ) {
        guard let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride),
              GPUFused.runsKernel(N.self, elements: outputGradient.count, reading: [input.gpuBuffer, filters.gpuBuffer, outputGradient.gpuBuffer])
        else {
            DefaultFusedOperations<GPU>.convolution2dBackward(
                input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: padding, stride: stride,
                inputGradient: inputGradient, filterGradient: filterGradient, biasGradient: biasGradient,
            )
            return
        }
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        let requested = [inputGradient, filterGradient, biasGradient]
        let outputs = requested.map { $0 != nil }
        guard outputs.contains(true) else {
            return
        }
        let math = BufferMath<N, GPU>()
        defer {
            math.release()
        }
        // The graph stores its results, so a gradient that is added to the accumulated gradient goes through an intermediate buffer.
        let gradients = requested.compactMap(\.self)
        let targets = gradients.map { $0.adds ? math.temporary($0.shape) : $0.values }
        let graph = GPUGraphCache.graph(for: geometry.key(outputs: outputs)) {
            GPUGraph(inputShapes: [input.shape, filters.shape, outputGradient.shape]) { graph, inputs in
                let (x, w, g) = (inputs[0], inputs[1], inputs[2])
                var results: [MPSGraphTensor] = []
                if outputs[0] {
                    results.append(graph.convolution2DDataGradient(g, weights: w, outputShape: x.shape!, forwardConvolutionDescriptor: geometry.descriptor, name: nil))
                }
                if outputs[1] {
                    results.append(graph.convolution2DWeightsGradient(g, source: x, outputShape: w.shape!, forwardConvolutionDescriptor: geometry.descriptor, name: nil))
                }
                if outputs[2] {
                    let sum = graph.reductionSum(with: g, axes: [0, 2, 3], name: nil)
                    results.append(graph.reshape(sum, shape: [NSNumber(value: filters.shape[0])], name: nil))
                }
                return results
            }
        }
        graph.encode(
            inputs: [(input.gpuBuffer, input.shape), (filters.gpuBuffer, filters.shape), (outputGradient.gpuBuffer, outputGradient.shape)],
            results: targets.map { ($0.gpuBuffer, $0.shape) },
        )
        for (gradient, target) in zip(gradients, targets) where gradient.adds {
            math.add(gradient.values, target, into: gradient.values)
        }
    }
}

/// Shapes and parameters of a convolution on the GPU.
private struct GPUConvolution {
    let inputShape: [Int]
    let filterShape: [Int]
    let outputShape: [Int]
    let hasBias: Bool
    let padding: Int
    let stride: Int

    /// Describes the convolution, or returns nil for a stride other than 1, which the graphs do not support.
    ///
    /// The arguments must have the shapes that ``FusedOperationsType/convolution2d(input:filters:bias:padding:stride:result:)`` states.
    init?(input: ShapedBuffer<some Any, GPU>, filters: ShapedBuffer<some Any, GPU>, bias: ShapedBuffer<some Any, GPU>?, padding: Int, stride: Int) {
        precondition(input.dim == 4 && filters.dim == 4, "The images and the filters must have 4 axes.")
        precondition(input.shape[1] == filters.shape[1], "The images must have one channel for every input channel of the filters.")
        precondition(bias.map { $0.shape == [filters.shape[0]] } ?? true, "The bias must have one element for every output channel.")
        precondition(stride > 0 && padding >= 0, "The stride must be positive, and the padding must not be negative.")
        precondition(input.shape[2] + 2 * padding >= filters.shape[2] && input.shape[3] + 2 * padding >= filters.shape[3], "The filters must fit into the padded images.")
        // The graphs reach a low throughput for strided convolutions, for which the window matrix and the matrix kernels are faster.
        guard stride == 1 else {
            return nil
        }
        inputShape = input.shape
        filterShape = filters.shape
        outputShape = [
            input.shape[0],
            filters.shape[0],
            ConvUtil.outputSize(inputSize: input.shape[2], kernelSize: filters.shape[2], padding: padding, stride: stride),
            ConvUtil.outputSize(inputSize: input.shape[3], kernelSize: filters.shape[3], padding: padding, stride: stride),
        ]
        hasBias = bias != nil
        self.padding = padding
        self.stride = stride
    }

    var descriptor: MPSGraphConvolution2DOpDescriptor {
        MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW,
        )!
    }

    /// Key of the graph that computes the given gradients, or the forward pass for no gradients.
    func key(outputs: [Bool]) -> GPUConvolutionKey {
        GPUConvolutionKey(inputShape: inputShape, filterShape: filterShape, hasBias: hasBias, padding: padding, stride: stride, outputs: outputs)
    }
}

private struct GPUConvolutionKey: Hashable, Sendable {
    let inputShape: [Int]
    let filterShape: [Int]
    let hasBias: Bool
    let padding: Int
    let stride: Int
    let outputs: [Bool]
}
#endif

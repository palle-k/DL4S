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

// Convolutions run as Metal Performance Shaders graphs, except for passes of strided convolutions for which the graphs are
// slow. The graphs select the algorithm for the shapes, such as Winograd convolutions for 3 x 3 kernels, and do not write the
// window matrix of the input to memory. A graph is compiled once for every combination of shapes, padding, stride, and
// computed gradients.
//
// The other passes, with the throughput at the strided layers of ResNet (M3 Max):
// - The forward pass of a strided convolution with few input channels, such as the first 7 x 7 layer of ResNet, runs as an
//   implicit matrix product (`GPUImplicitConvolution`): 3.9 TFLOPS, the graph 2.9.
// - The data gradient of a strided convolution with a kernel larger than 1 x 1 runs as implicit products, one per phase of the
//   stride, with many input channels (3.0 to 3.7 TFLOPS, the graph 1.7 to 1.9), and with the window matrix and the matrix
//   kernels of the default implementation with few (1.7 TFLOPS: the implicit products spend most of their tiles on rows that
//   do not exist).
// - The weights gradient of a strided convolution with few input channels is the weights gradient of a convolution with the
//   stride 1 on the space-to-depth input, whose blocks of stride x stride pixels become channels: 2.7 TFLOPS for the first
//   layer of ResNet, the strided graph 2.1.

public extension GPUFusedOperations {
    static func convolution2d<N: NumericType>(input: ShapedBuffer<N, GPU>, filters: ShapedBuffer<N, GPU>, bias: ShapedBuffer<N, GPU>?, padding: Int, stride: Int, result: MutableShapedBuffer<N, GPU>) {
        let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        precondition(result.shape == geometry.outputShape, "The result must have the shape of the convolved images.")
        guard GPUFused.runsKernel(N.self, elements: result.count, reading: [input.gpuBuffer, filters.gpuBuffer] + (bias.map { [$0.gpuBuffer] } ?? []))
        else {
            DefaultFusedOperations<GPU>.convolution2d(input: input, filters: filters, bias: bias, padding: padding, stride: stride, result: result)
            return
        }
        if geometry.forwardIsImplicit {
            geometry.implicitConvolution.forward(input: input.gpuBuffer, filters: filters.gpuBuffer, bias: bias?.gpuBuffer, result: result.gpuBuffer)
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
        graph.encode(inputs: [input, filters] + (bias.map { [$0] } ?? []), results: [result])
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
        let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        guard GPUFused.runsKernel(N.self, elements: outputGradient.count, reading: [input.gpuBuffer, filters.gpuBuffer, outputGradient.gpuBuffer])
        else {
            DefaultFusedOperations<GPU>.convolution2dBackward(
                input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: padding, stride: stride,
                inputGradient: inputGradient, filterGradient: filterGradient, biasGradient: biasGradient,
            )
            return
        }
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        if let inputGradient {
            switch geometry.dataGradientPath {
            case .graph:
                break
            case .implicit:
                geometry.implicitConvolution.dataGradient(outputGradient: outputGradient.gpuBuffer, filters: filters.gpuBuffer, inputGradient: inputGradient.gpuBuffer, accumulate: inputGradient.adds)
            case .windowMatrix:
                DefaultFusedOperations<GPU>.convolution2dBackward(
                    input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: padding, stride: stride,
                    inputGradient: inputGradient, filterGradient: nil, biasGradient: nil,
                )
            }
        }
        let requested = [geometry.dataGradientPath == .graph ? inputGradient : nil, filterGradient, biasGradient]
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
                    results.append(geometry.weightsGradientUsesSpaceToDepth
                        ? geometry.spaceToDepthWeightsGradient(graph, outputGradient: g, input: x)
                        : graph.convolution2DWeightsGradient(g, source: x, outputShape: w.shape!, forwardConvolutionDescriptor: geometry.descriptor, name: nil))
                }
                if outputs[2] {
                    let sum = graph.reductionSum(with: g, axes: [0, 2, 3], name: nil)
                    results.append(graph.reshape(sum, shape: [NSNumber(value: filters.shape[0])], name: nil))
                }
                return results
            }
        }
        graph.encode(inputs: [input, filters, outputGradient], results: targets)
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

    /// Describes the convolution. The arguments must have the shapes that ``FusedOperationsType/convolution2d(input:filters:bias:padding:stride:result:)`` states.
    init(input: ShapedBuffer<some Any, GPU>, filters: ShapedBuffer<some Any, GPU>, bias: ShapedBuffer<some Any, GPU>?, padding: Int, stride: Int) {
        precondition(input.dim == 4 && filters.dim == 4, "The images and the filters must have 4 axes.")
        precondition(input.shape[1] == filters.shape[1], "The images must have one channel for every input channel of the filters.")
        precondition(bias.map { $0.shape == [filters.shape[0]] } ?? true, "The bias must have one element for every output channel.")
        precondition(stride > 0 && padding >= 0, "The stride must be positive, and the padding must not be negative.")
        precondition(input.shape[2] + 2 * padding >= filters.shape[2] && input.shape[3] + 2 * padding >= filters.shape[3], "The filters must fit into the padded images.")
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

    /// Algorithm of the data gradient.
    enum DataGradientPath {
        case graph
        case implicit
        case windowMatrix
    }

    /// Number of input channels from which a strided convolution uses the graph for its forward pass and implicit products
    /// for its data gradient.
    private static let manyInputChannels = 16

    /// Whether the forward pass runs as an implicit matrix product.
    var forwardIsImplicit: Bool {
        stride > 1 && filterShape[2] * filterShape[3] > 1 && inputShape[1] < Self.manyInputChannels
    }

    var dataGradientPath: DataGradientPath {
        guard stride > 1, filterShape[2] * filterShape[3] > 1 else {
            return .graph
        }
        return inputShape[1] >= Self.manyInputChannels ? .implicit : .windowMatrix
    }

    /// Whether the weights gradient is computed on the space-to-depth input, with the stride 1.
    var weightsGradientUsesSpaceToDepth: Bool {
        forwardIsImplicit
    }

    /// Kernel size of the convolution with the stride 1 on the space-to-depth input: the filters, padded with zeros to a
    /// multiple of the stride, cover `stride` x `stride` blocks.
    private var blockKernelSize: (height: Int, width: Int) {
        ((filterShape[2] + stride - 1) / stride, (filterShape[3] + stride - 1) / stride)
    }

    /// Records the weights gradient of the convolution on the space-to-depth input and rearranges it into the filters.
    ///
    /// The input is padded, or cropped, so that the windows of the convolution with the stride 1 start at the windows of the
    /// strided convolution. The gradient of the padding taps of the filters is dropped.
    func spaceToDepthWeightsGradient(_ graph: MPSGraph, outputGradient: MPSGraphTensor, input: MPSGraphTensor) -> MPSGraphTensor {
        let (kernelHeight, kernelWidth) = blockKernelSize
        let (paddedHeight, paddedWidth) = (stride * (outputShape[2] + kernelHeight - 1), stride * (outputShape[3] + kernelWidth - 1))
        let (bottom, right) = (paddedHeight - inputShape[2] - padding, paddedWidth - inputShape[3] - padding)
        var padded = graph.padTensor(
            input, with: .constant,
            leftPadding: [0, 0, NSNumber(value: padding), NSNumber(value: padding)],
            rightPadding: [0, 0, NSNumber(value: Swift.max(bottom, 0)), NSNumber(value: Swift.max(right, 0))],
            constantValue: 0, name: nil,
        )
        if bottom < 0 || right < 0 {
            padded = graph.sliceTensor(padded, starts: [0, 0, 0, 0], ends: [inputShape[0], inputShape[1], paddedHeight, paddedWidth].map { NSNumber(value: $0) }, strides: [1, 1, 1, 1], name: nil)
        }
        let blocks = graph.space(toDepth2DTensor: padded, widthAxis: 3, heightAxis: 2, depthAxis: 1, blockSize: stride, usePixelShuffleOrder: true, name: nil)
        let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1, strideInY: 1, dilationRateInX: 1, dilationRateInY: 1, groups: 1,
            paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0, paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW,
        )!
        let (outputChannels, inputChannels) = (filterShape[0], filterShape[1])
        let blockGradient = graph.convolution2DWeightsGradient(
            outputGradient, source: blocks,
            outputShape: [outputChannels, inputChannels * stride * stride, kernelHeight, kernelWidth].map { NSNumber(value: $0) },
            forwardConvolutionDescriptor: descriptor, name: nil,
        )
        // [outputChannels, inputChannels, stride, stride, kernelHeight, kernelWidth], in the channel order of the pixel shuffle,
        // to [outputChannels, inputChannels, kernelHeight * stride, kernelWidth * stride]
        let split = graph.reshape(blockGradient, shape: [outputChannels, inputChannels, stride, stride, kernelHeight, kernelWidth].map { NSNumber(value: $0) }, name: nil)
        let interleaved = graph.reshape(
            graph.transpose(split, permutation: [0, 1, 4, 2, 5, 3], name: nil),
            shape: [outputChannels, inputChannels, kernelHeight * stride, kernelWidth * stride].map { NSNumber(value: $0) }, name: nil,
        )
        return graph.sliceTensor(interleaved, starts: [0, 0, 0, 0], ends: filterShape.map { NSNumber(value: $0) }, strides: [1, 1, 1, 1], name: nil)
    }

    var implicitConvolution: GPUImplicitConvolution {
        GPUImplicitConvolution(inputShape: inputShape, filterShape: filterShape, outputShape: outputShape, padding: padding, stride: stride)
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

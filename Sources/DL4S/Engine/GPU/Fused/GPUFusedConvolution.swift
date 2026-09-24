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
    static func convolution2d<N: NumericType>(input: Tensor<N, GPU>, filters: Tensor<N, GPU>, bias: Tensor<N, GPU>?, padding: Int, stride: Int) -> Tensor<N, GPU> {
        guard let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride),
              GPUFused.runsKernel(N.self, elements: geometry.outputShape.reduce(1, *), reading: [input, filters] + (bias.map { [$0] } ?? []))
        else {
            return DefaultFusedOperations<GPU>.convolution2d(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        }
        let result: Tensor<N, GPU> = GPUFused.makeTensor(shape: geometry.outputShape)
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
        return result
    }

    static func convolution2dBackward<N: NumericType>(input: Tensor<N, GPU>, filters: Tensor<N, GPU>, bias: Tensor<N, GPU>?, outputGradient: Tensor<N, GPU>, padding: Int, stride: Int, accumulating gradients: inout (input: Tensor<N, GPU>?, filters: Tensor<N, GPU>?, bias: Tensor<N, GPU>?)) {
        guard let geometry = GPUConvolution(input: input, filters: filters, bias: bias, padding: padding, stride: stride), outputGradient.shape == geometry.outputShape,
              GPUFused.runsKernel(N.self, elements: outputGradient.count, reading: [input, filters, outputGradient])
        else {
            DefaultFusedOperations<GPU>.convolution2dBackward(input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: padding, stride: stride, accumulating: &gradients)
            return
        }
        let outputs = [input.requiresGradient, filters.requiresGradient, bias?.requiresGradient ?? false]
        guard outputs.contains(true) else {
            return
        }
        let shapes = [input.shape, filters.shape, [filters.shape[0]]]
        let computed: [Tensor<N, GPU>?] = zip(outputs, shapes).map { computes, shape in computes ? GPUFused.makeTensor(shape: shape) : nil }
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
            results: computed.compactMap { $0.map { ($0.gpuBuffer, $0.shape) } },
        )
        Tensor.accumulate(computed[0], into: &gradients.input)
        Tensor.accumulate(computed[1], into: &gradients.filters)
        Tensor.accumulate(computed[2], into: &gradients.bias)
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

    /// Describes the convolution, or returns nil when the graphs do not support the shapes.
    init?(input: Tensor<some Any, GPU>, filters: Tensor<some Any, GPU>, bias: Tensor<some Any, GPU>?, padding: Int, stride: Int) {
        // The graphs reach a low throughput for strided convolutions, for which the window matrix and the matrix kernels are faster.
        guard input.dim == 4, filters.dim == 4, input.shape[1] == filters.shape[1], stride == 1, padding >= 0 else {
            return nil
        }
        if let bias, bias.shape != [filters.shape[0]] {
            return nil
        }
        let outputHeight = (input.shape[2] + 2 * padding - filters.shape[2]) / stride + 1
        let outputWidth = (input.shape[3] + 2 * padding - filters.shape[3]) / stride + 1
        guard outputHeight > 0, outputWidth > 0 else {
            return nil
        }
        inputShape = input.shape
        filterShape = filters.shape
        outputShape = [input.shape[0], filters.shape[0], outputHeight, outputWidth]
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

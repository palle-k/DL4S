//
//  FusedConvolution.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
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

import Foundation

/// Operations composed from differentiable tensor operations.
///
/// The default implementations of fused operations call these functions with tensors without context,
/// so the results have no context. Tensor operations call them with tensors of the compute graph
/// when the gradient must itself be differentiable.
enum Composed {}

// MARK: Default implementations

public extension FusedOperationsType {
    static func convolution2d<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, padding: Int, stride: Int) -> Tensor<N, Device> {
        let input = input.detached()
        let filters = filters.detached()
        let outputChannels = filters.shape[0]
        let outputShape = [
            input.shape[0],
            outputChannels,
            (input.shape[2] + 2 * padding - filters.shape[2]) / stride + 1,
            (input.shape[3] + 2 * padding - filters.shape[3]) / stride + 1,
        ]

        // [inputChannels * kernelHeight * kernelWidth, batchSize * outputHeight * outputWidth]
        let columns = input.img2col(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
        let convolved = filters
            .view(as: [outputChannels, -1])
            .matrixMultiplied(with: columns) // [outputChannels, batchSize * outputHeight * outputWidth]
            .view(as: [outputChannels, outputShape[0], outputShape[2], outputShape[3]])
            .permuted(to: [1, 0, 2, 3])

        guard let bias else {
            return convolved
        }
        return convolved + bias.detached().view(as: [1, outputChannels, 1, 1])
    }

    static func convolution2dBackward<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, padding: Int, stride: Int) -> (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?) {
        Composed.convolution2dGradients(
            input: input.detached(),
            filters: filters.detached(),
            outputGradient: outputGradient.detached(),
            padding: padding,
            stride: stride,
            computesInput: input.requiresGradient,
            computesFilters: filters.requiresGradient,
            computesBias: bias?.requiresGradient ?? false,
        )
    }

    static func transposedConvolution2d<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, inset: Int, stride: Int) -> Tensor<N, Device> {
        let input = input.detached()
        let filters = filters.detached()
        let outputChannels = filters.shape[0]
        let outputShape = [
            input.shape[0],
            outputChannels,
            (input.shape[2] - 1) * stride - 2 * inset + filters.shape[2],
            (input.shape[3] - 1) * stride - 2 * inset + filters.shape[3],
        ]

        // A transposed convolution is the input gradient of a convolution.
        let inputMatrix = input.permuted(to: [1, 0, 2, 3]).view(as: [input.shape[1], -1]) // [inputChannels, batchSize * height * width]
        let columns = filters
            .view(as: [filters.shape[1], -1]) // [inputChannels, outputChannels * kernelHeight * kernelWidth]
            .matrixMultiplied(with: inputMatrix, transposeSelf: true) // [outputChannels * kernelHeight * kernelWidth, batchSize * height * width]
        let convolved = columns.col2img(
            kernelWidth: filters.shape[3],
            kernelHeight: filters.shape[2],
            padding: inset,
            stride: stride,
            resultShape: outputShape,
        )

        guard let bias else {
            return convolved
        }
        return convolved + bias.detached().view(as: [1, outputChannels, 1, 1])
    }

    static func transposedConvolution2dBackward<N: NumericType>(input: Tensor<N, Device>, filters: Tensor<N, Device>, bias: Tensor<N, Device>?, outputGradient: Tensor<N, Device>, inset: Int, stride: Int) -> (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?) {
        Composed.transposedConvolution2dGradients(
            input: input.detached(),
            filters: filters.detached(),
            outputGradient: outputGradient.detached(),
            inset: inset,
            stride: stride,
            computesInput: input.requiresGradient,
            computesFilters: filters.requiresGradient,
            computesBias: bias?.requiresGradient ?? false,
        )
    }

    static func maxPooling2d<N: NumericType>(input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        Composed.poolingColumns(of: input.detached(), windowSize: windowSize, padding: padding, stride: stride)
            .reduceMax(along: [0])
            .view(as: Composed.poolingOutputShape(of: input, windowSize: windowSize, padding: padding, stride: stride))
    }

    static func maxPooling2dBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        Composed.maxPooling2dGradient(input: input.detached(), outputGradient: outputGradient.detached(), windowSize: windowSize, padding: padding, stride: stride)
    }

    static func averagePooling2d<N: NumericType>(input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        Composed.poolingColumns(of: input.detached(), windowSize: windowSize, padding: padding, stride: stride)
            .reduceSum(along: [0])
            .view(as: Composed.poolingOutputShape(of: input, windowSize: windowSize, padding: padding, stride: stride))
            / Tensor(N(windowSize * windowSize))
    }

    static func averagePooling2dBackward<N: NumericType>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        Composed.averagePooling2dGradient(inputShape: input.shape, outputGradient: outputGradient.detached(), windowSize: windowSize, padding: padding, stride: stride)
    }
}

// MARK: Composed gradients

extension Composed {
    /// Computes the gradients of a 2D convolution.
    static func convolution2dGradients<N, Device>(
        input: Tensor<N, Device>,
        filters: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        padding: Int,
        stride: Int,
        computesInput: Bool,
        computesFilters: Bool,
        computesBias: Bool,
    ) -> (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?) {
        let outputChannels = filters.shape[0]
        let kernelHeight = filters.shape[2]
        let kernelWidth = filters.shape[3]

        // The layout of the matrix product of the forward pass: [outputChannels, batchSize * outputHeight * outputWidth]
        let gradientMatrix = outputGradient.permuted(to: [1, 0, 2, 3]).view(as: [outputChannels, -1])

        let inputGradient = computesInput ? filters
            .view(as: [outputChannels, -1])
            .matrixMultiplied(with: gradientMatrix, transposeSelf: true)
            .col2img(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride, resultShape: input.shape) : nil

        // The windows are extracted again instead of being kept alive between the forward and the backward pass.
        let filterGradient = computesFilters ? gradientMatrix
            .matrixMultiplied(with: input.img2col(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride), transposeOther: true)
            .view(as: filters.shape) : nil

        let biasGradient = computesBias ? outputGradient.reduceSum(along: [0, 2, 3]) : nil

        return (inputGradient, filterGradient, biasGradient)
    }

    /// Computes the gradients of a transposed 2D convolution.
    static func transposedConvolution2dGradients<N, Device>(
        input: Tensor<N, Device>,
        filters: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inset: Int,
        stride: Int,
        computesInput: Bool,
        computesFilters: Bool,
        computesBias: Bool,
    ) -> (input: Tensor<N, Device>?, filters: Tensor<N, Device>?, bias: Tensor<N, Device>?) {
        let inputChannels = input.shape[1]

        // img2col is the adjoint of the col2img of the forward pass.
        // [outputChannels * kernelHeight * kernelWidth, batchSize * height * width]
        let columnGradient = outputGradient.img2col(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: inset, stride: stride)
        let filterMatrix = filters.view(as: [inputChannels, -1])

        let inputGradient = computesInput ? filterMatrix
            .matrixMultiplied(with: columnGradient) // [inputChannels, batchSize * height * width]
            .view(as: [inputChannels, input.shape[0], input.shape[2], input.shape[3]])
            .permuted(to: [1, 0, 2, 3]) : nil

        let filterGradient = computesFilters ? input
            .permuted(to: [1, 0, 2, 3])
            .view(as: [inputChannels, -1])
            .matrixMultiplied(with: columnGradient, transposeOther: true) // [inputChannels, outputChannels * kernelHeight * kernelWidth]
            .view(as: filters.shape) : nil

        let biasGradient = computesBias ? outputGradient.reduceSum(along: [0, 2, 3]) : nil

        return (inputGradient, filterGradient, biasGradient)
    }

    /// Computes the gradient of 2D max pooling.
    ///
    /// The input only selects the positions of the largest values, so the gradient is not differentiable with respect to it.
    static func maxPooling2dGradient<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        let imageShape = [input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3]]
        let positions = poolingColumns(of: input.detached(), windowSize: windowSize, padding: padding, stride: stride).argmax(along: 0)
        return outputGradient
            .view(as: [-1])
            .scatter(using: positions, alongAxis: 0, withSize: windowSize * windowSize)
            .col2img(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride, resultShape: imageShape)
            .view(as: input.shape)
    }

    /// Computes the gradient of 2D average pooling.
    static func averagePooling2dGradient<N, Device>(inputShape: [Int], outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        let imageShape = [inputShape[0] * inputShape[1], 1, inputShape[2], inputShape[3]]
        let windowElements = windowSize * windowSize
        let weights = Tensor<N, Device>(repeating: N.one / N(windowElements), shape: [windowElements, 1])
        return (weights * outputGradient.view(as: [1, -1]))
            .col2img(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride, resultShape: imageShape)
            .view(as: inputShape)
    }

    /// Extracts the pooling windows of every channel, shape [windowSize \* windowSize, batchSize \* channels \* outputHeight \* outputWidth].
    static func poolingColumns<N, Device>(of input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, Device> {
        input
            .view(as: [input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3]])
            .img2col(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride)
    }

    /// Shape of the result of 2D pooling.
    static func poolingOutputShape<N, Device>(of input: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int) -> [Int] {
        [
            input.shape[0],
            input.shape[1],
            (input.shape[2] + 2 * padding - windowSize) / stride + 1,
            (input.shape[3] + 2 * padding - windowSize) / stride + 1,
        ]
    }
}

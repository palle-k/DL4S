//
//  ComposedConvolution.swift
//  DL4S
//
//  Created by Palle Klewitz on 28.09.26.
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

// MARK: Composed gradients

extension Composed {
    static func convolution2dBackward<N, Device>(
        input: Tensor<N, Device>,
        filters: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        padding: Int,
        stride: Int,
        inputGradient: inout GradientAccumulator<N, Device>,
        filterGradient: inout GradientAccumulator<N, Device>,
        biasGradient: inout GradientAccumulator<N, Device>,
    ) {
        let outputChannels = filters.shape[0]
        let kernelHeight = filters.shape[2]
        let kernelWidth = filters.shape[3]

        // The layout of the matrix product of the forward pass: [outputChannels, batchSize * outputHeight * outputWidth]
        let gradientMatrix = outputGradient.permuted(to: [1, 0, 2, 3]).view(as: [outputChannels, -1])

        if inputGradient.isRequested {
            inputGradient.add(
                filters
                    .view(as: [outputChannels, -1])
                    .matrixMultiplied(with: gradientMatrix, transposeSelf: true)
                    .col2img(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride, resultShape: input.shape),
            )
        }
        if filterGradient.isRequested {
            // The windows are extracted again instead of being kept alive between the forward and the backward pass.
            let columns = input.img2col(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
            filterGradient.add(gradientMatrix.matrixMultiplied(with: columns, transposeOther: true).view(as: filters.shape))
        }
        if biasGradient.isRequested {
            biasGradient.add(outputGradient.reduceSum(along: [0, 2, 3]))
        }
    }

    static func transposedConvolution2dBackward<N, Device>(
        input: Tensor<N, Device>,
        filters: Tensor<N, Device>,
        outputGradient: Tensor<N, Device>,
        inset: Int,
        stride: Int,
        inputGradient: inout GradientAccumulator<N, Device>,
        filterGradient: inout GradientAccumulator<N, Device>,
        biasGradient: inout GradientAccumulator<N, Device>,
    ) {
        let inputChannels = input.shape[1]

        // img2col is the adjoint of the col2img of the forward pass.
        // [outputChannels * kernelHeight * kernelWidth, batchSize * height * width]
        let columnGradient = outputGradient.img2col(kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: inset, stride: stride)
        let filterMatrix = filters.view(as: [inputChannels, -1])

        if inputGradient.isRequested {
            inputGradient.add(
                filterMatrix
                    .matrixMultiplied(with: columnGradient) // [inputChannels, batchSize * height * width]
                    .view(as: [inputChannels, input.shape[0], input.shape[2], input.shape[3]])
                    .permuted(to: [1, 0, 2, 3]),
            )
        }
        if filterGradient.isRequested {
            filterGradient.add(
                input
                    .permuted(to: [1, 0, 2, 3])
                    .view(as: [inputChannels, -1])
                    .matrixMultiplied(with: columnGradient, transposeOther: true) // [inputChannels, outputChannels * kernelHeight * kernelWidth]
                    .view(as: filters.shape),
            )
        }
        if biasGradient.isRequested {
            biasGradient.add(outputGradient.reduceSum(along: [0, 2, 3]))
        }
    }

    /// The input only selects the positions of the largest values, so the gradient is not differentiable with respect to it.
    static func maxPooling2dBackward<N, Device>(input: Tensor<N, Device>, outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let imageShape = [input.shape[0] * input.shape[1], 1, input.shape[2], input.shape[3]]
        let positions = poolingColumns(of: input.detached(), windowSize: windowSize, padding: padding, stride: stride).argmax(along: 0)
        inputGradient.add(
            outputGradient
                .view(as: [-1])
                .scatter(using: positions, alongAxis: 0, withSize: windowSize * windowSize)
                .col2img(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride, resultShape: imageShape)
                .view(as: input.shape),
        )
    }

    static func averagePooling2dBackward<N, Device>(inputShape: [Int], outputGradient: Tensor<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: inout GradientAccumulator<N, Device>) {
        guard inputGradient.isRequested else {
            return
        }
        let imageShape = [inputShape[0] * inputShape[1], 1, inputShape[2], inputShape[3]]
        let windowElements = windowSize * windowSize
        let weights = Tensor<N, Device>(repeating: N.one / N(windowElements), shape: [windowElements, 1])
        inputGradient.add(
            (weights * outputGradient.view(as: [1, -1]))
                .col2img(kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride, resultShape: imageShape)
                .view(as: inputShape),
        )
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
            ConvUtil.outputSize(inputSize: input.shape[2], kernelSize: windowSize, padding: padding, stride: stride),
            ConvUtil.outputSize(inputSize: input.shape[3], kernelSize: windowSize, padding: padding, stride: stride),
        ]
    }
}

//
//  Conv.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//  Copyright (c) 2019 - Palle Klewitz
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

// MARK: Img2col

public extension Tensor {
    /// Performs an img2col transformation, which allows convolutions to be performed by matrix multiplication.
    ///
    /// The source tensor is expected to have a shape of [batchSize, channels, height, width].
    /// The result is a tensor with shape [window_size, window_count].
    ///
    /// The window size is the size of the kernel (width \* height \* depth).
    /// The window count is the number of windows that fit into the source tensor when using the given padding and stride.
    ///
    /// Windows are layed out from left to right and from top to bottom, where (0, 0) is the top left corner of the image.
    ///
    /// - Parameters:
    ///   - kernelWidth: Width of the convolution kernel
    ///   - kernelHeight: Height of the convolution kernel
    ///   - padding: Padding applied before and after the image in the horizontal and vertical direction
    ///   - stride: Stride, with which the kernel is moved along the image
    func img2col(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) -> Tensor<Element, Device> {
        let resultHeight = (shape[2] + 2 * padding - kernelHeight) / stride + 1
        let resultWidth = (shape[3] + 2 * padding - kernelWidth) / stride + 1

        let resultShape = [
            shape[1] * kernelWidth * kernelHeight,
            resultHeight * resultWidth * shape[0],
        ]

        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.img2col(
            values: values,
            result: resultBuffer,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride,
        )

        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "im2col",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.col2img(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride, resultShape: self.shape)
                }],
            ) : nil,
        )
    }

    /// Computes the inverse of the img2col operation.
    ///
    /// The source tensor is expected to be a tensor with shape [window_size, window_count].
    /// The result tensor will have the given result shape, which is expected to be 4-dimensional ([batch_size, channels, height, width])
    ///
    /// - Parameters:
    ///   - kernelWidth: Width of the convolution kernel
    ///   - kernelHeight: Height of the convolution kernel
    ///   - padding: Padding applied before and after the image in the horizontal and vertical direction
    ///   - stride: Stride, with which the kernel is moved along the image
    ///   - resultShape: Shape of the resulting tensor
    func col2img(kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int, resultShape: [Int]) -> Tensor<Element, Device> {
        let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        Device.Engine.col2img(
            matrix: values,
            image: resultBuffer,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            padding: padding,
            stride: stride,
        )

        return Tensor(
            using: resultBuffer,
            context: requiresGradient ? TensorContext(
                tag: "col2im",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.img2col(kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
                }],
            ) : nil,
        )
    }
}

// MARK: Convolution

public extension Tensor {
    /// Performs a 2d convolution
    ///
    /// the source tensor is expected to have a shape of [batchSize, channels, height, width]
    /// the filters tensor is expected to have a shape of [outputChannels, inputChannels, kernelHeight, kernelWidth]
    ///
    /// - Parameters:
    ///   - filters: Filters to convolve the tensor with
    ///   - bias: Bias with one value per output channel, which is added to the result, or nil for no bias. Any shape with `outputChannels` elements, such as [1, outputChannels, 1, 1], is valid.
    ///   - padding: Padding applied before and after the image in the horizontal and vertical direction
    ///   - stride: Stride, with which the kernel is moved along the image
    /// - Returns: A tensor of shape [batchSize, outputChannels, (height + 2 \* padding - kernelHeight) / stride + 1, (width + 2 \* padding - kernelWidth) / stride + 1)
    func convolved2d(filters: Tensor<Element, Device>, bias: Tensor<Element, Device>? = nil, padding: Int? = nil, stride: Int = 1) -> Tensor<Element, Device> {
        let padding = padding ?? ((filters.shape[2] - 1) / 2)
        let bias = bias.map { bias in
            precondition(bias.count == filters.shape[0], "The bias must have one element per output channel.")
            return bias.view(as: [filters.shape[0]])
        }
        let result = Device.FusedOperations.convolution2d(input: self, filters: filters, bias: bias, padding: padding, stride: stride)

        return result.attachingContext(tag: "conv2d", sources: [self, filters] + (bias.map { [$0] } ?? [])) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.convolution2dGradients(
                    input: self,
                    filters: filters,
                    outputGradient: resultGradient,
                    padding: padding,
                    stride: stride,
                    computesInput: self.requiresGradient,
                    computesFilters: filters.requiresGradient,
                    computesBias: bias?.requiresGradient ?? false,
                )
            } else {
                Device.FusedOperations.convolution2dBackward(input: self, filters: filters, bias: bias, outputGradient: resultGradient, padding: padding, stride: stride)
            }
            return [gradients.input, gradients.filters] + (bias == nil ? [] : [gradients.bias])
        }
    }

    /// Performs a transposed 2d convolution (also called fractionally strided convolution).
    ///
    /// The source tensor is expected to have a shape of [batchSize, channels, height, width]
    /// the filters tensor is expected to have a shape of [outputChannels, inputChannels, kernelHeight, kernelWidth]
    ///
    /// - Parameters:
    ///   - filters: Filters to convolve the tensor with
    ///   - bias: Bias with one value per output channel, which is added to the result, or nil for no bias. Any shape with `outputChannels` elements, such as [1, outputChannels, 1, 1], is valid.
    ///   - inset: Inset from edge of the source tensor
    ///   - stride: Stride, with which the kernel moves over the result image. Larger strides result in larger output shapes.
    /// - Returns: A tensor of shape [batchSize, outputChannels, (height - 1) * stride - 2 \* padding + kernelHeight, (width - 1) * stride - 2 \* padding + kernelWidth]
    func transposedConvolved2d(filters: Tensor<Element, Device>, bias: Tensor<Element, Device>? = nil, inset: Int? = nil, stride: Int = 1) -> Tensor<Element, Device> {
        let inset = inset ?? ((filters.shape[2] - 1) / 2)
        let bias = bias.map { bias in
            precondition(bias.count == filters.shape[0], "The bias must have one element per output channel.")
            return bias.view(as: [filters.shape[0]])
        }
        let result = Device.FusedOperations.transposedConvolution2d(input: self, filters: filters, bias: bias, inset: inset, stride: stride)

        return result.attachingContext(tag: "transposedConv2d", sources: [self, filters] + (bias.map { [$0] } ?? [])) { resultGradient in
            let gradients = if resultGradient.requiresGradient {
                Composed.transposedConvolution2dGradients(
                    input: self,
                    filters: filters,
                    outputGradient: resultGradient,
                    inset: inset,
                    stride: stride,
                    computesInput: self.requiresGradient,
                    computesFilters: filters.requiresGradient,
                    computesBias: bias?.requiresGradient ?? false,
                )
            } else {
                Device.FusedOperations.transposedConvolution2dBackward(input: self, filters: filters, bias: bias, outputGradient: resultGradient, inset: inset, stride: stride)
            }
            return [gradients.input, gradients.filters] + (bias == nil ? [] : [gradients.bias])
        }
    }
}

// MARK: Pooling

public extension Tensor {
    /// Performs max pooling on the tensor. Max pooling selects the maximum value for every given window of a tensor.
    ///
    /// The source tensor is expected to have a shape of [batchSize, channels, height, width].
    ///
    /// - Parameters:
    ///   - windowSize: Window size
    ///   - padding: Padding applied before and after the image in the horizontal and vertical direction
    ///   - stride: Stride, with which the kernel is moved along the image
    /// - Returns: A tensor of shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    func maxPooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        let padding = padding ?? ((windowSize - 1) / 2)
        let stride = stride ?? windowSize
        let result = Device.FusedOperations.maxPooling2d(input: self, windowSize: windowSize, padding: padding, stride: stride)

        return result.attachingContext(tag: "maxPool2d", sources: [self]) { resultGradient in
            if resultGradient.requiresGradient {
                [Composed.maxPooling2dGradient(input: self, outputGradient: resultGradient, windowSize: windowSize, padding: padding, stride: stride)]
            } else {
                [Device.FusedOperations.maxPooling2dBackward(input: self, outputGradient: resultGradient, windowSize: windowSize, padding: padding, stride: stride)]
            }
        }
    }

    /// Performs average pooling on the tensor. Average pooling computes the average of every given window of a tensor.
    ///
    /// The source tensor is expected to have a shape of [batchSize, channels, height, width].
    ///
    /// - Parameters:
    ///   - windowSize: Window size
    ///   - padding: Padding applied before and after the image in the horizontal and vertical direction
    ///   - stride: Stride, with which the kernel is moved along the image
    /// - Returns: A tensor of shape [batchSize, channels, (height + 2 \* padding - windowSize) / stride + 1, (width + 2 \* padding - windowSize) / stride + 1]
    func averagePooled2d(windowSize: Int, padding: Int? = nil, stride: Int? = nil) -> Tensor<Element, Device> {
        let padding = padding ?? ((windowSize - 1) / 2)
        let stride = stride ?? windowSize
        let result = Device.FusedOperations.averagePooling2d(input: self, windowSize: windowSize, padding: padding, stride: stride)

        return result.attachingContext(tag: "averagePool2d", sources: [self]) { resultGradient in
            if resultGradient.requiresGradient {
                [Composed.averagePooling2dGradient(inputShape: self.shape, outputGradient: resultGradient, windowSize: windowSize, padding: padding, stride: stride)]
            } else {
                [Device.FusedOperations.averagePooling2dBackward(input: self, outputGradient: resultGradient, windowSize: windowSize, padding: padding, stride: stride)]
            }
        }
    }
}

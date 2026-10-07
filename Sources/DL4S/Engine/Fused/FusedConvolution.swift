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

// MARK: Default implementations

// Convolutions are matrix products of the filters with the windows of the images, which img2col extracts.
// A transposed convolution is the input gradient of a convolution. Pooling reduces the windows of every channel.

public extension FusedOperationsType {
    static func convolution2d<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let (outputChannels, windowSize) = (filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3])
        let (batchSize, outputHeight, outputWidth) = (result.shape[0], result.shape[2], result.shape[3])
        let columns = math.temporary([windowSize, batchSize * outputHeight * outputWidth])
        let product = math.temporary([outputChannels, batchSize, outputHeight, outputWidth])
        Device.Engine.img2col(values: input, result: columns, kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
        math.multiplyMatrices(filters.reshaped(to: [outputChannels, windowSize]), columns, into: product.reshaped(to: [outputChannels, product.count / outputChannels]))
        math.permute(product, to: [1, 0, 2, 3], into: result)
        if let bias {
            math.add(result, bias.reshaped(to: [1, outputChannels, 1, 1]), into: result)
        }
    }

    static func convolution2dBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        filters: ShapedBuffer<N, Device>,
        bias: ShapedBuffer<N, Device>?,
        outputGradient: ShapedBuffer<N, Device>,
        padding: Int,
        stride: Int,
        inputGradient: GradientBuffer<N, Device>?,
        filterGradient: GradientBuffer<N, Device>?,
        biasGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.write(biasGradient) { db in
            math.sum(outputGradient, along: [0, 2, 3], into: db)
        }
        // The other gradients need the window matrix, which is the largest intermediate buffer.
        guard inputGradient != nil || filterGradient != nil else {
            return
        }
        let (outputChannels, windowSize) = (filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3])
        let windows = outputGradient.count / outputChannels
        let (kernelHeight, kernelWidth) = (filters.shape[2], filters.shape[3])
        // The layout of the matrix product of the forward pass: [outputChannels, batchSize * outputHeight * outputWidth]
        let gradientMatrix = math.temporary([outputChannels, outputGradient.shape[0], outputGradient.shape[2], outputGradient.shape[3]])
        math.permute(outputGradient, to: [1, 0, 2, 3], into: gradientMatrix)
        let columns = math.temporary([windowSize, windows])
        math.write(inputGradient) { dx in
            math.multiplyMatrices(filters.reshaped(to: [outputChannels, windowSize]), gradientMatrix.reshaped(to: [outputChannels, windows]), lhsTransposed: true, into: columns)
            Device.Engine.col2img(matrix: ShapedBuffer(columns), image: dx, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
        }
        if let filterGradient {
            // The windows are extracted again instead of being kept alive between the forward and the backward pass.
            Device.Engine.img2col(values: input, result: columns, kernelWidth: kernelWidth, kernelHeight: kernelHeight, padding: padding, stride: stride)
            math.multiplyMatrices(
                gradientMatrix.reshaped(to: [outputChannels, windows]),
                columns,
                rhsTransposed: true,
                into: filterGradient.values.reshaped(to: [outputChannels, windowSize]),
                beta: filterGradient.beta,
            )
        }
    }

    static func transposedConvolution2d<N: NumericType>(input: ShapedBuffer<N, Device>, filters: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, inset: Int, stride: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let (inputChannels, kernelSize) = (input.shape[1], filters.shape[0] * filters.shape[2] * filters.shape[3])
        let pixels = input.count / inputChannels
        // The filters are read as a [inputChannels, outputChannels * kernelHeight * kernelWidth] matrix.
        let inputMatrix = math.temporary([inputChannels, input.shape[0], input.shape[2], input.shape[3]])
        let columns = math.temporary([kernelSize, pixels])
        math.permute(input, to: [1, 0, 2, 3], into: inputMatrix)
        math.multiplyMatrices(filters.reshaped(to: [inputChannels, kernelSize]), inputMatrix.reshaped(to: [inputChannels, pixels]), lhsTransposed: true, into: columns)
        Device.Engine.col2img(matrix: ShapedBuffer(columns), image: result, kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: inset, stride: stride)
        if let bias {
            math.add(result, bias.reshaped(to: [1, filters.shape[0], 1, 1]), into: result)
        }
    }

    static func transposedConvolution2dBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        filters: ShapedBuffer<N, Device>,
        bias: ShapedBuffer<N, Device>?,
        outputGradient: ShapedBuffer<N, Device>,
        inset: Int,
        stride: Int,
        inputGradient: GradientBuffer<N, Device>?,
        filterGradient: GradientBuffer<N, Device>?,
        biasGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        math.write(biasGradient) { db in
            math.sum(outputGradient, along: [0, 2, 3], into: db)
        }
        // The other gradients need the window matrix, which is the largest intermediate buffer.
        guard inputGradient != nil || filterGradient != nil else {
            return
        }
        let (inputChannels, kernelSize) = (input.shape[1], filters.shape[0] * filters.shape[2] * filters.shape[3])
        let pixels = input.count / inputChannels
        let matrixShape = [inputChannels, input.shape[0], input.shape[2], input.shape[3]]
        // img2col is the adjoint of the col2img of the forward pass.
        let columnGradient = math.temporary([kernelSize, pixels])
        Device.Engine.img2col(values: outputGradient, result: columnGradient, kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: inset, stride: stride)
        let matrix = math.temporary(matrixShape)
        math.write(inputGradient) { dx in
            math.multiplyMatrices(filters.reshaped(to: [inputChannels, kernelSize]), columnGradient, into: matrix.reshaped(to: [inputChannels, pixels]))
            math.permute(matrix, to: [1, 0, 2, 3], into: dx)
        }
        if let filterGradient {
            math.permute(input, to: [1, 0, 2, 3], into: matrix)
            math.multiplyMatrices(
                matrix.reshaped(to: [inputChannels, pixels]),
                columnGradient,
                rhsTransposed: true,
                into: filterGradient.values.reshaped(to: [inputChannels, kernelSize]),
                beta: filterGradient.beta,
            )
        }
    }

    static func maxPooling2d<N: NumericType>(input: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let columns = poolingColumns(of: input, windowSize: windowSize, padding: padding, stride: stride, math: math)
        math.maximum(columns, along: 0, into: result)
    }

    static func maxPooling2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // The gradient of a window goes to the position of its largest value.
        let columns = poolingColumns(of: input, windowSize: windowSize, padding: padding, stride: stride, math: math)
        let windows = outputGradient.count
        let positions = math.positions([windows])
        math.maximum(columns, along: 0, into: math.temporary([windows]), positions: positions)
        math.write(inputGradient) { dx in
            Device.Engine.scatter(reduced: outputGradient.reshaped(to: [windows]), context: ShapedBuffer(positions), result: columns, axis: 0, ignoreIndex: -1)
            Device.Engine.col2img(matrix: ShapedBuffer(columns), image: dx.reshaped(to: poolingImageShape(of: input.shape)), kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride)
        }
    }

    static func averagePooling2d<N: NumericType>(input: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        // Padding elements are zeros that count for the mean.
        let columns = poolingColumns(of: input, windowSize: windowSize, padding: padding, stride: stride, math: math)
        math.sum(columns, along: [0], into: result)
        math.multiply(result, 1 / N(windowSize * windowSize), into: result)
    }

    static func averagePooling2dBackward<N: NumericType>(input: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let windowElements = windowSize * windowSize
        let windows = outputGradient.count
        math.write(inputGradient) { dx in
            // Every element of a window gets the gradient of the window, divided by the number of elements.
            let columns = math.temporary([windowElements, windows])
            math.multiply(math.constant(1 / N(windowElements), shape: [windowElements, 1]), outputGradient.reshaped(to: [1, windows]), into: columns)
            Device.Engine.col2img(matrix: ShapedBuffer(columns), image: dx.reshaped(to: poolingImageShape(of: input.shape)), kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride)
        }
    }
}

extension FusedOperationsType {
    /// The images of pooling with one channel each, [batchSize \* channels, 1, height, width].
    static func poolingImageShape(of shape: [Int]) -> [Int] {
        [shape[0] * shape[1], 1, shape[2], shape[3]]
    }

    /// Extracts the pooling windows of every channel, shape [windowSize \* windowSize, batchSize \* channels \* outputHeight \* outputWidth].
    static func poolingColumns<N: NumericType>(of input: ShapedBuffer<N, Device>, windowSize: Int, padding: Int, stride: Int, math: BufferMath<N, Device>) -> MutableShapedBuffer<N, Device> {
        let windows = ConvUtil.outputSize(inputSize: input.shape[2], kernelSize: windowSize, padding: padding, stride: stride)
            * ConvUtil.outputSize(inputSize: input.shape[3], kernelSize: windowSize, padding: padding, stride: stride)
        let columns = math.temporary([windowSize * windowSize, input.shape[0] * input.shape[1] * windows])
        Device.Engine.img2col(values: input.reshaped(to: poolingImageShape(of: input.shape)), result: columns, kernelWidth: windowSize, kernelHeight: windowSize, padding: padding, stride: stride)
        return columns
    }
}

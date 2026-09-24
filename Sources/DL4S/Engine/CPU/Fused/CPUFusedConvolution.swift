//
//  CPUFusedConvolution.swift
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

// The convolutions process the batch in chunks of images. A chunk extracts its windows with img2col into one
// scratch matrix of at most `CPUKernels.maximumColumnElements` elements, which is reused for every chunk, and
// multiplies it with the filters directly into the layout of the result. Pooling works on the images directly.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func convolution2d<N: NumericType>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, padding: Int, stride: Int) -> Tensor<N, CPU> {
        guard let geometry = ConvolutionGeometry(input: input, filters: filters, bias: bias, padding: padding, stride: stride) else {
            return DefaultFusedOperations<CPU>.convolution2d(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        }
        let (batchSize, outputChannels, windows, windowSize) = (geometry.batchSize, geometry.outputChannels, geometry.windows, geometry.windowSize)
        let (result, y) = CPUKernels.makeTensor(shape: [batchSize, outputChannels, geometry.outputHeight, geometry.outputWidth]) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (x, w, b) = (input.elementPointer, filters.elementPointer, bias?.elementPointer)
        let chunk = geometry.chunkSize(rowsPerImage: windowSize)

        CPUKernels.withScratch(N.self, count: windowSize * chunk * windows + (chunk > 1 ? outputChannels * chunk * windows : 0)) { scratch in
            let columns = scratch
            let products = scratch + windowSize * chunk * windows
            for first in Swift.stride(from: 0, to: batchSize, by: chunk) {
                let count = Swift.min(chunk, batchSize - first)
                geometry.extractWindows(from: x, firstImage: first, imageCount: count, into: columns)
                if count == 1 {
                    // A single image is multiplied directly into its slice of the result, which starts with the bias.
                    let output = y + first * outputChannels * windows
                    if let b {
                        for channel in 0 ..< outputChannels {
                            CPUKernels.fill(output + channel * windows, with: b[channel], count: windows)
                        }
                    }
                    CPUKernels.gemm(w, shape: (outputChannels, windowSize), columns, shape: (windowSize, windows), into: output, beta: b == nil ? 0 : 1)
                    continue
                }
                CPUKernels.gemm(w, shape: (outputChannels, windowSize), columns, shape: (windowSize, count * windows), into: products)
                // The product has the layout [outputChannels, images, windows]. The result has the layout [images, outputChannels, windows].
                for image in 0 ..< count {
                    for channel in 0 ..< outputChannels {
                        N.vsAdd(
                            lhs: UnsafeBufferPointer(start: products + (channel * count + image) * windows, count: windows),
                            rhs: b?[channel] ?? 0,
                            result: UnsafeMutableBufferPointer(start: y + ((first + image) * outputChannels + channel) * windows, count: windows),
                            count: windows,
                        )
                    }
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func convolution2dBackward<N: NumericType>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, outputGradient: Tensor<N, CPU>, padding: Int, stride: Int, accumulating gradients: inout (input: Tensor<N, CPU>?, filters: Tensor<N, CPU>?, bias: Tensor<N, CPU>?)) {
        guard let geometry = ConvolutionGeometry(input: input, filters: filters, bias: bias, padding: padding, stride: stride),
              outputGradient.shape == [geometry.batchSize, geometry.outputChannels, geometry.outputHeight, geometry.outputWidth]
        else {
            DefaultFusedOperations<CPU>.convolution2dBackward(input: input, filters: filters, bias: bias, outputGradient: outputGradient, padding: padding, stride: stride, accumulating: &gradients)
            return
        }
        let (batchSize, outputChannels, windows, windowSize) = (geometry.batchSize, geometry.outputChannels, geometry.windows, geometry.windowSize)
        let (x, w, g) = (input.elementPointer, filters.elementPointer, outputGradient.elementPointer)
        let inputGradient = input.requiresGradient ? CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>) : nil
        // The filter and bias gradients are added to the accumulated gradients directly.
        let filterGradient = filters.requiresGradient ? GradientTarget(taking: &gradients.filters, shape: filters.shape) : nil
        let biasGradient = (bias?.requiresGradient ?? false) ? GradientTarget(taking: &gradients.bias, shape: [outputChannels], zeroed: true) : nil
        let chunk = geometry.chunkSize(rowsPerImage: windowSize)

        CPUKernels.withScratch(N.self, count: windowSize * chunk * windows + (chunk > 1 ? outputChannels * chunk * windows : 0)) { scratch in
            let columns = scratch
            let gradientMatrix = scratch + windowSize * chunk * windows
            for first in Swift.stride(from: 0, to: batchSize, by: chunk) {
                let count = Swift.min(chunk, batchSize - first)
                // The gradient in the layout [outputChannels, images, windows] of the matrix product.
                let gradient: UnsafePointer<N>
                if count == 1 {
                    gradient = g + first * outputChannels * windows
                } else {
                    for image in 0 ..< count {
                        for channel in 0 ..< outputChannels {
                            (gradientMatrix + (channel * count + image) * windows)
                                .update(from: g + ((first + image) * outputChannels + channel) * windows, count: windows)
                        }
                    }
                    gradient = UnsafePointer(gradientMatrix)
                }
                if let filterGradient {
                    // The windows are extracted again instead of being kept alive between the forward and the backward pass.
                    geometry.extractWindows(from: x, firstImage: first, imageCount: count, into: columns)
                    CPUKernels.gemm(gradient, shape: (outputChannels, count * windows), columns, shape: (windowSize, count * windows), rhsTransposed: true, into: filterGradient.pointer, beta: first == 0 ? filterGradient.beta : 1)
                }
                if let (_, dx) = inputGradient {
                    CPUKernels.gemm(w, shape: (outputChannels, windowSize), lhsTransposed: true, gradient, shape: (outputChannels, count * windows), into: columns)
                    geometry.accumulateWindows(columns, firstImage: first, imageCount: count, into: dx)
                }
                if let db = biasGradient?.pointer {
                    for image in 0 ..< count {
                        for channel in 0 ..< outputChannels {
                            db[channel] += CPUKernels.sum(g + ((first + image) * outputChannels + channel) * windows, count: windows)
                        }
                    }
                }
            }
        }
        Tensor.accumulate(inputGradient?.0, into: &gradients.input)
        filterGradient?.finish(into: &gradients.filters)
        biasGradient?.finish(into: &gradients.bias)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func transposedConvolution2d<N: NumericType>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, inset: Int, stride: Int) -> Tensor<N, CPU> {
        guard let geometry = TransposedConvolutionGeometry(input: input, filters: filters, bias: bias, inset: inset, stride: stride) else {
            return DefaultFusedOperations<CPU>.transposedConvolution2d(input: input, filters: filters, bias: bias, inset: inset, stride: stride)
        }
        let (batchSize, inputChannels, outputChannels, pixels, kernelSize) = (geometry.batchSize, geometry.inputChannels, geometry.outputChannels, geometry.pixels, geometry.kernelSize)
        let outputPixels = geometry.outputHeight * geometry.outputWidth
        let (result, y) = CPUKernels.makeTensor(shape: [batchSize, outputChannels, geometry.outputHeight, geometry.outputWidth]) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (x, a, b) = (input.elementPointer, filters.elementPointer, bias?.elementPointer)
        let chunk = geometry.chunkSize

        CPUKernels.withScratch(N.self, count: kernelSize * chunk * pixels + (chunk > 1 ? inputChannels * chunk * pixels : 0)) { scratch in
            let columns = scratch
            let inputMatrix = scratch + kernelSize * chunk * pixels
            for first in Swift.stride(from: 0, to: batchSize, by: chunk) {
                let count = Swift.min(chunk, batchSize - first)
                let images = geometry.inputMatrix(x, firstImage: first, imageCount: count, scratch: inputMatrix)
                // The filters, read as a [inputChannels, outputChannels * kernelHeight * kernelWidth] matrix, transposed.
                CPUKernels.gemm(a, shape: (inputChannels, kernelSize), lhsTransposed: true, images, shape: (inputChannels, count * pixels), into: columns)
                N.col2img(
                    values: UnsafeBufferPointer(start: columns, count: kernelSize * count * pixels),
                    result: UnsafeMutableBufferPointer(start: y + first * outputChannels * outputPixels, count: count * outputChannels * outputPixels),
                    batchSize: count,
                    channels: outputChannels,
                    height: geometry.outputHeight,
                    width: geometry.outputWidth,
                    kernelHeight: geometry.kernelHeight,
                    kernelWidth: geometry.kernelWidth,
                    padding: inset,
                    stride: stride,
                )
                if let b {
                    for image in first ..< first + count {
                        for channel in 0 ..< outputChannels {
                            let row = y + (image * outputChannels + channel) * outputPixels
                            let value = b[channel]
                            for j in 0 ..< outputPixels {
                                row[j] += value
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func transposedConvolution2dBackward<N: NumericType>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, outputGradient: Tensor<N, CPU>, inset: Int, stride: Int, accumulating gradients: inout (input: Tensor<N, CPU>?, filters: Tensor<N, CPU>?, bias: Tensor<N, CPU>?)) {
        guard let geometry = TransposedConvolutionGeometry(input: input, filters: filters, bias: bias, inset: inset, stride: stride),
              outputGradient.shape == [geometry.batchSize, geometry.outputChannels, geometry.outputHeight, geometry.outputWidth]
        else {
            DefaultFusedOperations<CPU>.transposedConvolution2dBackward(input: input, filters: filters, bias: bias, outputGradient: outputGradient, inset: inset, stride: stride, accumulating: &gradients)
            return
        }
        let (batchSize, inputChannels, outputChannels, pixels, kernelSize) = (geometry.batchSize, geometry.inputChannels, geometry.outputChannels, geometry.pixels, geometry.kernelSize)
        let outputPixels = geometry.outputHeight * geometry.outputWidth
        let (x, a, g) = (input.elementPointer, filters.elementPointer, outputGradient.elementPointer)
        // The gradients are added to the accumulated gradients directly.
        let inputGradient = input.requiresGradient ? GradientTarget(taking: &gradients.input, shape: input.shape) : nil
        let filterGradient = filters.requiresGradient ? GradientTarget(taking: &gradients.filters, shape: filters.shape) : nil
        let biasGradient = (bias?.requiresGradient ?? false) ? GradientTarget(taking: &gradients.bias, shape: [outputChannels], zeroed: true) : nil
        let chunk = geometry.chunkSize

        CPUKernels.withScratch(N.self, count: kernelSize * chunk * pixels + (chunk > 1 ? inputChannels * chunk * pixels : 0)) { scratch in
            let columns = scratch
            let inputMatrix = scratch + kernelSize * chunk * pixels
            for first in Swift.stride(from: 0, to: batchSize, by: chunk) {
                let count = Swift.min(chunk, batchSize - first)
                // img2col is the adjoint of the col2img of the forward pass.
                N.img2col(
                    values: UnsafeBufferPointer(start: g + first * outputChannels * outputPixels, count: count * outputChannels * outputPixels),
                    result: UnsafeMutableBufferPointer(start: columns, count: kernelSize * count * pixels),
                    batchSize: count,
                    channels: outputChannels,
                    height: geometry.outputHeight,
                    width: geometry.outputWidth,
                    kernelHeight: geometry.kernelHeight,
                    kernelWidth: geometry.kernelWidth,
                    padding: inset,
                    stride: stride,
                )
                if let filterGradient {
                    let images = geometry.inputMatrix(x, firstImage: first, imageCount: count, scratch: inputMatrix)
                    CPUKernels.gemm(images, shape: (inputChannels, count * pixels), columns, shape: (kernelSize, count * pixels), rhsTransposed: true, into: filterGradient.pointer, beta: first == 0 ? filterGradient.beta : 1)
                }
                if let inputGradient {
                    let dx = inputGradient.pointer
                    if count == 1 {
                        CPUKernels.gemm(a, shape: (inputChannels, kernelSize), columns, shape: (kernelSize, pixels), into: dx + first * inputChannels * pixels, beta: inputGradient.beta)
                    } else {
                        CPUKernels.gemm(a, shape: (inputChannels, kernelSize), columns, shape: (kernelSize, count * pixels), into: inputMatrix)
                        // The product has the layout [inputChannels, images, pixels]. The gradient has the layout [images, inputChannels, pixels].
                        for image in 0 ..< count {
                            for channel in 0 ..< inputChannels {
                                CPUKernels.store(
                                    inputMatrix + (channel * count + image) * pixels,
                                    into: dx + ((first + image) * inputChannels + channel) * pixels,
                                    beta: inputGradient.beta,
                                    count: pixels,
                                )
                            }
                        }
                    }
                }
                if let db = biasGradient?.pointer {
                    for image in first ..< first + count {
                        for channel in 0 ..< outputChannels {
                            db[channel] += CPUKernels.sum(g + (image * outputChannels + channel) * outputPixels, count: outputPixels)
                        }
                    }
                }
            }
        }
        inputGradient?.finish(into: &gradients.input)
        filterGradient?.finish(into: &gradients.filters)
        biasGradient?.finish(into: &gradients.bias)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func maxPooling2d<N: NumericType>(input: Tensor<N, CPU>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, CPU> {
        guard let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride) else {
            return DefaultFusedOperations<CPU>.maxPooling2d(input: input, windowSize: windowSize, padding: padding, stride: stride)
        }
        let (result, y) = CPUKernels.makeTensor(shape: geometry.outputShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let (outputHeight, outputWidth) = (geometry.outputShape[2], geometry.outputShape[3])
        for plane in 0 ..< geometry.planes {
            let image = x + plane * geometry.pixels
            let output = y + plane * outputHeight * outputWidth
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let (firstRow, firstColumn) = (row &* stride &- padding, column &* stride &- padding)
                    output[row &* outputWidth &+ column] = geometry.isInside(firstRow: firstRow, firstColumn: firstColumn)
                        ? geometry.interiorMaximum(in: image, firstRow: firstRow, firstColumn: firstColumn).value
                        : geometry.maximum(in: image, firstRow: firstRow, firstColumn: firstColumn).value
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func maxPooling2dBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, windowSize: Int, padding: Int, stride: Int, accumulating gradient: inout Tensor<N, CPU>?) {
        guard let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride), outputGradient.shape == geometry.outputShape else {
            DefaultFusedOperations<CPU>.maxPooling2dBackward(input: input, outputGradient: outputGradient, windowSize: windowSize, padding: padding, stride: stride, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeZeroTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let (outputHeight, outputWidth) = (geometry.outputShape[2], geometry.outputShape[3])
        for plane in 0 ..< geometry.planes {
            let (image, imageGradient) = (x + plane * geometry.pixels, dx + plane * geometry.pixels)
            let gradient = g + plane * outputHeight * outputWidth
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let (firstRow, firstColumn) = (row &* stride &- padding, column &* stride &- padding)
                    // The maximum is found again instead of being kept alive between the forward and the backward pass.
                    // The gradient of a padding element is dropped.
                    let position = geometry.isInside(firstRow: firstRow, firstColumn: firstColumn)
                        ? geometry.interiorMaximum(in: image, firstRow: firstRow, firstColumn: firstColumn).position
                        : geometry.maximum(in: image, firstRow: firstRow, firstColumn: firstColumn).position
                    if let position {
                        imageGradient[position] += gradient[row &* outputWidth &+ column]
                    }
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func averagePooling2d<N: NumericType>(input: Tensor<N, CPU>, windowSize: Int, padding: Int, stride: Int) -> Tensor<N, CPU> {
        guard let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride) else {
            return DefaultFusedOperations<CPU>.averagePooling2d(input: input, windowSize: windowSize, padding: padding, stride: stride)
        }
        let (result, y) = CPUKernels.makeTensor(shape: geometry.outputShape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let (outputHeight, outputWidth) = (geometry.outputShape[2], geometry.outputShape[3])
        // Padding elements are zeros that count for the mean.
        let inverseWindowElements = 1 / N(windowSize * windowSize)
        for plane in 0 ..< geometry.planes {
            let image = x + plane * geometry.pixels
            let output = y + plane * outputHeight * outputWidth
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let (firstRow, firstColumn) = (row &* stride &- padding, column &* stride &- padding)
                    let (rows, columns) = (geometry.clampedRows(firstRow: firstRow), geometry.clampedColumns(firstColumn: firstColumn))
                    var sum: N = 0
                    for imageRow in rows {
                        let line = image + imageRow &* geometry.width
                        for imageColumn in columns {
                            sum += line[imageColumn]
                        }
                    }
                    output[row &* outputWidth &+ column] = sum * inverseWindowElements
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func averagePooling2dBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, windowSize: Int, padding: Int, stride: Int, accumulating gradient: inout Tensor<N, CPU>?) {
        guard let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride), outputGradient.shape == geometry.outputShape else {
            DefaultFusedOperations<CPU>.averagePooling2dBackward(input: input, outputGradient: outputGradient, windowSize: windowSize, padding: padding, stride: stride, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeZeroTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let g = outputGradient.elementPointer
        let (outputHeight, outputWidth) = (geometry.outputShape[2], geometry.outputShape[3])
        let inverseWindowElements = 1 / N(windowSize * windowSize)
        for plane in 0 ..< geometry.planes {
            let imageGradient = dx + plane * geometry.pixels
            let gradient = g + plane * outputHeight * outputWidth
            for row in 0 ..< outputHeight {
                for column in 0 ..< outputWidth {
                    let (firstRow, firstColumn) = (row &* stride &- padding, column &* stride &- padding)
                    let value = gradient[row &* outputWidth &+ column] * inverseWindowElements
                    for imageRow in geometry.clampedRows(firstRow: firstRow) {
                        let line = imageGradient + imageRow &* geometry.width
                        for imageColumn in geometry.clampedColumns(firstColumn: firstColumn) {
                            line[imageColumn] += value
                        }
                    }
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }
}

/// Shapes of a 2D convolution.
struct ConvolutionGeometry {
    let batchSize: Int
    let inputChannels: Int
    let height: Int
    let width: Int
    let outputChannels: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let outputHeight: Int
    let outputWidth: Int
    let padding: Int
    let stride: Int

    /// Number of windows per image.
    var windows: Int {
        outputHeight * outputWidth
    }

    /// Number of elements per window.
    var windowSize: Int {
        inputChannels * kernelHeight * kernelWidth
    }

    /// Returns nil for shapes that the kernels do not support.
    init?<N>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, padding: Int, stride: Int) {
        guard input.dim == 4, filters.dim == 4, input.count > 0, filters.count > 0, input.shape[1] == filters.shape[1], stride > 0, padding >= 0,
              bias.map({ $0.count == filters.shape[0] }) ?? true
        else {
            return nil
        }
        batchSize = input.shape[0]
        inputChannels = input.shape[1]
        height = input.shape[2]
        width = input.shape[3]
        outputChannels = filters.shape[0]
        kernelHeight = filters.shape[2]
        kernelWidth = filters.shape[3]
        outputHeight = (height + 2 * padding - kernelHeight) / stride + 1
        outputWidth = (width + 2 * padding - kernelWidth) / stride + 1
        self.padding = padding
        self.stride = stride
        guard height + 2 * padding >= kernelHeight, width + 2 * padding >= kernelWidth else {
            return nil
        }
    }

    /// Number of images per chunk, so that the window matrix of a chunk stays below the upper bound.
    func chunkSize(rowsPerImage: Int) -> Int {
        Swift.max(1, Swift.min(batchSize, CPUKernels.maximumColumnElements / Swift.max(rowsPerImage * windows, 1)))
    }

    /// Writes the windows of consecutive images as a [windowSize, imageCount \* windows] matrix.
    func extractWindows<N: NumericType>(from x: UnsafePointer<N>, firstImage: Int, imageCount: Int, into columns: UnsafeMutablePointer<N>) {
        let imageElements = inputChannels * height * width
        N.img2col(
            values: UnsafeBufferPointer(start: x + firstImage * imageElements, count: imageCount * imageElements),
            result: UnsafeMutableBufferPointer(start: columns, count: windowSize * imageCount * windows),
            batchSize: imageCount,
            channels: inputChannels,
            height: height,
            width: width,
            kernelHeight: kernelHeight,
            kernelWidth: kernelWidth,
            padding: padding,
            stride: stride,
        )
    }

    /// Adds the windows of a [windowSize, imageCount \* windows] matrix to the images, after the images were set to 0.
    func accumulateWindows<N: NumericType>(_ columns: UnsafePointer<N>, firstImage: Int, imageCount: Int, into dx: UnsafeMutablePointer<N>) {
        let imageElements = inputChannels * height * width
        N.col2img(
            values: UnsafeBufferPointer(start: columns, count: windowSize * imageCount * windows),
            result: UnsafeMutableBufferPointer(start: dx + firstImage * imageElements, count: imageCount * imageElements),
            batchSize: imageCount,
            channels: inputChannels,
            height: height,
            width: width,
            kernelHeight: kernelHeight,
            kernelWidth: kernelWidth,
            padding: padding,
            stride: stride,
        )
    }
}

/// Shapes of a transposed 2D convolution.
struct TransposedConvolutionGeometry {
    let batchSize: Int
    let inputChannels: Int
    let pixels: Int
    let outputChannels: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let outputHeight: Int
    let outputWidth: Int

    /// Number of rows of the window matrix: outputChannels \* kernelHeight \* kernelWidth.
    var kernelSize: Int {
        outputChannels * kernelHeight * kernelWidth
    }

    /// Number of images per chunk, so that the window matrix of a chunk stays below the upper bound.
    var chunkSize: Int {
        Swift.max(1, Swift.min(batchSize, CPUKernels.maximumColumnElements / Swift.max(kernelSize * pixels, 1)))
    }

    /// Returns nil for shapes that the kernels do not support.
    init?<N>(input: Tensor<N, CPU>, filters: Tensor<N, CPU>, bias: Tensor<N, CPU>?, inset: Int, stride: Int) {
        guard input.dim == 4, filters.dim == 4, input.count > 0, filters.count > 0, input.shape[1] == filters.shape[1], stride > 0, inset >= 0,
              bias.map({ $0.count == filters.shape[0] }) ?? true
        else {
            return nil
        }
        batchSize = input.shape[0]
        inputChannels = input.shape[1]
        pixels = input.shape[2] * input.shape[3]
        outputChannels = filters.shape[0]
        kernelHeight = filters.shape[2]
        kernelWidth = filters.shape[3]
        outputHeight = (input.shape[2] - 1) * stride - 2 * inset + kernelHeight
        outputWidth = (input.shape[3] - 1) * stride - 2 * inset + kernelWidth
        guard outputHeight > 0, outputWidth > 0 else {
            return nil
        }
    }

    /// Returns consecutive images as an [inputChannels, imageCount \* pixels] matrix. It uses the scratch buffer for more than one image.
    func inputMatrix<N>(_ x: UnsafePointer<N>, firstImage: Int, imageCount: Int, scratch: UnsafeMutablePointer<N>) -> UnsafePointer<N> {
        if imageCount == 1 {
            return x + firstImage * inputChannels * pixels
        }
        for image in 0 ..< imageCount {
            for channel in 0 ..< inputChannels {
                (scratch + (channel * imageCount + image) * pixels)
                    .update(from: x + ((firstImage + image) * inputChannels + channel) * pixels, count: pixels)
            }
        }
        return UnsafePointer(scratch)
    }
}

/// Shapes of 2D pooling.
struct PoolingGeometry {
    let planes: Int
    let height: Int
    let width: Int
    let windowSize: Int
    let padding: Int
    let stride: Int
    let outputShape: [Int]

    var pixels: Int {
        height * width
    }

    /// Returns nil for shapes that the kernels do not support.
    init?<N>(input: Tensor<N, CPU>, windowSize: Int, padding: Int, stride: Int) {
        guard input.dim == 4, input.count > 0, windowSize > 0, stride > 0, padding >= 0,
              input.shape[2] + 2 * padding >= windowSize, input.shape[3] + 2 * padding >= windowSize
        else {
            return nil
        }
        planes = input.shape[0] * input.shape[1]
        height = input.shape[2]
        width = input.shape[3]
        self.windowSize = windowSize
        self.padding = padding
        self.stride = stride
        outputShape = [
            input.shape[0],
            input.shape[1],
            (height + 2 * padding - windowSize) / stride + 1,
            (width + 2 * padding - windowSize) / stride + 1,
        ]
    }

    /// Whether a window with the given top left position lies completely inside the image.
    @inline(__always)
    func isInside(firstRow: Int, firstColumn: Int) -> Bool {
        firstRow >= 0 && firstColumn >= 0 && firstRow &+ windowSize <= height && firstColumn &+ windowSize <= width
    }

    /// Rows of a window that are not in the padding.
    @inline(__always)
    func clampedRows(firstRow: Int) -> Range<Int> {
        Swift.max(firstRow, 0) ..< Swift.min(firstRow &+ windowSize, height)
    }

    /// Columns of a window that are not in the padding.
    @inline(__always)
    func clampedColumns(firstColumn: Int) -> Range<Int> {
        Swift.max(firstColumn, 0) ..< Swift.min(firstColumn &+ windowSize, width)
    }

    /// Returns the largest value of a window that lies completely inside the image, and its index in the plane.
    ///
    /// The elements are compared in row-major order and the first largest value wins, as in the default implementation.
    @inline(__always)
    func interiorMaximum<N: NumericType>(in image: UnsafePointer<N>, firstRow: Int, firstColumn: Int) -> (value: N, position: Int?) {
        var position = firstRow &* width &+ firstColumn
        var best = image[position]
        for row in firstRow ..< firstRow &+ windowSize {
            let lineStart = row &* width
            for column in firstColumn ..< firstColumn &+ windowSize {
                let value = image[lineStart &+ column]
                let isLarger = value > best
                best = isLarger ? value : best
                position = isLarger ? lineStart &+ column : position
            }
        }
        return (best, position)
    }

    /// Returns the largest value of a window and its index in the plane, or nil as the index when the largest value is a padding zero.
    ///
    /// The elements are compared in row-major order and the first largest value wins, as in the default implementation.
    @inline(__always)
    func maximum<N: NumericType>(in image: UnsafePointer<N>, firstRow: Int, firstColumn: Int) -> (value: N, position: Int?) {
        var best: N = 0
        var position: Int?
        var found = false
        for row in firstRow ..< firstRow + windowSize {
            for column in firstColumn ..< firstColumn + windowSize {
                let isInside = row >= 0 && row < height && column >= 0 && column < width
                let value = isInside ? image[row * width + column] : 0
                if !found || value > best {
                    best = value
                    position = isInside ? row * width + column : nil
                    found = true
                }
            }
        }
        return (best, position)
    }
}

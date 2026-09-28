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
// multiplies it with the filters directly into the layout of the result. A transposed convolution is the adjoint
// of a convolution, so it uses the same operations with the roles of the images and the windows exchanged.
// Pooling works on the images directly.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func convolution2d<N: NumericType>(input: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, padding: Int, stride: Int, result: MutableShapedBuffer<N, CPU>) {
        let geometry = ConvolutionGeometry(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        geometry.multiplyWindows(of: input.elementPointer, filters: filters.elementPointer, bias: bias?.elementPointer, output: (result.elementPointer, 0), filterGradient: nil)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func convolution2dBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        filters: ShapedBuffer<N, CPU>,
        bias: ShapedBuffer<N, CPU>?,
        outputGradient: ShapedBuffer<N, CPU>,
        padding: Int,
        stride: Int,
        inputGradient: GradientBuffer<N, CPU>?,
        filterGradient: GradientBuffer<N, CPU>?,
        biasGradient: GradientBuffer<N, CPU>?,
    ) {
        let geometry = ConvolutionGeometry(input: input, filters: filters, bias: bias, padding: padding, stride: stride)
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        let (x, w, g) = (input.elementPointer, filters.elementPointer, outputGradient.elementPointer)
        geometry.multiplyOutputGradient(
            g,
            filters: w,
            imageGradient: inputGradient?.elementsToWrite(),
            filterGradient: filterGradient.map { (x, $0.elementsToAddTo()) },
        )
        if let biasGradient {
            CPUKernels.addChannelSums(g, images: geometry.batchSize, channels: geometry.outputChannels, pixels: geometry.windows, to: biasGradient.elementsToAddTo())
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func transposedConvolution2d<N: NumericType>(input: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, inset: Int, stride: Int, result: MutableShapedBuffer<N, CPU>) {
        let geometry = ConvolutionGeometry(transposedInput: input, filters: filters, bias: bias, inset: inset, stride: stride)
        let y = result.elementPointer
        geometry.multiplyOutputGradient(input.elementPointer, filters: filters.elementPointer, imageGradient: (y, 0), filterGradient: nil)
        if let bias {
            CPUKernels.addChannelOffsets(bias.elementPointer, to: y, images: geometry.batchSize, channels: geometry.imageChannels, pixels: geometry.imagePixels)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func transposedConvolution2dBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        filters: ShapedBuffer<N, CPU>,
        bias: ShapedBuffer<N, CPU>?,
        outputGradient: ShapedBuffer<N, CPU>,
        inset: Int,
        stride: Int,
        inputGradient: GradientBuffer<N, CPU>?,
        filterGradient: GradientBuffer<N, CPU>?,
        biasGradient: GradientBuffer<N, CPU>?,
    ) {
        let geometry = ConvolutionGeometry(transposedInput: input, filters: filters, bias: bias, inset: inset, stride: stride)
        precondition(outputGradient.shape == geometry.imageShape, "The gradient of the result must have the shape of the result.")
        let (x, w, g) = (input.elementPointer, filters.elementPointer, outputGradient.elementPointer)
        // The adjoint convolution reads the gradient of the result as its images.
        geometry.multiplyWindows(
            of: g,
            filters: w,
            bias: nil,
            output: inputGradient?.elementsToWrite(),
            filterGradient: filterGradient.map { (x, $0.elementsToAddTo()) },
        )
        if let biasGradient {
            CPUKernels.addChannelSums(g, images: geometry.batchSize, channels: geometry.imageChannels, pixels: geometry.imagePixels, to: biasGradient.elementsToAddTo())
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func maxPooling2d<N: NumericType>(input: ShapedBuffer<N, CPU>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, CPU>) {
        let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride)
        let (x, y) = (input.elementPointer, result.elementPointer)
        let columnMaxima = UnsafeMutablePointer<N>.allocate(capacity: geometry.width)
        defer {
            columnMaxima.deallocate()
        }
        for plane in 0 ..< geometry.planes {
            let (image, output) = (x + plane * geometry.pixels, y + plane * geometry.outputPixels)
            for row in 0 ..< geometry.outputHeight {
                geometry.columnMaxima(of: image, outputRow: row, into: columnMaxima)
                geometry.windowMaxima(of: columnMaxima, into: output + row * geometry.outputWidth)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func maxPooling2dBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        if geometry.isHalving {
            // Every element belongs to at most one window, so every element of the gradient is written once.
            let (dx, beta) = inputGradient.elementsToWrite()
            for plane in 0 ..< geometry.planes {
                geometry.writeHalvingMaximumGradients(g + plane * geometry.outputPixels, of: x + plane * geometry.pixels, into: dx + plane * geometry.pixels, beta: beta)
            }
            return
        }
        // Every window adds its gradient to the position of its largest value.
        let dx = inputGradient.elementsToAddTo()
        for plane in 0 ..< geometry.planes {
            let (image, imageGradient) = (x + plane * geometry.pixels, dx + plane * geometry.pixels)
            let gradient = g + plane * geometry.outputPixels
            for row in 0 ..< geometry.outputHeight {
                // The maxima are found again instead of being kept alive between the forward and the backward pass.
                geometry.addMaximumGradients(gradient + row * geometry.outputWidth, of: image, outputRow: row, to: imageGradient)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func averagePooling2d<N: NumericType>(input: ShapedBuffer<N, CPU>, windowSize: Int, padding: Int, stride: Int, result: MutableShapedBuffer<N, CPU>) {
        let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride)
        let (x, y) = (input.elementPointer, result.elementPointer)
        // Padding elements are zeros that count for the mean.
        let inverseWindowElements = 1 / N(windowSize * windowSize)
        let columnSums = UnsafeMutablePointer<N>.allocate(capacity: geometry.width)
        defer {
            columnSums.deallocate()
        }
        for plane in 0 ..< geometry.planes {
            let (image, output) = (x + plane * geometry.pixels, y + plane * geometry.outputPixels)
            for row in 0 ..< geometry.outputHeight {
                geometry.columnSums(of: image, outputRow: row, into: columnSums)
                geometry.windowSums(of: columnSums, scale: inverseWindowElements, into: output + row * geometry.outputWidth)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func averagePooling2dBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, windowSize: Int, padding: Int, stride: Int, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        let geometry = PoolingGeometry(input: input, windowSize: windowSize, padding: padding, stride: stride)
        precondition(outputGradient.shape == geometry.outputShape, "The gradient of the result must have the shape of the result.")
        let g = outputGradient.elementPointer
        // Every window adds its gradient to the elements that it reads.
        let dx = inputGradient.elementsToAddTo()
        let inverseWindowElements = 1 / N(windowSize * windowSize)
        let columnGradient = UnsafeMutablePointer<N>.allocate(capacity: geometry.width)
        defer {
            columnGradient.deallocate()
        }
        for plane in 0 ..< geometry.planes {
            let (imageGradient, gradient) = (dx + plane * geometry.pixels, g + plane * geometry.outputPixels)
            for row in 0 ..< geometry.outputHeight {
                // The gradients of the windows of the row go to their columns first, and then to every row of the windows.
                geometry.spreadWindows(gradient + row * geometry.outputWidth, scale: inverseWindowElements, into: columnGradient)
                for imageRow in geometry.clampedRows(firstRow: row &* stride &- padding) {
                    let line = imageGradient + imageRow &* geometry.width
                    for j in 0 ..< geometry.width {
                        line[j] += columnGradient[j]
                    }
                }
            }
        }
    }
}

/// Shapes of a 2D convolution, which moves windows over images.
///
/// A convolution reads images of the shape [batchSize, imageChannels, height, width] and writes one value per window and
/// filter, shape [batchSize, outputChannels, outputHeight, outputWidth]. A transposed convolution is the adjoint of the
/// convolution whose images are the result of the transposed convolution, so it uses the geometry of that convolution.
struct ConvolutionGeometry {
    let batchSize: Int
    let imageChannels: Int
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
        imageChannels * kernelHeight * kernelWidth
    }

    /// Number of pixels per image channel.
    var imagePixels: Int {
        height * width
    }

    /// Number of elements per image.
    var imageElements: Int {
        imageChannels * imagePixels
    }

    var imageShape: [Int] {
        [batchSize, imageChannels, height, width]
    }

    var outputShape: [Int] {
        [batchSize, outputChannels, outputHeight, outputWidth]
    }

    /// Number of images per chunk, so that the window matrix of a chunk stays below the upper bound.
    var chunkSize: Int {
        Swift.max(1, Swift.min(batchSize, CPUKernels.maximumColumnElements / Swift.max(windowSize * windows, 1)))
    }

    /// The geometry of a convolution. The arguments must have the shapes that ``FusedOperationsType/convolution2d(input:filters:bias:padding:stride:result:)`` states.
    init<N>(input: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, padding: Int, stride: Int) {
        Self.checkShapes(input: input, filters: filters, bias: bias, stride: stride)
        precondition(padding >= 0, "The padding must not be negative.")
        precondition(input.shape[2] + 2 * padding >= filters.shape[2] && input.shape[3] + 2 * padding >= filters.shape[3], "The filters must fit into the padded images.")
        self.init(images: input.shape, filters: filters.shape, padding: padding, stride: stride)
    }

    /// The geometry of the convolution that is the adjoint of a transposed convolution.
    ///
    /// The arguments must have the shapes that ``FusedOperationsType/transposedConvolution2d(input:filters:bias:inset:stride:result:)`` states.
    /// The filters of the transposed convolution, shape [outputChannels, inputChannels, kernelHeight, kernelWidth], have the
    /// memory layout of the filters of the adjoint convolution, shape [inputChannels, outputChannels, kernelHeight, kernelWidth].
    init<N>(transposedInput input: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, inset: Int, stride: Int) {
        Self.checkShapes(input: input, filters: filters, bias: bias, stride: stride)
        precondition(inset >= 0, "The inset must not be negative.")
        let height = ConvUtil.transposedOutputSize(inputSize: input.shape[2], kernelSize: filters.shape[2], inset: inset, stride: stride)
        let width = ConvUtil.transposedOutputSize(inputSize: input.shape[3], kernelSize: filters.shape[3], inset: inset, stride: stride)
        precondition(height > 0 && width > 0, "The inset must leave a result of at least one element per axis.")
        self.init(images: [input.shape[0], filters.shape[0], height, width], filters: [filters.shape[1], filters.shape[0], filters.shape[2], filters.shape[3]], padding: inset, stride: stride)
    }

    /// Checks the shapes that both convolutions state.
    private static func checkShapes<N>(input: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, bias: ShapedBuffer<N, CPU>?, stride: Int) {
        precondition(input.dim == 4 && filters.dim == 4, "The images and the filters must have 4 axes.")
        precondition(input.shape[1] == filters.shape[1], "The images must have one channel for every input channel of the filters.")
        precondition(bias.map { $0.shape == [filters.shape[0]] } ?? true, "The bias must have one element for every output channel.")
        precondition(stride > 0, "The stride must be positive.")
    }

    private init(images: [Int], filters: [Int], padding: Int, stride: Int) {
        batchSize = images[0]
        imageChannels = images[1]
        height = images[2]
        width = images[3]
        outputChannels = filters[0]
        kernelHeight = filters[2]
        kernelWidth = filters[3]
        outputHeight = ConvUtil.outputSize(inputSize: height, kernelSize: kernelHeight, padding: padding, stride: stride)
        outputWidth = ConvUtil.outputSize(inputSize: width, kernelSize: kernelWidth, padding: padding, stride: stride)
        self.padding = padding
        self.stride = stride
    }

    /// Computes the products of the windows of the images, one chunk of images at a time, so that the windows are extracted once:
    /// the convolution `output = filters × windows(images) + bias + beta * output` for a beta of 0 or 1,
    /// and the gradient of the filters, `filterGradient += outputGradient × windows(images)ᵀ`.
    func multiplyWindows<N: NumericType>(
        of x: UnsafePointer<N>,
        filters w: UnsafePointer<N>,
        bias b: UnsafePointer<N>?,
        output: (elements: UnsafeMutablePointer<N>, beta: N)?,
        filterGradient: (outputGradient: UnsafePointer<N>, elements: UnsafeMutablePointer<N>)?,
    ) {
        let chunk = chunkSize
        let columns = UnsafeMutablePointer<N>.allocate(capacity: windowSize * chunk * windows)
        let matrix = UnsafeMutablePointer<N>.allocate(capacity: chunk > 1 ? outputChannels * chunk * windows : 0)
        defer {
            columns.deallocate()
            matrix.deallocate()
        }
        forEachChunk { first, count in
            extractWindows(from: x, firstImage: first, imageCount: count, into: columns)
            if let (y, beta) = output {
                let result = y + first * outputChannels * windows
                if count == 1 {
                    // A single image is multiplied directly into its slice of the result, which starts with the bias.
                    if let b, beta == 0 {
                        for channel in 0 ..< outputChannels {
                            CPUKernels.fill(result + channel * windows, with: b[channel], count: windows)
                        }
                        CPUKernels.gemm(w, shape: (outputChannels, windowSize), columns, shape: (windowSize, windows), into: result, beta: 1)
                    } else {
                        CPUKernels.gemm(w, shape: (outputChannels, windowSize), columns, shape: (windowSize, windows), into: result, beta: beta)
                        if let b {
                            CPUKernels.addChannelOffsets(b, to: result, images: 1, channels: outputChannels, pixels: windows)
                        }
                    }
                } else {
                    // The product has the layout [outputChannels, images, windows]. The result has the layout [images, outputChannels, windows].
                    CPUKernels.gemm(w, shape: (outputChannels, windowSize), columns, shape: (windowSize, count * windows), into: matrix)
                    CPUKernels.swapLeadingAxes(matrix, first: outputChannels, second: count, length: windows, adding: b, into: result, beta: beta)
                }
            }
            if let (g, dW) = filterGradient {
                let gradient = gradientInProductLayout(g, firstImage: first, imageCount: count, scratch: matrix)
                CPUKernels.gemm(gradient, shape: (outputChannels, count * windows), columns, shape: (windowSize, count * windows), rhsTransposed: true, into: dW, beta: 1)
            }
        }
    }

    /// Computes the products of the gradient of the output, one chunk of images at a time, so that the gradient is brought
    /// into the layout of the matrix product once: the gradient of the images, `images = windowsᵀ(filtersᵀ × outputGradient) + beta * images`
    /// for a beta of 0 or 1, and the gradient of the filters, `filterGradient += outputGradient × windows(images)ᵀ`.
    func multiplyOutputGradient<N: NumericType>(
        _ g: UnsafePointer<N>,
        filters w: UnsafePointer<N>,
        imageGradient: (elements: UnsafeMutablePointer<N>, beta: N)?,
        filterGradient: (images: UnsafePointer<N>, elements: UnsafeMutablePointer<N>)?,
    ) {
        let chunk = chunkSize
        let columns = UnsafeMutablePointer<N>.allocate(capacity: windowSize * chunk * windows)
        let gradientMatrix = UnsafeMutablePointer<N>.allocate(capacity: chunk > 1 ? outputChannels * chunk * windows : 0)
        // col2img overwrites its result, so the images of a chunk go through a scratch buffer when they are added.
        let addsImages = imageGradient.map { $0.beta != 0 } ?? false
        let chunkImages = UnsafeMutablePointer<N>.allocate(capacity: addsImages ? chunk * imageElements : 0)
        defer {
            columns.deallocate()
            gradientMatrix.deallocate()
            chunkImages.deallocate()
        }
        forEachChunk { first, count in
            let gradient = gradientInProductLayout(g, firstImage: first, imageCount: count, scratch: gradientMatrix)
            if let (x, dW) = filterGradient {
                // The windows are extracted again instead of being kept alive between the forward and the backward pass.
                extractWindows(from: x, firstImage: first, imageCount: count, into: columns)
                CPUKernels.gemm(gradient, shape: (outputChannels, count * windows), columns, shape: (windowSize, count * windows), rhsTransposed: true, into: dW, beta: 1)
            }
            if let (dx, _) = imageGradient {
                CPUKernels.gemm(w, shape: (outputChannels, windowSize), lhsTransposed: true, gradient, shape: (outputChannels, count * windows), into: columns)
                let images = dx + first * imageElements
                if addsImages {
                    writeImages(from: columns, imageCount: count, into: chunkImages)
                    CPUKernels.store(chunkImages, into: images, beta: 1, count: count * imageElements)
                } else {
                    writeImages(from: columns, imageCount: count, into: images)
                }
            }
        }
    }

    /// Calls `body` with the first image and the number of images of every chunk.
    private func forEachChunk(_ body: (_ firstImage: Int, _ imageCount: Int) -> Void) {
        CPUKernels.forEachBlock(count: batchSize, blockSize: chunkSize, body)
    }

    /// Returns the gradient of the output of consecutive images in the layout of the matrix product, [outputChannels, images, windows].
    /// It uses the scratch buffer for more than one image.
    private func gradientInProductLayout<N: NumericType>(_ g: UnsafePointer<N>, firstImage: Int, imageCount: Int, scratch: UnsafeMutablePointer<N>) -> UnsafePointer<N> {
        let gradient = g + firstImage * outputChannels * windows
        guard imageCount > 1 else {
            return gradient
        }
        CPUKernels.swapLeadingAxes(gradient, first: imageCount, second: outputChannels, length: windows, into: scratch)
        return UnsafePointer(scratch)
    }

    /// Writes the windows of consecutive images as a [windowSize, imageCount \* windows] matrix.
    private func extractWindows<N: NumericType>(from x: UnsafePointer<N>, firstImage: Int, imageCount: Int, into columns: UnsafeMutablePointer<N>) {
        N.img2col(
            values: UnsafeBufferPointer(start: x + firstImage * imageElements, count: imageCount * imageElements),
            result: UnsafeMutableBufferPointer(start: columns, count: windowSize * imageCount * windows),
            batchSize: imageCount,
            channels: imageChannels,
            height: height,
            width: width,
            kernelHeight: kernelHeight,
            kernelWidth: kernelWidth,
            padding: padding,
            stride: stride,
        )
    }

    /// Writes the sums of the windows of a [windowSize, imageCount \* windows] matrix into consecutive images.
    private func writeImages<N: NumericType>(from columns: UnsafePointer<N>, imageCount: Int, into images: UnsafeMutablePointer<N>) {
        N.col2img(
            values: UnsafeBufferPointer(start: columns, count: windowSize * imageCount * windows),
            result: UnsafeMutableBufferPointer(start: images, count: imageCount * imageElements),
            batchSize: imageCount,
            channels: imageChannels,
            height: height,
            width: width,
            kernelHeight: kernelHeight,
            kernelWidth: kernelWidth,
            padding: padding,
            stride: stride,
        )
    }
}

extension CPUKernels {
    /// Adds the sum of every channel of an [images, channels, pixels] array to the elements of `sums`.
    static func addChannelSums<N: NumericType>(_ values: UnsafePointer<N>, images: Int, channels: Int, pixels: Int, to sums: UnsafeMutablePointer<N>) {
        for image in 0 ..< images {
            for channel in 0 ..< channels {
                sums[channel] += sum(values + (image * channels + channel) * pixels, count: pixels)
            }
        }
    }

    /// Adds one value per channel to every pixel of the channel of an [images, channels, pixels] array.
    static func addChannelOffsets<N: NumericType>(_ offsets: UnsafePointer<N>, to values: UnsafeMutablePointer<N>, images: Int, channels: Int, pixels: Int) {
        for image in 0 ..< images {
            for channel in 0 ..< channels {
                let row = values + (image * channels + channel) * pixels
                let offset = offsets[channel]
                for j in 0 ..< pixels {
                    row[j] += offset
                }
            }
        }
    }
}

// The pooling kernels reduce the rows of the windows of one output row first, which is an element-wise loop over the
// columns, and then the columns of every window. Windows of size 2 with stride 2 and no padding have a loop without
// window bounds, which the compiler vectorizes.

/// Shapes of 2D pooling.
struct PoolingGeometry {
    let planes: Int
    let height: Int
    let width: Int
    let windowSize: Int
    let padding: Int
    let stride: Int
    let outputHeight: Int
    let outputWidth: Int

    var pixels: Int {
        height * width
    }

    var outputPixels: Int {
        outputHeight * outputWidth
    }

    var outputShape: [Int] {
        [planes / channels, channels, outputHeight, outputWidth]
    }

    private let channels: Int

    /// Whether the windows have the size 2, the stride 2, and no padding.
    var isHalving: Bool {
        windowSize == 2 && stride == 2 && padding == 0
    }

    /// The geometry of a pooling operation. The arguments must have the shapes that ``FusedOperationsType/maxPooling2d(input:windowSize:padding:stride:result:)`` states.
    init<N>(input: ShapedBuffer<N, CPU>, windowSize: Int, padding: Int, stride: Int) {
        precondition(input.dim == 4, "The images must have 4 axes.")
        precondition(windowSize > 0 && stride > 0 && padding >= 0, "The window size and the stride must be positive, and the padding must not be negative.")
        precondition(input.shape[2] + 2 * padding >= windowSize && input.shape[3] + 2 * padding >= windowSize, "The windows must fit into the padded images.")
        channels = input.shape[1]
        planes = input.shape[0] * input.shape[1]
        height = input.shape[2]
        width = input.shape[3]
        self.windowSize = windowSize
        self.padding = padding
        self.stride = stride
        outputHeight = ConvUtil.outputSize(inputSize: height, kernelSize: windowSize, padding: padding, stride: stride)
        outputWidth = ConvUtil.outputSize(inputSize: width, kernelSize: windowSize, padding: padding, stride: stride)
    }

    /// Rows of a window that are not in the padding.
    @inline(__always)
    func clampedRows(firstRow: Int) -> Range<Int> {
        Swift.max(firstRow, 0) ..< Swift.max(Swift.min(firstRow &+ windowSize, height), Swift.max(firstRow, 0))
    }

    /// Columns of a window that are not in the padding.
    @inline(__always)
    func clampedColumns(firstColumn: Int) -> Range<Int> {
        Swift.max(firstColumn, 0) ..< Swift.max(Swift.min(firstColumn &+ windowSize, width), Swift.max(firstColumn, 0))
    }

    /// Writes the largest value of every column of the rows of the windows of an output row. Padding rows add zeros.
    @inline(__always)
    func columnMaxima<N: NumericType>(of image: UnsafePointer<N>, outputRow: Int, into maxima: UnsafeMutablePointer<N>) {
        let rows = clampedRows(firstRow: outputRow &* stride &- padding)
        guard let firstRow = rows.first else {
            CPUKernels.fill(maxima, with: 0, count: width)
            return
        }
        maxima.update(from: image + firstRow &* width, count: width)
        for row in rows.dropFirst() {
            let line = image + row &* width
            for j in 0 ..< width {
                maxima[j] = Swift.max(maxima[j], line[j])
            }
        }
        if rows.count < windowSize {
            for j in 0 ..< width {
                maxima[j] = Swift.max(maxima[j], 0)
            }
        }
    }

    /// Writes the largest value of every window of an output row, given the maxima of its columns. Padding columns add zeros.
    @inline(__always)
    func windowMaxima<N: NumericType>(of columnMaxima: UnsafePointer<N>, into maxima: UnsafeMutablePointer<N>) {
        if isHalving {
            for column in 0 ..< outputWidth {
                let (left, right) = (columnMaxima[column &* 2], columnMaxima[column &* 2 &+ 1])
                maxima[column] = Swift.max(left, right)
            }
            return
        }
        for column in 0 ..< outputWidth {
            let columns = clampedColumns(firstColumn: column &* stride &- padding)
            var best: N = columns.count < windowSize ? 0 : columnMaxima[columns.lowerBound]
            for j in columns {
                best = Swift.max(best, columnMaxima[j])
            }
            maxima[column] = best
        }
    }

    /// Adds the gradient of every window of an output row to the position of the largest value of the window.
    ///
    /// The elements of a window are compared in row-major order, and the first largest value wins, as in the default
    /// implementation. The gradient of a padding element is dropped.
    @inline(__always)
    func addMaximumGradients<N: NumericType>(_ gradient: UnsafePointer<N>, of image: UnsafePointer<N>, outputRow: Int, to imageGradient: UnsafeMutablePointer<N>) {
        let firstRow = outputRow &* stride &- padding
        for column in 0 ..< outputWidth {
            let firstColumn = column &* stride &- padding
            let position = isInside(firstRow: firstRow, firstColumn: firstColumn)
                ? interiorMaximumPosition(in: image, firstRow: firstRow, firstColumn: firstColumn)
                : maximumPosition(in: image, firstRow: firstRow, firstColumn: firstColumn)
            if let position {
                imageGradient[position] += gradient[column]
            }
        }
    }

    /// Computes the gradient of max pooling of one plane with windows of the size 2, the stride 2, and no padding.
    ///
    /// The gradient of a window goes to its first largest value in row-major order. With beta 0, the other elements of the
    /// gradient are set to 0, and with beta 1, they do not change.
    @inline(__always)
    func writeHalvingMaximumGradients<N: NumericType>(_ gradient: UnsafePointer<N>, of image: UnsafePointer<N>, into imageGradient: UnsafeMutablePointer<N>, beta: N) {
        for outputRow in 0 ..< outputHeight {
            let (top, bottom) = (image + outputRow &* 2 &* width, image + (outputRow &* 2 &+ 1) &* width)
            let (topGradient, bottomGradient) = (imageGradient + outputRow &* 2 &* width, imageGradient + (outputRow &* 2 &+ 1) &* width)
            let rowGradient = gradient + outputRow &* outputWidth
            if beta == 0 {
                for column in 0 ..< outputWidth {
                    let (left, right) = (column &* 2, column &* 2 &+ 1)
                    let gradients = Self.halvingMaximumGradients(top[left], top[right], bottom[left], bottom[right], gradient: rowGradient[column])
                    (topGradient[left], topGradient[right], bottomGradient[left], bottomGradient[right]) = gradients
                }
            } else {
                for column in 0 ..< outputWidth {
                    let (left, right) = (column &* 2, column &* 2 &+ 1)
                    let gradients = Self.halvingMaximumGradients(top[left], top[right], bottom[left], bottom[right], gradient: rowGradient[column])
                    topGradient[left] += gradients.topLeft
                    topGradient[right] += gradients.topRight
                    bottomGradient[left] += gradients.bottomLeft
                    bottomGradient[right] += gradients.bottomRight
                }
            }
            // With an odd width, no window reads the last column.
            if beta == 0, width > outputWidth &* 2 {
                topGradient[width &- 1] = 0
                bottomGradient[width &- 1] = 0
            }
        }
        // With an odd height, no window reads the last row.
        if beta == 0, height > outputHeight &* 2 {
            CPUKernels.fill(imageGradient + (height &- 1) &* width, with: 0, count: width)
        }
    }

    /// The gradients of the four elements of a window of the size 2: the gradient of the window at its first largest value
    /// in row-major order, and 0 elsewhere.
    @inline(__always)
    private static func halvingMaximumGradients<N: NumericType>(_ a: N, _ b: N, _ c: N, _ d: N, gradient: N) -> (topLeft: N, topRight: N, bottomLeft: N, bottomRight: N) {
        let bottomWins = Swift.max(c, d) > Swift.max(a, b)
        let (bIsTopMaximum, dIsBottomMaximum) = (b > a, d > c)
        return (
            !bottomWins && !bIsTopMaximum ? gradient : 0,
            !bottomWins && bIsTopMaximum ? gradient : 0,
            bottomWins && !dIsBottomMaximum ? gradient : 0,
            bottomWins && dIsBottomMaximum ? gradient : 0,
        )
    }

    /// Whether a window with the given top left position lies completely inside the image.
    @inline(__always)
    private func isInside(firstRow: Int, firstColumn: Int) -> Bool {
        firstRow >= 0 && firstColumn >= 0 && firstRow &+ windowSize <= height && firstColumn &+ windowSize <= width
    }

    /// Returns the index in the plane of the largest value of a window that lies completely inside the image.
    @inline(__always)
    private func interiorMaximumPosition<N: NumericType>(in image: UnsafePointer<N>, firstRow: Int, firstColumn: Int) -> Int {
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
        return position
    }

    /// Returns the index in the plane of the largest value of a window, or nil when the largest value is a padding zero.
    @inline(__always)
    private func maximumPosition<N: NumericType>(in image: UnsafePointer<N>, firstRow: Int, firstColumn: Int) -> Int? {
        var best: N = 0
        var position: Int?
        var found = false
        for row in firstRow ..< firstRow &+ windowSize {
            for column in firstColumn ..< firstColumn &+ windowSize {
                let isInside = row >= 0 && row < height && column >= 0 && column < width
                let value = isInside ? image[row &* width &+ column] : 0
                if !found || value > best {
                    best = value
                    position = isInside ? row &* width &+ column : nil
                    found = true
                }
            }
        }
        return position
    }

    /// Writes the sum of every column of the rows of the windows of an output row. Padding rows add zeros.
    @inline(__always)
    func columnSums<N: NumericType>(of image: UnsafePointer<N>, outputRow: Int, into sums: UnsafeMutablePointer<N>) {
        let rows = clampedRows(firstRow: outputRow &* stride &- padding)
        guard let firstRow = rows.first else {
            CPUKernels.fill(sums, with: 0, count: width)
            return
        }
        sums.update(from: image + firstRow &* width, count: width)
        for row in rows.dropFirst() {
            let line = image + row &* width
            for j in 0 ..< width {
                sums[j] += line[j]
            }
        }
    }

    /// Writes the sum of every window of an output row, times `scale`, given the sums of its columns.
    @inline(__always)
    func windowSums<N: NumericType>(of columnSums: UnsafePointer<N>, scale: N, into sums: UnsafeMutablePointer<N>) {
        if isHalving {
            for column in 0 ..< outputWidth {
                let (left, right) = (columnSums[column &* 2], columnSums[column &* 2 &+ 1])
                sums[column] = (left + right) * scale
            }
            return
        }
        for column in 0 ..< outputWidth {
            var sum: N = 0
            for j in clampedColumns(firstColumn: column &* stride &- padding) {
                sum += columnSums[j]
            }
            sums[column] = sum * scale
        }
    }

    /// Writes, for every column of the image, the sum of the gradients of the windows of an output row that read the column, times `scale`.
    @inline(__always)
    func spreadWindows<N: NumericType>(_ gradient: UnsafePointer<N>, scale: N, into columnGradient: UnsafeMutablePointer<N>) {
        if isHalving {
            for column in 0 ..< outputWidth {
                let value = gradient[column] * scale
                columnGradient[column &* 2] = value
                columnGradient[column &* 2 &+ 1] = value
            }
            // An odd width has a last column that no window reads.
            for j in outputWidth &* 2 ..< width {
                columnGradient[j] = 0
            }
            return
        }
        CPUKernels.fill(columnGradient, with: 0, count: width)
        for column in 0 ..< outputWidth {
            let value = gradient[column] * scale
            for j in clampedColumns(firstColumn: column &* stride &- padding) {
                columnGradient[j] += value
            }
        }
    }
}

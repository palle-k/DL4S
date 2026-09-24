//
//  ConvTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 13.03.19.
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

import DL4S
import Testing

struct ConvTests {
    /// Convolution with plain loops, as a reference for the img2col-based kernel.
    ///
    /// `input` has the shape [batch, channels, height, width]. `filters` has the shape [outputChannels, channels, kernelHeight, kernelWidth].
    private func referenceConvolution(_ input: Tensor<Float, CPU>, filters: Tensor<Float, CPU>, padding: Int) -> Tensor<Float, CPU> {
        let (batch, channels, height, width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        let (outputChannels, kernelHeight, kernelWidth) = (filters.shape[0], filters.shape[2], filters.shape[3])
        let outputHeight = height + 2 * padding - kernelHeight + 1
        let outputWidth = width + 2 * padding - kernelWidth + 1
        let values = input.elements
        let weights = filters.elements
        var result = [Float](repeating: 0, count: batch * outputChannels * outputHeight * outputWidth)

        for b in 0 ..< batch {
            for o in 0 ..< outputChannels {
                for y in 0 ..< outputHeight {
                    for x in 0 ..< outputWidth {
                        var sum: Float = 0
                        for c in 0 ..< channels {
                            for ky in 0 ..< kernelHeight {
                                for kx in 0 ..< kernelWidth {
                                    let inputY = y + ky - padding
                                    let inputX = x + kx - padding
                                    guard 0 ..< height ~= inputY, 0 ..< width ~= inputX else {
                                        continue
                                    }
                                    sum += values[((b * channels + c) * height + inputY) * width + inputX] * weights[((o * channels + c) * kernelHeight + ky) * kernelWidth + kx]
                                }
                            }
                        }
                        result[((b * outputChannels + o) * outputHeight + y) * outputWidth + x] = sum
                    }
                }
            }
        }
        return Tensor(result, shape: [batch, outputChannels, outputHeight, outputWidth])
    }

    /// Transposed convolution with plain loops. Every input pixel adds a scaled copy of the kernel to the output.
    private func referenceTransposedConvolution(_ input: Tensor<Float, CPU>, filters: Tensor<Float, CPU>, inset: Int, stride: Int) -> Tensor<Float, CPU> {
        let (batch, channels, height, width) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        let (outputChannels, kernelHeight, kernelWidth) = (filters.shape[0], filters.shape[2], filters.shape[3])
        let outputHeight = (height - 1) * stride - 2 * inset + kernelHeight
        let outputWidth = (width - 1) * stride - 2 * inset + kernelWidth
        let values = input.elements
        let weights = filters.elements
        var result = [Float](repeating: 0, count: batch * outputChannels * outputHeight * outputWidth)

        for b in 0 ..< batch {
            for o in 0 ..< outputChannels {
                for c in 0 ..< channels {
                    for y in 0 ..< height {
                        for x in 0 ..< width {
                            let value = values[((b * channels + c) * height + y) * width + x]
                            for ky in 0 ..< kernelHeight {
                                for kx in 0 ..< kernelWidth {
                                    let outputY = y * stride + ky - inset
                                    let outputX = x * stride + kx - inset
                                    guard 0 ..< outputHeight ~= outputY, 0 ..< outputWidth ~= outputX else {
                                        continue
                                    }
                                    result[((b * outputChannels + o) * outputHeight + outputY) * outputWidth + outputX] += value * weights[((o * channels + c) * kernelHeight + ky) * kernelWidth + kx]
                                }
                            }
                        }
                    }
                }
            }
        }
        return Tensor(result, shape: [batch, outputChannels, outputHeight, outputWidth])
    }

    /// Four copies of a 4x4 ramp image, each scaled by a different factor.
    private func makeScaledRamps() -> Tensor<Float, CPU> {
        let ramp = Tensor<Float, CPU>((0 ..< 16).map(Float.init), shape: 1, 1, 4, 4)
        return ramp.repeated(4) * Tensor<Float, CPU>([1, 0.5, 0.25, 0.125]).view(as: 4, 1, 1, 1)
    }

    @Test func testIm2col() {
        let images = makeScaledRamps()
        let scales: [Float] = [1, 0.5, 0.25, 0.125]

        let result = images.img2col(kernelWidth: 3, kernelHeight: 3, padding: 0, stride: 1)

        // Rows are kernel positions, columns are windows in batch-major order.
        #expect(result.shape == [9, 16])
        var expected = [Float](repeating: 0, count: 9 * 16)
        for ky in 0 ..< 3 {
            for kx in 0 ..< 3 {
                for b in 0 ..< 4 {
                    for y in 0 ..< 2 {
                        for x in 0 ..< 2 {
                            expected[(ky * 3 + kx) * 16 + b * 4 + y * 2 + x] = scales[b] * Float((y + ky) * 4 + (x + kx))
                        }
                    }
                }
            }
        }
        #expect(result == Tensor(expected, shape: [9, 16]))
    }

    @Test func testConv1() {
        let images = makeScaledRamps()
        let filters = Tensor<Float, CPU>([
            [
                [[1]],
            ],
            [
                [[-1]],
            ],
        ])

        let result = images.convolved2d(filters: filters)

        #expect(result.shape == [4, 2, 4, 4])
        #expect(result[nil, 0] == images[nil, 0])
        #expect(result[nil, 1] == -images[nil, 0])
    }

    @Test func testConv() {
        let filters = Tensor<Float, CPU>([
            [
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]],
            ],
            [
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]],
            ],
        ]) / Tensor<Float, CPU>([16, 4]).view(as: -1, 1, 1, 1)
        let batch = MNIST.sample.trainingImages[0 ..< 8]

        let filtered = batch.convolved2d(filters: filters)

        #expect(filtered.shape == [8, 2, 28, 28])
        expectClose(filtered, referenceConvolution(batch, filters: filters, padding: 1), tolerance: 1e-6)
    }

    @Test func testTransposedConv() {
        let filters = Tensor<Float, CPU>([
            [
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]],
            ],
            [
                [[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]],
            ],
        ]) / Tensor<Float, CPU>([4, 1]).view(as: -1, 1, 1, 1)
        let batch = MNIST.sample.trainingImages[0 ..< 8]

        let filtered = batch.transposedConvolved2d(filters: filters, stride: 2)

        #expect(filtered.shape == [8, 2, 55, 55])
        expectClose(filtered, referenceTransposedConvolution(batch, filters: filters, inset: 1, stride: 2), tolerance: 1e-6)
    }

    /// Window matrix with plain loops, as a reference for img2col: row (channel, kernel row, kernel column), column (image, output row, output column).
    private func referenceWindows(_ input: [Double], shape: [Int], kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) -> [Double] {
        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3])
        let outputHeight = (height + 2 * padding - kernelHeight) / stride + 1
        let outputWidth = (width + 2 * padding - kernelWidth) / stride + 1
        let columnCount = batch * outputHeight * outputWidth
        var result = [Double](repeating: 0, count: channels * kernelHeight * kernelWidth * columnCount)
        for channel in 0 ..< channels {
            for kernelRow in 0 ..< kernelHeight {
                for kernelColumn in 0 ..< kernelWidth {
                    let row = (channel * kernelHeight + kernelRow) * kernelWidth + kernelColumn
                    for image in 0 ..< batch {
                        for outputRow in 0 ..< outputHeight {
                            for outputColumn in 0 ..< outputWidth {
                                let (y, x) = (outputRow * stride - padding + kernelRow, outputColumn * stride - padding + kernelColumn)
                                guard y >= 0, y < height, x >= 0, x < width else {
                                    continue
                                }
                                let column = (image * outputHeight + outputRow) * outputWidth + outputColumn
                                result[row * columnCount + column] = input[((image * channels + channel) * height + y) * width + x]
                            }
                        }
                    }
                }
            }
        }
        return result
    }

    /// img2col matches its definition, and col2img is its adjoint: `<img2col(x), c> == <x, col2img(c)>`.
    @Test(arguments: [
        (kernel: (3, 3), padding: 1, stride: 1),
        (kernel: (3, 2), padding: 0, stride: 1),
        (kernel: (5, 5), padding: 2, stride: 2),
        (kernel: (4, 4), padding: 1, stride: 3),
        (kernel: (7, 7), padding: 3, stride: 2),
        (kernel: (1, 1), padding: 0, stride: 2),
        // Stride 1 with an output as wide as the input, with an output height that differs from the input height.
        (kernel: (5, 3), padding: 1, stride: 1),
        (kernel: (3, 5), padding: 2, stride: 1),
        // Some kernel columns move the window completely out of the image.
        (kernel: (1, 13), padding: 6, stride: 1),
    ])
    func testWindowsMatchDefinition(kernel: (height: Int, width: Int), padding: Int, stride: Int) {
        let shape = [2, 3, 7, 6]
        var generator = WyHash(seed: 3)
        let input = Tensor<Double, CPU>(uniformlyDistributedWithShape: shape, min: -1, max: 1, using: &generator)
        let windows = input.img2col(kernelWidth: kernel.width, kernelHeight: kernel.height, padding: padding, stride: stride)
        #expect(windows.elements == referenceWindows(input.elements, shape: shape, kernelHeight: kernel.height, kernelWidth: kernel.width, padding: padding, stride: stride))

        let columns = Tensor<Double, CPU>(uniformlyDistributedWithShape: windows.shape, min: -1, max: 1, using: &generator)
        let image = columns.col2img(kernelWidth: kernel.width, kernelHeight: kernel.height, padding: padding, stride: stride, resultShape: shape)
        let lhs = zip(windows.elements, columns.elements).map(*).reduce(0, +)
        let rhs = zip(input.elements, image.elements).map(*).reduce(0, +)
        expectEqual(lhs, rhs, accuracy: 1e-9)
    }
}

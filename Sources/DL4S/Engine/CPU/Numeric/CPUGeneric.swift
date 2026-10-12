//
//  CPUGeneric.swift
//  DL4S
//
//  Created by Palle Klewitz on 31.10.19.
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

/// Shapes of the window matrix of img2col and col2img.
struct WindowGeometry {
    let channels: Int
    let width: Int
    let kernelHeight: Int
    let kernelWidth: Int
    let padding: Int
    let stride: Int
    let outputHeight: Int
    let outputWidth: Int
    /// Number of elements of one channel of an image.
    let imageElements: Int
    /// Number of windows of one image.
    let windowsPerImage: Int
    /// Number of elements of a row of the window matrix: the windows of all images.
    let resultRowLength: Int

    /// Number of rows of the window matrix: one per channel and position in the kernel.
    var rows: Int {
        channels &* kernelHeight &* kernelWidth
    }

    init(batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        self.channels = channels
        self.width = width
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth
        self.padding = padding
        self.stride = stride
        outputHeight = (height + 2 * padding - kernelHeight) / stride + 1
        outputWidth = (width + 2 * padding - kernelWidth) / stride + 1
        imageElements = height * width
        windowsPerImage = outputHeight * outputWidth
        resultRowLength = windowsPerImage * batchSize
    }

    /// The channel and the position in the kernel of a row of the window matrix.
    @inline(__always)
    func kernelPosition(ofRow row: Int) -> (channel: Int, kernelRow: Int, kernelColumn: Int) {
        let kernelColumn = row % kernelWidth
        let rest = row / kernelWidth
        return (rest / kernelHeight, rest % kernelHeight, kernelColumn)
    }

    /// Writes the windows of one image, one channel, and one kernel position, for stride 1 and an output as wide as the input.
    ///
    /// The output rows then have the row length of the input, so the valid rows are one contiguous copy of the input,
    /// shifted by the kernel column. The columns that the shift moves across a row boundary, and the rows in the padding, are set to 0.
    @inline(__always)
    func copyShiftedImage<N: CPUNumeric>(from image: UnsafePointer<N>, into target: UnsafeMutablePointer<N>, height: Int, kernelRow: Int, kernelColumn: Int) {
        let firstRow = Swift.max(0, padding - kernelRow)
        let endRow = Swift.max(firstRow, Swift.min(outputHeight, height + padding - kernelRow))
        let shift = kernelColumn - padding
        target.initialize(repeating: .zero, count: firstRow &* width)
        (target + endRow &* width).initialize(repeating: .zero, count: (outputHeight &- endRow) &* width)
        guard endRow > firstRow else {
            return
        }
        let rows = target + firstRow &* width
        let rowCount = endRow &- firstRow
        guard Swift.abs(shift) < width else {
            rows.initialize(repeating: .zero, count: rowCount &* width)
            return
        }
        let source = image + (firstRow &+ kernelRow &- padding) &* width
        let count = rowCount &* width &- Swift.abs(shift)
        if shift >= 0 {
            rows.update(from: source + shift, count: count)
            for row in 0 ..< rowCount {
                (rows + (row &* width &+ width &- shift)).initialize(repeating: .zero, count: shift)
            }
        } else {
            (rows - shift).update(from: source, count: count)
            for row in 0 ..< rowCount {
                (rows + row &* width).initialize(repeating: .zero, count: -shift)
            }
        }
    }

    /// Output columns whose input column `column * stride - padding + kernelColumn` is inside the image.
    @inline(__always)
    func validOutputColumns(kernelColumn: Int) -> Range<Int> {
        // The first column is the smallest one with column * stride >= padding - kernelColumn,
        // the last one the largest with column * stride <= width - 1 + padding - kernelColumn.
        let lowerLimit = padding - kernelColumn
        let first = lowerLimit <= 0 ? 0 : (lowerLimit + stride - 1) / stride
        let upperLimit = width - 1 + padding - kernelColumn
        let last = upperLimit < 0 ? -1 : Swift.min(upperLimit / stride, outputWidth - 1)
        return first <= last ? first ..< last + 1 : first ..< first
    }
}

public extension CPUNumeric {
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func img2col(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        let geometry = WindowGeometry(batchSize: batchSize, channels: channels, height: height, width: width, kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride)
        let src = values.baseAddress!
        let dst = result.baseAddress!

        // Every row of the result belongs to one channel and one position in the kernel. For one image and one output row,
        // the row holds a contiguous run of outputWidth elements: zeros for the padding on the left, the elements of one
        // input row, and zeros for the padding on the right. The columns in the padding only depend on the kernel column.
        for row in 0 ..< geometry.rows {
            let (channel, kernelRow, kernelColumn) = geometry.kernelPosition(ofRow: row)
            let columns = geometry.validOutputColumns(kernelColumn: kernelColumn)
            let firstInputColumn = columns.lowerBound &* stride &- padding &+ kernelColumn
            for image in 0 ..< batchSize {
                let inputChannel = src + (image &* channels &+ channel) &* geometry.imageElements
                let resultImage = dst + (row &* geometry.resultRowLength &+ image &* geometry.windowsPerImage)
                if stride == 1, geometry.outputWidth == width {
                    geometry.copyShiftedImage(from: inputChannel, into: resultImage, height: height, kernelRow: kernelRow, kernelColumn: kernelColumn)
                    continue
                }
                for outputRow in 0 ..< geometry.outputHeight {
                    let target = resultImage + outputRow &* geometry.outputWidth
                    let inputRow = outputRow &* stride &- padding &+ kernelRow
                    guard inputRow >= 0, inputRow < height, !columns.isEmpty else {
                        target.initialize(repeating: .zero, count: geometry.outputWidth)
                        continue
                    }
                    target.initialize(repeating: .zero, count: columns.lowerBound)
                    (target + columns.upperBound).initialize(repeating: .zero, count: geometry.outputWidth &- columns.upperBound)
                    let source = inputChannel + (inputRow &* width &+ firstInputColumn)
                    let run = target + columns.lowerBound
                    if stride == 1 {
                        run.update(from: source, count: columns.count)
                    } else {
                        for i in 0 ..< columns.count {
                            run[i] = source[i &* stride]
                        }
                    }
                }
            }
        }
    }

    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func col2img(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        let geometry = WindowGeometry(batchSize: batchSize, channels: channels, height: height, width: width, kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride)
        let src = values.baseAddress!
        let dst = result.baseAddress!
        dst.initialize(repeating: .zero, count: batchSize &* channels &* geometry.imageElements)

        // The adjoint of img2col: the runs of every row are added to the input rows that img2col copied them from.
        // The rows are added in the same order as before, so the sums are the same.
        for row in 0 ..< geometry.rows {
            let (channel, kernelRow, kernelColumn) = geometry.kernelPosition(ofRow: row)
            let columns = geometry.validOutputColumns(kernelColumn: kernelColumn)
            guard !columns.isEmpty else {
                continue
            }
            let firstInputColumn = columns.lowerBound &* stride &- padding &+ kernelColumn
            for image in 0 ..< batchSize {
                let resultChannel = dst + (image &* channels &+ channel) &* geometry.imageElements
                let sourceImage = src + (row &* geometry.resultRowLength &+ image &* geometry.windowsPerImage)
                for outputRow in 0 ..< geometry.outputHeight {
                    let inputRow = outputRow &* stride &- padding &+ kernelRow
                    guard inputRow >= 0, inputRow < height else {
                        continue
                    }
                    let run = sourceImage + (outputRow &* geometry.outputWidth &+ columns.lowerBound)
                    let target = resultChannel + (inputRow &* width &+ firstInputColumn)
                    if stride == 1 {
                        for i in 0 ..< columns.count {
                            target[i] += run[i]
                        }
                    } else {
                        for i in 0 ..< columns.count {
                            target[i &* stride] += run[i]
                        }
                    }
                }
            }
        }
    }

    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    internal static func gemm_generic(_ transA: Bool, _ transB: Bool, _ __M: Int, _ __N: Int, _ __K: Int, _ alpha: Self, _ __A: UnsafePointer<Self>, _ lda: Int, _ __B: UnsafePointer<Self>, _ ldb: Int, _ beta: Self, _ __C: UnsafeMutablePointer<Self>, _ ldc: Int) {
        if __M == 0 || __N == 0 || ((alpha == 0 || __K == 0) && beta == 1) {
            return
        }

        if beta == 0 {
            for i in 0 ..< __M * __N {
                __C[i] = 0
            }
        } else {
            for i in 0 ..< __M * __N {
                __C[i] *= beta
            }
        }

        if alpha == 0 {
            return
        }

        if transA {
            if transB {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[r &+ l &* __M] * __B[l &+ c &* __K]
                        }
                        __C[r &* __N &+ c] += alpha * tmp
                    }
                }
            } else {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[r &+ l &* __M] * __B[l &* __N &+ c]
                        }
                        __C[r &* __N &+ c] += alpha * tmp
                    }
                }
            }
        } else {
            if transB {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[l &+ r &* __K] * __B[l &+ c &* __K]
                        }
                        __C[r &* __N &+ c] += alpha * tmp
                    }
                }
            } else {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[l &+ r &* __K] * __B[l &* __N &+ c]
                        }
                        __C[r &* __N &+ c] += alpha * tmp
                    }
                }
            }
        }
    }

    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func scatter(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, resultShape: [Int], axis: Int, ignoreIndex: Int32) {
        // The result is viewed as [outer, axis, inner], the values and the context as [outer, inner].
        let axisSize = resultShape[axis]
        let outer = resultShape[..<axis].reduce(1, *)
        let inner = resultShape[(axis + 1)...].reduce(1, *)
        fill(value: .zero, result: result, count: outer * axisSize * inner)
        guard outer * inner > 0 else {
            return
        }
        let (source, target, context) = (values.baseAddress!, result.baseAddress!, context.baseAddress!)
        for row in 0 ..< outer {
            for column in 0 ..< inner {
                let position = row * inner + column
                let index = context[position]
                if index == ignoreIndex {
                    continue
                }
                precondition(index >= 0 && Int(index) < axisSize, "Scatter index \(index) is out of range for an axis of size \(axisSize).")
                target[(row * axisSize + Int(index)) * inner + column] = source[position]
            }
        }
    }

    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func gather(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, valuesShape: [Int], axis: Int, ignoreIndex: Int32) {
        // The values are viewed as [outer, axis, inner], the result and the context as [outer, inner].
        let axisSize = valuesShape[axis]
        let outer = valuesShape[..<axis].reduce(1, *)
        let inner = valuesShape[(axis + 1)...].reduce(1, *)
        guard outer * inner > 0 else {
            return
        }
        let (source, target, context) = (values.baseAddress!, result.baseAddress!, context.baseAddress!)
        for row in 0 ..< outer {
            for column in 0 ..< inner {
                let position = row * inner + column
                let index = context[position]
                if index == ignoreIndex {
                    target[position] = 0
                    continue
                }
                precondition(index >= 0 && Int(index) < axisSize, "Gather index \(index) is out of range for an axis of size \(axisSize).")
                target[position] = source[(row * axisSize + Int(index)) * inner + column]
            }
        }
    }

    @_specialize(where Self == Int32)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhsPtr = lhs.baseAddress!
        let rhsPtr = rhs.baseAddress!
        let resultPtr = result.baseAddress!
        let contextPtr = context.baseAddress!

        var i = 0
        while i < count {
            let l = lhsPtr[i]
            let r = rhsPtr[i]
            if l >= r {
                resultPtr[i] = l
                contextPtr[i] = 0
            } else {
                resultPtr[i] = r
                contextPtr[i] = 1
            }

            i &+= 1
        }
    }

    @_specialize(where Self == Int32)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhsPtr = lhs.baseAddress!
        let rhsPtr = rhs.baseAddress!
        let resultPtr = result.baseAddress!
        let contextPtr = context.baseAddress!

        var i = 0
        while i < count {
            let l = lhsPtr[i]
            let r = rhsPtr[i]
            if l <= r {
                resultPtr[i] = l
                contextPtr[i] = 0
            } else {
                resultPtr[i] = r
                contextPtr[i] = 1
            }

            i &+= 1
        }
    }
}

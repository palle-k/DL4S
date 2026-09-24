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

#if MKL_ENABLE
import CMKL
#elseif canImport(Accelerate)
import Accelerate
#endif

// `Self.self == Float.self` is used as a test before buffers are force cast.
// This is safe.
// swiftlint:disable force_cast

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
    static func scatter(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, dst_shape: [Int], axis: Int, ignoreIndex: Int32) {
        let src = values.baseAddress!
        let target = result.baseAddress!
        let context = context.baseAddress!

        let dst_dim = dst_shape.count

        let src_strides = UnsafeMutablePointer<Int>.allocate(capacity: dst_dim - 1)
        let src_shape = UnsafeMutablePointer<Int>.allocate(capacity: dst_dim - 1)
        let dst_strides = UnsafeMutablePointer<Int>.allocate(capacity: dst_dim)

        defer {
            src_strides.deallocate()
            src_shape.deallocate()
            dst_strides.deallocate()
        }

        dst_strides[dst_dim - 1] = 1
        src_strides[dst_dim - 2] = 1

        for i in (0 ... (dst_dim - 2)).reversed() {
            dst_strides[i] = dst_shape[i &+ 1] &* dst_strides[i &+ 1]
        }
        for i in (0 ... (dst_dim - 2)).reversed() {
            src_shape[i] = dst_shape[i >= axis ? i &+ 1 : i]
            if i < dst_dim - 2 {
                src_strides[i] = src_shape[i &+ 1] &* src_strides[i &+ 1]
            } else {
                src_strides[i] = 1
            }
        }
        let count = src_shape[0] * src_strides[0]

        let dst_count = dst_strides[0] * dst_shape[0]
        if Self.self == Float.self {
            #if MKL_ENABLE
            ippsSet_32f(0, target as! UnsafeMutablePointer<Float>, Int32(dst_count))
            #elseif canImport(Accelerate)
            vDSP_vfill([0], target as! UnsafeMutablePointer<Float>, 1, UInt(dst_count))
            #else
            for i in 0 ..< dst_count {
                target[i] = 0
            }
            #endif
        } else {
            for i in 0 ..< dst_count {
                target[i] = 0
            }
        }

        for i in 0 ..< count {
            let src_idx = i
            let c = context[i]
            if c == ignoreIndex {
                continue
            }
            var dst_idx = Int(c) &* dst_strides[axis]

            for a in 0 ..< dst_dim - 1 {
                let src_dim_idx = (i / src_strides[a]) % src_shape[a]
                dst_idx = dst_idx &+ src_dim_idx &* dst_strides[a >= axis ? a &+ 1 : a]
            }
            target[dst_idx] = src[src_idx]
        }
    }

    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func gather(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, src_shape: [Int], axis: Int, ignoreIndex: Int32) {
        let src_dim = src_shape.count

        let src = values.baseAddress!
        let target = result.baseAddress!
        let context = context.baseAddress!

        let dst_strides = UnsafeMutablePointer<Int>.allocate(capacity: src_dim - 1)
        let dst_shape = UnsafeMutablePointer<Int>.allocate(capacity: src_dim - 1)
        let src_strides = UnsafeMutablePointer<Int>.allocate(capacity: src_dim)

        defer {
            dst_strides.deallocate()
            dst_shape.deallocate()
            src_strides.deallocate()
        }

        src_strides[src_dim - 1] = 1
        dst_strides[src_dim - 2] = 1

        for i in (0 ... (src_dim - 2)).reversed() {
            src_strides[i] = src_shape[i &+ 1] * src_strides[i &+ 1]
        }
        for i in (0 ... (src_dim - 2)).reversed() {
            dst_shape[i] = src_shape[i >= axis ? i &+ 1 : i]
            if i < src_dim &- 2 {
                dst_strides[i] = dst_shape[i &+ 1] &* dst_strides[i &+ 1]
            } else {
                dst_strides[i] = 1
            }
        }

        let count = dst_shape[0] &* dst_strides[0]

        for i in 0 ..< count {
            let dst_idx = i
            let c = context[i]
            if c == ignoreIndex {
                target[dst_idx] = 0
                continue
            }

            var src_idx = Int(c) &* src_strides[axis]

            for a in 0 ..< src_dim - 1 {
                let dst_dim_idx = (i / dst_strides[a]) % dst_shape[a]
                src_idx = src_idx &+ dst_dim_idx &* src_strides[a >= axis ? a &+ 1 : a]
            }
            target[dst_idx] = src[src_idx]
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

// swiftlint:enable force_cast

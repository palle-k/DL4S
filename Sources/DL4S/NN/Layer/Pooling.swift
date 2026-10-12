//
//  Pooling.swift
//  DL4S
//
//  Created by Palle Klewitz on 17.10.19.
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

/// A 2D max pooling layer
@Layer
public struct MaxPool2D<Element: NumericType, Device: DeviceType>: Codable, Sendable {
    /// Pooling window size
    public let windowSize: Int

    /// Pooling window stride
    public let stride: Int

    /// Padding applied around the edges of the input of the layer.
    public let padding: Int?

    /// Creates a 2D max pooling layer.
    /// - Parameters:
    ///   - windowSize: Size of the window
    ///   - stride: Stride, with which the window moves over the input tensor >= 1.
    ///   - padding: Padding applied around the edges of the input of the layer.
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.maxPooled2d(windowSize: windowSize, padding: padding, stride: stride)
    }
}

/// A 2D average pooling layer
@Layer
public struct AvgPool2D<Element: NumericType, Device: DeviceType>: Codable, Sendable {
    /// Pooling window size
    public let windowSize: Int

    /// Pooling window stride
    public let stride: Int

    /// Padding applied around the edges of the input of the layer.
    public let padding: Int?

    /// Creates a 2D average pooling layer.
    /// - Parameters:
    ///   - windowSize: Size of the window
    ///   - stride: Stride, with which the window moves over the input tensor >= 1.
    ///   - padding: Padding applied around the edges of the input of the layer.
    public init(windowSize: Int = 2, stride: Int = 2, padding: Int? = nil) {
        self.windowSize = windowSize
        self.stride = stride
        self.padding = padding
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.averagePooled2d(windowSize: windowSize, padding: padding, stride: stride)
    }
}

/// A 2D adaptive max pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
///
/// The layer expects inputs with the shape [batchSize, channels, height, width] and returns the shape
/// [batchSize, channels, targetSize, targetSize]. The output element at row `i` and column `j` is the maximum of the rows
/// `floor(i * height / targetSize) ..< ceil((i + 1) * height / targetSize)` and the matching columns of the input.
/// The height and the width of the input must be at least `targetSize`.
@Layer
public struct AdaptiveMaxPool2D<Element: NumericType, Device: DeviceType>: Codable, Sendable {
    /// Width and height of the output tensor
    public let targetSize: Int

    /// A 2D adaptive max pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
    /// - Parameter targetSize: Width and height of the output tensor
    public init(targetSize: Int) {
        precondition(targetSize > 0, "The target size must be positive.")
        self.targetSize = targetSize
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.adaptivelyPooled2d(targetSize: targetSize, reduction: .maximum)
    }
}

/// A 2D adaptive average pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
///
/// The layer expects inputs with the shape [batchSize, channels, height, width] and returns the shape
/// [batchSize, channels, targetSize, targetSize]. The output element at row `i` and column `j` is the mean of the rows
/// `floor(i * height / targetSize) ..< ceil((i + 1) * height / targetSize)` and the matching columns of the input.
/// The height and the width of the input must be at least `targetSize`.
@Layer
public struct AdaptiveAvgPool2D<Element: NumericType, Device: DeviceType>: Codable, Sendable {
    /// Width and height of the output tensor
    public let targetSize: Int

    /// A 2D adaptive average pooling layer that pools its inputs with an automatically computed stride and window size to reach the desired output size
    /// - Parameter targetSize: Width and height of the output tensor
    public init(targetSize: Int) {
        precondition(targetSize > 0, "The target size must be positive.")
        self.targetSize = targetSize
    }

    #if canImport(Metal) && canImport(MetalPerformanceShaders)
    @_specialize(where Element == Float, Device == GPU)
    #endif
    @_specialize(where Element == Float, Device == CPU)
    public func callAsFunction(_ inputs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        inputs.adaptivelyPooled2d(targetSize: targetSize, reduction: .mean)
    }
}

/// The reduction of the windows of adaptive pooling.
private enum AdaptivePoolingReduction {
    case maximum
    case mean
}

private extension Tensor {
    /// Reduces windows of the last two axes of a tensor with the shape [batchSize, channels, height, width] to the
    /// shape [batchSize, channels, targetSize, targetSize].
    func adaptivelyPooled2d(targetSize: Int, reduction: AdaptivePoolingReduction) -> Self {
        precondition(dim == 4, "The input must have the shape [batchSize, channels, height, width].")
        let height = shape[2]
        let width = shape[3]
        precondition(height >= targetSize && width >= targetSize, "The height and the width of the input (\(height) x \(width)) must be at least the target size \(targetSize).")

        // When the windows have the same size and do not overlap, the pooling operation or one reduction computes
        // the result. Other sizes need windows of different sizes, which are reduced one axis at a time.
        guard height.isMultiple(of: targetSize), width.isMultiple(of: targetSize) else {
            return reducingWindows(along: 2, targetSize: targetSize, reduction: reduction)
                .reducingWindows(along: 3, targetSize: targetSize, reduction: reduction)
        }
        let windowHeight = height / targetSize
        let windowWidth = width / targetSize
        if windowHeight == windowWidth {
            return switch reduction {
            case .maximum: maxPooled2d(windowSize: windowHeight, padding: 0, stride: windowHeight)
            case .mean: averagePooled2d(windowSize: windowHeight, padding: 0, stride: windowHeight)
            }
        }
        return view(as: [shape[0], shape[1], targetSize, windowHeight, targetSize, windowWidth])
            .reduced(along: 5, by: reduction)
            .reduced(along: 3, by: reduction)
    }

    /// Reduces the windows of adaptive pooling along one axis, which then has the size `targetSize`.
    func reducingWindows(along axis: Int, targetSize: Int, reduction: AdaptivePoolingReduction) -> Self {
        let size = shape[axis]
        let windows = (0 ..< targetSize).map { index in
            var ranges: [Range<Int>?] = Array(repeating: nil, count: dim)
            ranges[axis] = (index * size / targetSize) ..< ((index + 1) * size + targetSize - 1) / targetSize
            return self[ranges].reduced(along: axis, by: reduction).unsqueezed(at: axis)
        }
        return Tensor(stacking: windows, along: axis)
    }

    // The gradient of reduceMax supports one axis only, so the axes are reduced one at a time.
    func reduced(along axis: Int, by reduction: AdaptivePoolingReduction) -> Self {
        switch reduction {
        case .maximum: reduceMax(along: [axis])
        case .mean: reduceMean(along: [axis])
        }
    }
}

//
//  TensorExt.swift
//  DL4S
//
//  Created by Palle Klewitz on 04.10.19.
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

// MARK: Tensor extensions

extension Tensor: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        values.description
    }

    public var debugDescription: String {
        let contextDescription = if let ctx = context?.tag {
            ", context: \(ctx) "
        } else {
            " "
        }
        let elementString = if count == 1 {
            "\(count) element"
        } else {
            "\(count) elements"
        }
        let shapeString = if shape == [] {
            "scalar"
        } else {
            "\(shape)"
        }

        return """
        \(elementString) (\(shapeString))\(contextDescription){
            \(values.description.replacingOccurrences(of: "\n", with: "\n    "))
        }
        """
    }
}

extension Tensor: Equatable where Element: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.shape == rhs.shape && lhs.elements == rhs.elements
    }
}

extension Tensor: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init([Element(value)], shape: [])
    }
}

extension Tensor: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init([Element(value)], shape: [])
    }
}

public extension Tensor {
    /// Creates a scalar tensor with the given value. The tensor will have a shape of []
    /// - Parameter value: Value of the tensor.
    init(_ value: Element) {
        self.init([value], shape: [])
    }
}

public extension Tensor {
    /// Element at the first index in the tensor.
    var item: Element {
        Device.Memory.getValue(from: values.values)
    }
}

// MARK: Tensor - array conversion

public extension Tensor {
    /// Creates a tensor value holding the provided scalar. The tensor will have an empty shape.
    /// - Parameters:
    ///   - e: Element
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(_ e: Element, requiresGradient: Bool = false) {
        self.init([e], shape: [], requiresGradient: requiresGradient)
    }

    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Values to fill tensor with
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(_ v: [[Element]], requiresGradient: Bool = false) {
        self.init(Array(v.joined()), shape: [v.count, v.first?.count ?? 0], requiresGradient: requiresGradient)
    }

    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Values to fill tensor with
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(_ v: [[[Element]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined()),
            shape: [v.count, v.first?.count ?? 0, v.first?.first?.count ?? 0],
            requiresGradient: requiresGradient,
        )
    }

    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Values to fill tensor with
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(_ v: [[[[Element]]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined().joined()),
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0,
            ],
            requiresGradient: requiresGradient,
        )
    }

    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Values to fill tensor with
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(_ v: [[[[[Element]]]]], requiresGradient: Bool = false) {
        self.init(
            Array(v.joined().joined().joined().joined()),
            shape: [
                v.count,
                v.first?.count ?? 0,
                v.first?.first?.count ?? 0,
                v.first?.first?.first?.count ?? 0,
                v.first?.first?.first?.first?.count ?? 0,
            ],
            requiresGradient: requiresGradient,
        )
    }
}

// MARK: Tensor initialization

public extension Tensor where Element: RandomizableType {
    /// Creates a weight matrix with the Xavier (Glorot) normal initialization.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / (shape[0] + shape[1]))`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(xavierNormalWithShape shape: [Int], requiresGradient: Bool = false) {
        var generator = WyHash()
        self.init(xavierNormalWithShape: shape, requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a weight matrix with the Xavier (Glorot) normal initialization and values from the given generator.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / (shape[0] + shape[1]))`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    ///   - generator: Random number generator that provides the values.
    init<Generator: RandomNumberGenerator>(xavierNormalWithShape shape: [Int], requiresGradient: Bool = false, using generator: inout Generator) {
        precondition(shape.count == 2, "Shape must be 2-dimensional")
        self.init(normalDistributedWithShape: shape, mean: 0, stdev: (2 / Element(shape[0] + shape[1])).sqrt(), requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a weight matrix with the Xavier (Glorot) normal initialization.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / (shape[0] + shape[1]))`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(xavierNormalWithShape shape: Int..., requiresGradient: Bool = false) {
        self.init(xavierNormalWithShape: shape, requiresGradient: requiresGradient)
    }

    /// Creates a weight matrix with the He (Kaiming) normal initialization.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / shape[0])`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(heNormalWithShape shape: [Int], requiresGradient: Bool = false) {
        var generator = WyHash()
        self.init(heNormalWithShape: shape, requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a weight matrix with the He (Kaiming) normal initialization and values from the given generator.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / shape[0])`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    ///   - generator: Random number generator that provides the values.
    init<Generator: RandomNumberGenerator>(heNormalWithShape shape: [Int], requiresGradient: Bool = false, using generator: inout Generator) {
        precondition(shape.count == 2, "Shape must be 2-dimensional")
        self.init(normalDistributedWithShape: shape, mean: 0, stdev: (2 / Element(shape[0])).sqrt(), requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a weight matrix with the He (Kaiming) normal initialization.
    ///
    /// The values are sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / shape[0])`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, [fan in, fan out]. It must be two dimensional.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(heNormalWithShape shape: Int..., requiresGradient: Bool = false) {
        self.init(heNormalWithShape: shape, requiresGradient: requiresGradient)
    }

    /// Creates a tensor and fills it with random values sampled from a normal distribution with the given mean and variance.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - mean: Mean of the normal distribution.
    ///   - stdev: Standard deviation of the normal distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(normalDistributedWithShape shape: [Int], mean: Element = 0, stdev: Element = 1, requiresGradient: Bool = false) {
        var generator = WyHash()
        self.init(normalDistributedWithShape: shape, mean: mean, stdev: stdev, requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a tensor and fills it with random values from the given generator, sampled from a normal distribution with the given mean and variance.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - mean: Mean of the normal distribution.
    ///   - stdev: Standard deviation of the normal distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    ///   - generator: Random number generator that provides the values.
    init<Generator: RandomNumberGenerator>(normalDistributedWithShape shape: [Int], mean: Element = 0, stdev: Element = 1, requiresGradient: Bool = false, using generator: inout Generator) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.fillNormal(mutableValues, mean: mean, stdev: stdev, using: &generator)
    }

    /// Creates a tensor and fills it with random values sampled from a normal distribution with the given mean and variance.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - mean: Mean of the normal distribution.
    ///   - stdev: Standard deviation of the normal distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(normalDistributedWithShape shape: Int..., mean: Element = 0, stdev: Element = 1, requiresGradient: Bool = false) {
        self.init(normalDistributedWithShape: shape, mean: mean, stdev: stdev, requiresGradient: requiresGradient)
    }

    /// Creates a tensor and fills it with random values sampled from a uniform distribution with the given minimum and maximum.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - min: Minimum value of the uniform distribution
    ///   - max: Maximum value of the uniform distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(uniformlyDistributedWithShape shape: [Int], min: Element = 0, max: Element = 1, requiresGradient: Bool = false) {
        var generator = WyHash()
        self.init(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a tensor and fills it with random values from the given generator, sampled from a uniform distribution with the given minimum and maximum.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - min: Minimum value of the uniform distribution
    ///   - max: Maximum value of the uniform distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    ///   - generator: Random number generator that provides the values.
    init<Generator: RandomNumberGenerator>(uniformlyDistributedWithShape shape: [Int], min: Element = 0, max: Element = 1, requiresGradient: Bool = false, using generator: inout Generator) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.fill(mutableValues, a: min, b: max, using: &generator)
    }

    /// Creates a tensor and fills it with random values sampled from a uniform distribution with the given minimum and maximum.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - min: Minimum value of the uniform distribution
    ///   - max: Maximum value of the uniform distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(uniformlyDistributedWithShape shape: Int..., min: Element = 0, max: Element = 1, requiresGradient: Bool = false) {
        self.init(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient)
    }
}

public extension Tensor {
    /// Creates a tensor of ones and zeros, where each element is 1 with the given probability.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - probability: Probability of a 1
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(bernoulliDistributedWithShape shape: [Int], probability: Float, requiresGradient: Bool = false) {
        var generator = WyHash()
        self.init(bernoulliDistributedWithShape: shape, probability: probability, requiresGradient: requiresGradient, using: &generator)
    }

    /// Creates a tensor of ones and zeros from the given generator, where each element is 1 with the given probability.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - probability: Probability of a 1
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    ///   - generator: Random number generator that provides the values.
    init<Generator: RandomNumberGenerator>(bernoulliDistributedWithShape shape: [Int], probability: Float, requiresGradient: Bool = false, using generator: inout Generator) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.bernoulli(mutableValues, p: probability, using: &generator)
    }

    /// Creates a tensor of ones and zeros, where each element is 1 with the given probability.
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - probability: Probability of a 1
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(bernoulliDistributedWithShape shape: Int..., probability: Float, requiresGradient: Bool = false) {
        self.init(bernoulliDistributedWithShape: shape, probability: probability, requiresGradient: requiresGradient)
    }
}

extension Tensor: Codable where Element: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        requiresGradient = try container.decode(Bool.self, forKey: .requiresGradient)
        shape = try container.decode([Int].self, forKey: .shape)
        let data = try container.decode(Data.self, forKey: .data)
        let elementCount = shape.reduce(1, *)
        guard shape.allSatisfy({ $0 >= 0 }), data.count == elementCount * MemoryLayout<Element>.stride else {
            throw DecodingError.dataCorruptedError(
                forKey: .data,
                in: container,
                debugDescription: "The data has \(data.count) bytes, but a tensor with the shape \(shape) has \(elementCount * MemoryLayout<Element>.stride) bytes.",
            )
        }
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        handle = TensorHandle(values: buffer.values)
        if elementCount > 0 {
            // Data does not guarantee the alignment of Element, so the bytes go through an aligned copy.
            withUnsafeTemporaryAllocation(of: Element.self, capacity: elementCount) { elements in
                _ = data.copyBytes(to: elements)
                Device.Memory.assign(from: UnsafeBufferPointer(elements), to: buffer.values, count: elementCount)
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        let data = elements.withUnsafeBufferPointer { Data(buffer: $0) }

        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(requiresGradient, forKey: .requiresGradient)
        try container.encode(data, forKey: .data)
        try container.encode(shape, forKey: .shape)
    }

    private enum CodingKeys: String, CodingKey {
        case requiresGradient
        case data
        case shape
    }
}

public extension Tensor {
    // Retreives the elements of the tensor as a flattened array.
    var elements: [Element] {
        var array = [Element](repeating: 0, count: count)
        array.withUnsafeMutableBufferPointer { pointer in
            Device.Memory.assign(from: values.values, to: pointer, count: count)
        }
        return array
    }

    /// Creates a tensor with the shape and the values of a tensor on another device.
    ///
    /// The new tensor is not part of the compute graph of the source tensor, so no gradient flows back to the source.
    /// - Parameters:
    ///   - tensor: Tensor to copy.
    ///   - requiresGradient: Whether it is desired to compute gradients of the new tensor.
    init<Source: DeviceType>(_ tensor: Tensor<Element, Source>, requiresGradient: Bool = false) {
        let buffer = Device.Memory.allocateBuffer(withShape: tensor.shape, type: Element.self)
        let count = tensor.count
        if count > 0 {
            // A read of the source waits for the device that computes it, then the values go to the target in one copy.
            withUnsafeTemporaryAllocation(of: Element.self, capacity: count) { staging in
                Source.Memory.assign(from: tensor.values.values, to: staging, count: count)
                Device.Memory.assign(from: UnsafeBufferPointer(staging), to: buffer.values, count: count)
            }
        }
        self.init(using: buffer, context: nil)
        self.requiresGradient = requiresGradient
    }

    /// Returns a copy of the tensor on another device.
    ///
    /// The copy is not part of the compute graph of the tensor, so no gradient flows back to the tensor.
    /// - Parameter device: Device of the copy.
    func copied<Target: DeviceType>(to device: Target.Type) -> Tensor<Element, Target> {
        Tensor<Element, Target>(self)
    }
}

public extension Tensor {
    /// Indicates whether any element of the tensor is not a number.
    var containsNaN: Bool {
        elements.contains(where: \.isNaN)
    }

    /// Indicates whether all elements of the tensor are finite.
    var isFinite: Bool {
        let abs = detached().rectifiedLinear() + (-detached()).rectifiedLinear()
        return abs.reduceMax().item.isFinite
    }
}

// MARK: Tensor - Image conversion

#if canImport(CoreGraphics)
import CoreGraphics

// The image is drawn into a bitmap with 8 bits per channel, so that the layout of the pixels does not depend on the
// format of the image. Color images are drawn as RGB with an unused fourth byte, because CoreGraphics does not
// support RGB bitmaps with three bytes per pixel.

/// The layout of the 8-bit bitmap that an image is drawn into.
private struct ImageBitmapLayout {
    /// Number of channels of the tensor: 1 for gray images, 3 for color images
    let channels: Int

    /// Number of bytes of a pixel in the bitmap
    let bytesPerPixel: Int

    let colorSpace: CGColorSpace
    let bitmapInfo: UInt32

    init(for image: CGImage) {
        if image.colorSpace?.model == .monochrome {
            channels = 1
            bytesPerPixel = 1
            colorSpace = CGColorSpaceCreateDeviceGray()
            bitmapInfo = CGImageAlphaInfo.none.rawValue
        } else {
            channels = 3
            bytesPerPixel = 4
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.noneSkipLast.rawValue
        }
    }
}

public extension Tensor {
    /// Creates a tensor from the given CGImage.
    ///
    /// The tensor has the shape [channels, height, width], with 1 channel for gray images and 3 channels (red, green,
    /// blue) for all other images. The alpha channel is not included. A pixel value of 0 becomes `range.lowerBound`
    /// and a pixel value of 255 becomes `range.upperBound`.
    /// - Parameters:
    ///   - image: Image
    ///   - range: Range to normalize pixel values to
    init?(_ image: CGImage, normalizedTo range: ClosedRange<Element> = 0 ... 1) {
        let layout = ImageBitmapLayout(for: image)
        let width = image.width
        let height = image.height
        let bytesPerRow = width * layout.bytesPerPixel
        let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: Swift.max(height * bytesPerRow, 1))
        defer {
            pixels.deallocate()
        }
        guard let context = CGContext(
            data: pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: layout.colorSpace,
            bitmapInfo: layout.bitmapInfo,
        ) else {
            return nil
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let scale = (range.upperBound - range.lowerBound).doubleValue / 255
        let lowerBound = range.lowerBound.doubleValue
        var elements = [Element](repeating: 0, count: layout.channels * height * width)
        for channel in 0 ..< layout.channels {
            for row in 0 ..< height {
                let source = pixels + row * bytesPerRow + channel
                let rowOffset = (channel * height + row) * width
                for column in 0 ..< width {
                    elements[rowOffset + column] = Element(lowerBound + Double(source[column * layout.bytesPerPixel]) * scale)
                }
            }
        }
        self.init(elements, shape: [layout.channels, height, width])
    }

    /// Creates an image from a tensor with the shape [channels, height, width].
    ///
    /// The tensor must have 1 channel (gray), 3 channels (red, green, blue), or 4 channels (red, green, blue, alpha).
    /// The value `tensorRange.lowerBound` becomes a pixel value of 0 and `tensorRange.upperBound` becomes 255.
    /// Values outside of the range are clamped.
    /// - Parameter tensorRange: Range of the values of the tensor
    /// - Returns: The image, or nil when the tensor does not have a supported number of channels
    func cgImage(normalizeFrom tensorRange: ClosedRange<Element> = 0 ... 1) -> CGImage? {
        guard dim == 3 else {
            return nil
        }
        let channels = shape[0]
        let height = shape[1]
        let width = shape[2]
        let bytesPerRow = width * channels

        let colorSpace: CGColorSpace
        let bitmapInfo: UInt32
        switch channels {
        case 1:
            colorSpace = CGColorSpaceCreateDeviceGray()
            bitmapInfo = CGImageAlphaInfo.none.rawValue
        case 3:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.none.rawValue
        case 4:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.last.rawValue
        default:
            return nil
        }

        let elements = elements
        let lowerBound = tensorRange.lowerBound.doubleValue
        let scale = 255 / (tensorRange.upperBound - tensorRange.lowerBound).doubleValue
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        for channel in 0 ..< channels {
            for row in 0 ..< height {
                let rowOffset = (channel * height + row) * width
                for column in 0 ..< width {
                    let value = ((elements[rowOffset + column].doubleValue - lowerBound) * scale).rounded()
                    pixels[row * bytesPerRow + column * channels + channel] = value.isNaN ? 0 : UInt8(Swift.min(Swift.max(value, 0), 255))
                }
            }
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else {
            return nil
        }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8 * channels,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent,
        )
    }
}

#endif

#if canImport(Cocoa)
import Cocoa

public extension Tensor {
    /// Creates a tensor from the given NSImage
    /// - Parameters:
    ///   - image: Image
    ///   - range: Range to normalize pixel values to
    init?(_ image: NSImage, normalizedTo range: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        self.init(cgImage, normalizedTo: range)
    }
}

public extension NSImage {
    /// Creates a NSImage from the given tensor
    /// - Parameters:
    ///   - tensor: Tensor
    ///   - tensorRange: Range to normalize pixel values to
    convenience init?<Element, Device>(_ tensor: Tensor<Element, Device>, tensorRange: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = tensor.cgImage(normalizeFrom: tensorRange) else {
            return nil
        }
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}
#endif

#if canImport(UIKit)
import UIKit

public extension Tensor {
    /// Creates a tensor from the given UIImage
    /// - Parameters:
    ///   - image: Image
    ///   - range: Range to normalize pixel values to
    init?(_ image: UIImage, normalizedTo range: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = image.cgImage else {
            return nil
        }
        self.init(cgImage, normalizedTo: range)
    }
}

public extension UIImage {
    /// Creates a UIImage from the given tensor
    /// - Parameters:
    ///   - tensor: Tensor
    ///   - tensorRange: Range to normalize pixel values to
    convenience init?<Element, Device>(_ tensor: Tensor<Element, Device>, tensorRange: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = tensor.cgImage(normalizeFrom: tensorRange) else {
            return nil
        }
        self.init(cgImage: cgImage)
    }
}
#endif

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

//MARK: Tensor extensions

extension Tensor: CustomStringConvertible, CustomDebugStringConvertible {
    public var description: String {
        values.description
    }
    
    public var debugDescription: String {
        return """
        Tensor<\(Element.self), \(Device.self)>(
            \(values.description.replacingOccurrences(of: "\n", with: "\n    ")),
            context: \(self.context as Any? ?? "nil" as Any)
        )
        """
    }
}

extension Tensor: Equatable where Element: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        if lhs.shape != rhs.shape {
            return false
        }
        let d = (lhs - rhs)
        return (d * d).reduceSum().item == 0
    }
}

extension Tensor: ExpressibleByFloatLiteral {
    public init(floatLiteral value: Double) {
        self.init([Element.init(value)], shape: [])
    }
}

extension Tensor: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init([Element.init(value)], shape: [])
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

//MARK: Tensor - array conversion
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
            requiresGradient: requiresGradient
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
                v.first?.first?.first?.count ?? 0
            ],
            requiresGradient: requiresGradient
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
                v.first?.first?.first?.first?.count ?? 0
            ],
            requiresGradient: requiresGradient
        )
    }
}

//MARK: Tensor initialization

public extension Tensor where Element: RandomizableType {
    /// Creates a tensor and fills it with random values sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / shape[0])`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(xavierNormalWithShape shape: [Int], requiresGradient: Bool = false) {
        precondition(shape.count == 2, "Shape must be 2-dimensional")
        self.init(normalDistributedWithShape: shape, mean: 0, stdev: (2 / Element(shape[0])).sqrt(), requiresGradient: requiresGradient)
    }
    
    /// Creates a tensor and fills it with random values sampled from a normal distribution with mean 0 and standard deviation `sqrt(2 / shape[0])`.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(xavierNormalWithShape shape: Int..., requiresGradient: Bool = false) {
        self.init(xavierNormalWithShape: shape, requiresGradient: requiresGradient)
    }
    
    /// Creates a tensor and fills it with random values sampled from a normal distribution with the given mean and variance.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - mean: Mean of the normal distribution.
    ///   - stdev: Standard deviation of the normal distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(normalDistributedWithShape shape: [Int], mean: Element = 0, stdev: Element = 1, requiresGradient: Bool = false) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.fillNormal(self.values, mean: mean, stdev: stdev)
    }
    
    /// Creates a tensor and fills it with random values sampled from a normal distribution with the given mean and variance.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - mean: Mean of the normal distribution.
    ///   - stdev: Standard deviation of the normal distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(normalDistributedWithShape shape: Int..., mean: Element = 0, stdev: Element = 1, requiresGradient: Bool = false) {
        self.init(normalDistributedWithShape: shape, mean: mean, stdev: stdev, requiresGradient: requiresGradient)
    }
    
    /// Creates a tensor and fills it with random values sampled from a uniform distribution with the given minimum and maximum.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - min: Minimum value of the uniform distribution
    ///   - max: Maximum value of the uniform distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(uniformlyDistributedWithShape shape: [Int], min: Element = 0, max: Element = 1, requiresGradient: Bool = false) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.fill(self.values, a: min, b: max)
    }
    
    /// Creates a tensor and fills it with random values sampled from a uniform distribution with the given minimum and maximum.
    /// - Parameters:
    ///   - shape: Shape of the tensor, must be two dimensional
    ///   - min: Minimum value of the uniform distribution
    ///   - max: Maximum value of the uniform distribution
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    init(uniformlyDistributedWithShape shape: Int..., min: Element = 0, max: Element = 1, requiresGradient: Bool = false) {
        self.init(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient)
    }
}

public extension Tensor {
    init(bernoulliDistributedWithShape shape: [Int], probability: Float, requiresGradient: Bool = false) {
        self.init(repeating: 0, shape: shape, requiresGradient: requiresGradient)
        Random.bernoulli(values, p: probability)
    }
    
    init(bernoulliDistributedWithShape shape: Int..., probability: Float, requiresGradient: Bool = false) {
        self.init(bernoulliDistributedWithShape: shape, probability: probability, requiresGradient: requiresGradient)
    }
}

extension Tensor: Codable where Element: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        requiresGradient = try container.decode(Bool.self, forKey: .requiresGradient)
        shape = try container.decode([Int].self, forKey: .shape)
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        handle = TensorHandle(values: buffer.values)
        let data = try container.decode(Data.self, forKey: .data)
        data.withUnsafeBytes { bytes in
            Device.Memory.assign(from: bytes.bindMemory(to: Element.self), to: buffer.values, count: count)
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        Device.Memory.assign(from: values.values, to: buffer, count: count)
        let data = Data(buffer: buffer)
        
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
}

public extension Tensor {
    /// Indicates whether any element of the tensor is not a number.
    var containsNaN: Bool {
        elements.contains(where: {$0.isNaN})
    }
    
    
    /// Indicates whether all elements of the tensor are finite.
    var isFinite: Bool {
        let abs = self.detached().rectifiedLinear() + (-self.detached()).rectifiedLinear()
        return abs.reduceMax().item.isFinite
    }
}

//MARK: Tensor - Image conversion
#if canImport(CoreGraphics)
import CoreGraphics

private func copy<Element: NumericType>(from image: CGImage, to buffer: UnsafeMutableBufferPointer<Element>, normalizeTo range: ClosedRange<Element> = 0 ... 1) -> Bool {
    let byteCount = image.height * image.bytesPerRow
    let data = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
    defer {
        data.deallocate()
    }
    guard let ctx = CGContext(
        data: data,
        width: image.width,
        height: image.height,
        bitsPerComponent: image.bitsPerComponent,
        bytesPerRow: image.bytesPerRow,
        space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: image.bitmapInfo.rawValue
    ) else {
        return false
    }
    
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    ctx.flush()
    
    let shape = [
        (image.colorSpace ?? CGColorSpaceCreateDeviceRGB()).numberOfComponents,
        image.height,
        image.width
    ]
    let strides = CPU.Memory.strides(from: shape)
    
    let pixels = data.assumingMemoryBound(to: UInt8.self)
    
    for idx in iterate(shape) {
        let (ch, row, col) = (idx[0], idx[1], idx[2])
        let val = pixels[col * image.bytesPerRow + row * image.bitsPerPixel / 8 + ch]
        buffer[ch * strides[0] + row * strides[1] + col] = Element(val) / ((range.upperBound - range.lowerBound) * 255)
    }
    return true
}

public extension Tensor {
    /// Creates a tensor from the given CGImage
    /// - Parameters:
    ///   - image: Image
    ///   - range: Range to normalize pixel values to
    init?(_ image: CGImage, normalizedTo range: ClosedRange<Element> = 0 ... 1) {
        let shape = [
            (image.colorSpace ?? CGColorSpaceCreateDeviceRGB()).numberOfComponents,
            image.height,
            image.width
        ]
        let imgBuffer = CPU.Memory.allocateBuffer(withShape: shape, type: Element.self)
        defer {
            CPU.Memory.free(imgBuffer)
        }
        guard copy(from: image, to: imgBuffer.values.pointer) else {
            return nil
        }
        let buffer = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Memory.assign(from: imgBuffer.immutable, to: buffer.values, count: buffer.count)
        self.init(using: buffer, context: nil)
    }
    
    func cgImage(normalizeFrom tensorRange: ClosedRange<Element> = 0 ... 1) -> CGImage? {
        let tensor = self
        
        let pixels = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: tensor.count)
        defer {
            pixels.deallocate()
        }
        
        let width = tensor.shape[2]
        let height = tensor.shape[1]
        let bytesPerRow = tensor.shape[2] * tensor.shape[0]
        let bytesPerPixel = tensor.shape[0]
        
        let colorSpace: CGColorSpace
        let bitmapInfo: UInt32
        switch bytesPerPixel {
        case 1:
            colorSpace = CGColorSpaceCreateDeviceGray()
            bitmapInfo = 0
        case 3:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.none.rawValue
        case 4:
            colorSpace = CGColorSpaceCreateDeviceRGB()
            bitmapInfo = CGImageAlphaInfo.last.rawValue
        default:
            return nil
        }
        
        for chan in 0 ..< tensor.shape[0] {
            for row in 0 ..< tensor.shape[1] {
                for col in 0 ..< tensor.shape[2] {
                    let val = (tensor[chan, row, col].item - tensorRange.lowerBound) * (255 / (tensorRange.upperBound - tensorRange.lowerBound))
                    pixels[col * bytesPerRow + row * bytesPerPixel + chan] = UInt8(val)
                }
            }
        }
        
        guard let ctx = CGContext(
            data: UnsafeMutableRawPointer(pixels.baseAddress!),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
            ) else {
                return nil
        }
        return ctx.makeImage()
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

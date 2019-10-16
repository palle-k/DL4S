//
//  Ext.swift
//  DL4STests
//
//  Created by Palle Klewitz on 08.03.19.
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

#if canImport(CoreGraphics)
import CoreGraphics

public extension Tensor {
    convenience init?(_ image: CGImage, normalizeTo range: ClosedRange<Element> = 0 ... 1) {
        let cgImage = image
        
        let byteCount = cgImage.height * cgImage.bytesPerRow
        let data = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer {
            data.deallocate()
        }
        
        guard let ctx = CGContext(
            data: data,
            width: cgImage.width,
            height: cgImage.height,
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: cgImage.bytesPerRow,
            space: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: cgImage.bitmapInfo.rawValue
        ) else {
            return nil
        }
        
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
        ctx.flush()
        
        self.init(
            repeating: Element(0),
            shape: [
                (cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB()).numberOfComponents,
                cgImage.height,
                cgImage.width
            ]
        )
        
        let pixels = data.assumingMemoryBound(to: UInt8.self)
        
        for chan in 0 ..< shape[0] {
            for row in 0 ..< shape[1] {
                for col in 0 ..< shape[2] {
                    let val = pixels[col * cgImage.bytesPerRow + row * cgImage.bitsPerPixel / 8 + chan]
                    self[chan, row, col] = Tensor(Element(val) / (range.upperBound - range.lowerBound))
                }
            }
        }
    }
}

public extension Tensor {
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

#if os(macOS)
import AppKit

public extension Tensor {
    convenience init?(_ image: NSImage, normalizeTo range: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        
        self.init(cgImage, normalizeTo: range)
    }
}

public extension NSImage {
    convenience init?<Element, Device>(_ tensor: Tensor<Element, Device>, tensorRange: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = tensor.cgImage(normalizeFrom: tensorRange) else {
            return nil
        }
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}
#endif

#if os(iOS) || os(tvOS) || os(watchOS)
import UIKit

public extension Tensor {
    convenience init?(_ image: UIImage, normalizeTo range: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = image.cgImage else {
            return nil
        }
        self.init(cgImage, normalizeTo: range)
    }
}

public extension UIImage {
    convenience init?<Element, Device>(_ tensor: Tensor<Element, Device>, tensorRange: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = tensor.cgImage(normalizeFrom: tensorRange) else {
            return nil
        }
        self.init(cgImage: cgImage)
    }
}
#endif

extension Tensor: CustomStringConvertible, CustomDebugStringConvertible {
    private func generateDescription() -> String {
        
        if dim > 1 {
            let d = (0 ..< shape[0])
                .map {self[$0].generateDescription()}
                .joined(separator: ",\n")
                .replacingOccurrences(of: "\n", with: "\n ")
            return "[\(d)]"
        } else if let count = self.shape.first {
            let d = (0 ..< count)
                .map {self[$0].item.format(maxDecimals: 3)}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(item.format(maxDecimals: 3))"
        }
    }
    
    private func generateGradientDescription() -> String? {
        guard self.gradient != nil else {
            return nil
        }
        if dim > 1 {
            let d = (0 ..< shape[0])
                .map {self[$0].generateGradientDescription()!}
                .joined(separator: ",\n")
                .replacingOccurrences(of: "\n", with: "\n ")
            return "[\(d)]"
        } else if let count = self.shape.first {
            let d = (0 ..< count)
                .map {self[$0].gradientItem!.format(maxDecimals: 3)}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(gradientItem!.format(maxDecimals: 3))"
        }
    }
    
    public var description: String {
        return generateDescription()
    }
    
    public var gradientDescription: String? {
        return generateGradientDescription()
    }
    
    public var debugDescription: String {
        return """
        Tensor<\(Element.self)>(
        shape: \(self.shape)
        elements: \(generateDescription().replacingOccurrences(of: "\n", with: "\n\t")),
        gradient: \(generateGradientDescription()?.replacingOccurrences(of: "\n", with: "\n\t") ?? "not required")
        )
        """
    }
}

// Memory operation extensions
extension Tensor {
    func buffer(from indices: [Int?]) -> (Buffer<Element, Device>, Bool, [Int]) {
        return Device.Memory.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Int?], source: Buffer<Element, Device>, sourceShape: [Int]) {
        Device.Memory.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Int?]) -> (Buffer<Element, Device>, Bool, [Int])? {
        guard let gradient = self.gradient else {
            return nil
        }
        return Device.Memory.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Int?], source: Buffer<Element, Device>, sourceShape: [Int]) {
        guard let gradient = self.gradient else {
            return
        }
        Device.Memory.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
    
    func buffer(from indices: [Range<Int>?]) -> (Buffer<Element, Device>, Bool, [Int]) {
        return Device.Memory.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Range<Int>?], source: Buffer<Element, Device>, sourceShape: [Int]) {
        Device.Memory.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Range<Int>?]) -> (Buffer<Element, Device>, Bool, [Int])? {
        guard let gradient = self.gradient else {
            return nil
        }
        return Device.Memory.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Range<Int>?], source: Buffer<Element, Device>, sourceShape: [Int]) {
        guard let gradient = self.gradient else {
            return
        }
        Device.Memory.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
}

extension Tensor: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        // removed for performance reasons
        //hasher.combine(bytes: UnsafeRawBufferPointer(self.values))
        hasher.combine(self.values.pointee)
        if let gradient = self.gradient {
            //hasher.combine(bytes: UnsafeRawBufferPointer(gradient))
            hasher.combine(gradient.pointee)
        }
    }
    
    public static func == (lhs: Tensor<Element, Device>, rhs: Tensor<Element, Device>) -> Bool {
        return lhs.shape == rhs.shape && lhs.values == rhs.values && lhs.gradient == rhs.gradient
    }
}

public extension Tensor {
    static func diagonal(size: Int, value: Element) -> Tensor<Element, Device> {
        let matrix = Tensor<Element, Device>(repeating: 0, shape: size, size)
        Device.Engine.fill(value: value, result: matrix.values, stride: size + 1, count: matrix.count)
        return matrix
    }
}

public extension Tensor {
    var graph: String {
        guard let ctx = self.context else {
            return """
            digraph {
                node [label=\"\(self.tag ?? "Tensor")\" shape=box]
            }
            """
        }
        return ctx.graph
    }
}

extension Tensor {
    @_specialize(where Element == Float, Device == CPU)
    @inline(__always)
    static func operationOrder(from initialTensor: Tensor<Element, Device>) -> [Tensor<Element, Device>] {
        var stack: [(Tensor<Element, Device>, Int)] = []
        var sorting: [Tensor<Element, Device>] = []
        var visited: Set<Tensor<Element, Device>> = []
        
        stack.append((initialTensor, 0))
        
        while let (current, idx) = stack.last {
            if visited.contains(current) || !current.requiresGradient {
                stack.removeLast()
                continue
            }
            
            if let context = current.context, context.sourceTensors.indices ~= idx {
                stack.removeLast()
                stack.append((current, idx + 1))
                stack.append((context.sourceTensors[idx], 0))
            } else {
                visited.insert(current)
                sorting.append(current)
                stack.removeLast()
            }
        }
        
        return sorting
    }
    
    public func detachAll() {
        let sorted = Tensor.operationOrder(from: self)
        for tensor in sorted {
            tensor.context = nil
        }
    }
}

public extension Tensor where Element == Int32 {
    func toOneHot<Target>(dim: Int) -> Tensor<Target, Device> {
        let result = Tensor<Target, Device>(repeating: 0, shape: self.shape + [dim])
        
        for idx in iterate(self.shape) {
            let target = Int(self[idx].item)
            result[idx + [target]] = 1
        }
        
        return result
    }
}

public extension Tensor {
    func toLabels() -> Tensor<Int32, Device> {
        let result = Tensor<Int32, Device>(repeating: 0, shape: shape.dropLast())
        
        for idx in iterate(self.shape.dropLast()) {
            let slice = self[idx]
            let (arg, _) = Device.Engine.argmax(values: slice.values, count: slice.count)
            result[idx] = Tensor<Int32, Device>(Int32(arg))
        }
        
        return result
    }
}

extension Tensor {
    var shapedValues: ShapedBuffer<Element, Device> {
        return ShapedBuffer(values: self.values, shape: self.shape)
    }
    
    var shapedGradient: ShapedBuffer<Element, Device>? {
        if let gradient = self.gradient {
            return ShapedBuffer(values: gradient, shape: self.shape)
        } else {
            return nil
        }
    }
}

public extension Tensor {
    func copied<Destination>(to device: Destination.Type) -> Tensor<Element, Destination> {
        return Tensor<Element, Destination>(self)
    }
}

public extension Tensor {
    var flattenedArray: [Element] {
        let ramBuffer = CPU.Memory.allocateBuffer(withCapacity: self.count, type: Element.self)
        defer {
            CPU.Memory.free(ramBuffer)
        }
        Device.Memory.assign(from: self.values, to: ramBuffer.pointer, count: self.count)
        return Array(ramBuffer.immutable)
    }
    
    var flattenedGradientArray: [Element]? {
        guard let gradient = self.gradient else {
            return nil
        }
        let ramBuffer = CPU.Memory.allocateBuffer(withCapacity: self.count, type: Element.self)
        defer {
            CPU.Memory.free(ramBuffer)
        }
        Device.Memory.assign(from: gradient, to: ramBuffer.pointer, count: self.count)
        return Array(ramBuffer.immutable)
    }
}

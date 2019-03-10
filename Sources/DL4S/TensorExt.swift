//
//  Ext.swift
//  DL4STests
//
//  Created by Palle Klewitz on 08.03.19.
//

import Foundation
import AppKit

public extension Tensor {
    convenience init?(_ image: NSImage, normalizeTo range: ClosedRange<Element> = 0 ... 1) {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        
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

public extension NSImage {
    convenience init?<Element>(_ tensor: Tensor<Element>, tensorRange: ClosedRange<Element> = 0 ... 1) {
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
        guard let cgImage = ctx.makeImage() else {
            return nil
        }
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}

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
                .map {"\(self[$0].item)"}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(item)"
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
                .map {"\(self[$0].gradientItem!)"}
                .joined(separator: ", ")
            return "[\(d)]"
        } else {
            return "\(gradientItem!)"
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
    func buffer(from indices: [Int?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Int?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Int?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int])? {
        guard let gradient = self.gradient else {
            return nil
        }
        return MemoryOps.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Int?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        guard let gradient = self.gradient else {
            return
        }
        MemoryOps.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
    
    func buffer(from indices: [Range<Int>?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int]) {
        return MemoryOps.get(slice: indices, of: values, with: shape)
    }
    
    func setBuffer(at indices: [Range<Int>?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        MemoryOps.set(slice: indices, of: values, with: shape, from: source, with: sourceShape)
    }
    
    func gradient(from indices: [Range<Int>?]) -> (UnsafeMutableBufferPointer<Element>, Bool, [Int])? {
        guard let gradient = self.gradient else {
            return nil
        }
        return MemoryOps.get(slice: indices, of: gradient, with: shape)
    }
    
    func setGradient(at indices: [Range<Int>?], source: UnsafeBufferPointer<Element>, sourceShape: [Int]) {
        guard let gradient = self.gradient else {
            return
        }
        MemoryOps.set(slice: indices, of: gradient, with: shape, from: source, with: sourceShape)
    }
}

extension Tensor: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
        // removed for performance reasons
        //hasher.combine(bytes: UnsafeRawBufferPointer(self.values))
        hasher.combine(self.values.first)
        hasher.combine(self.values.last)
        if let gradient = self.gradient {
            //hasher.combine(bytes: UnsafeRawBufferPointer(gradient))
            hasher.combine(gradient.first)
            hasher.combine(gradient.last)
        }
    }
    
    public static func == (lhs: Tensor<Element>, rhs: Tensor<Element>) -> Bool {
        return lhs.shape == rhs.shape && lhs.values == rhs.values && lhs.gradient == rhs.gradient
    }
}

public extension Tensor {
    static func diagonal(size: Int, value: Element) -> Tensor<Element> {
        let matrix = Tensor<Element>(repeating: 0, shape: size, size)
        Element.fill(value: value, result: matrix.values, stride: size + 1, count: matrix.count)
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
    func sorted(sorting: inout [Tensor<Element>], visited: inout Set<Tensor<Element>>) {
        guard !visited.contains(self), self.requiresGradient else {
            return
        }
        
        for source in context?.sourceTensors ?? [] {
            source.sorted(sorting: &sorting, visited: &visited)
        }
        
        visited.insert(self)
        sorting.append(self)
    }
}

extension Tensor where Element == Int32 {
    func toOneHot(dim: Int) -> Tensor<Float> {
        let result = Tensor<Float>(repeating: 0, shape: self.shape + [dim])
        
        for idx in iterate(self.shape) {
            let target = Int(self[idx].item)
            result[idx + [target]] = 1
        }
        
        return result
    }
}

extension Tensor {
    func toLabels() -> Tensor<Int32> {
        let result = Tensor<Int32>(repeating: 0, shape: shape.dropLast())
        
        for idx in iterate(self.shape.dropLast()) {
            let slice = self[idx]
            let (arg, _) = Element.argmax(values: slice.values.immutable, count: slice.count)
            result[idx] = Tensor<Int32>(Int32(arg))
        }
        
        return result
    }
}

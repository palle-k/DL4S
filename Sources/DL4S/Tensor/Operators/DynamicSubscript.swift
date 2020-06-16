//
//  File.swift
//  
//
//  Created by Palle Klewitz on 15.06.20.
//  Copyright (c) 2020 - Palle Klewitz
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

public enum SubscriptElement<Device: DeviceType> {
    case single(Int)
    case range(Range<Int>)
    case span
    case indices(Tensor<Int32, Device>)
    case mask(Tensor<UInt8, Device>)
    
    func count(forSize size: Int) -> Int {
        switch self {
            
        case .single:
            return 1
        case .range(let r):
            return r.count
        case .span:
            return size
        case .indices(let indexTensor):
            return indexTensor.count
        case .mask(let mask):
            return Int(mask.reduceSum().item)
        }
    }
    
    var retainsAxis: Bool {
        switch self {
        case .single:
            return false
        default:
            return true
        }
    }
}

public protocol SubscriptElementProtocol {
    func get<Device: DeviceType>() -> SubscriptElement<Device>
}

extension Int: SubscriptElementProtocol {
    public func get<Device>() -> SubscriptElement<Device> where Device : DeviceType {
        return .single(self)
    }
}

extension Range: SubscriptElementProtocol where Bound == Int {
    public func get<Device>() -> SubscriptElement<Device> where Device : DeviceType {
        return .range(self)
    }
}

public protocol SubscriptScalar: NumericType {
    static func makeElement<Device: DeviceType>(using tensor: Tensor<Self, Device>) -> SubscriptElement<Device>
}

extension UInt8: SubscriptScalar {
    public static func makeElement<Device>(using tensor: Tensor<UInt8, Device>) -> SubscriptElement<Device> where Device : DeviceType {
        return .mask(tensor)
    }
}

extension Int32: SubscriptScalar {
    public static func makeElement<Device>(using tensor: Tensor<Int32, Device>) -> SubscriptElement<Device> where Device : DeviceType {
        return .indices(tensor)
    }
}

extension Tensor: SubscriptElementProtocol where Element: SubscriptScalar {
    public func get<Dev>() -> SubscriptElement<Dev> where Dev : DeviceType {
        if Dev.self == Device.self {
            return Element.makeElement(using: self as! Tensor<Element, Dev>)
        } else {
            // TODO: Copy to correct device (requires feature/arrayfire to be merged)
            fatalError("Invalid device for index tensor")
        }
    }
}

public extension Tensor {
    subscript(dynamicElements: [SubscriptElementProtocol?]) -> Self {
        get {
            let index = dynamicElements.map {$0?.get() ?? SubscriptElement<Device>.span}
            let resultShape = zip(index + Array(repeating: .span, count: self.dim - dynamicElements.count), self.shape)
                .filter { idx, _ in idx.retainsAxis }
                .map { idx, size in idx.count(forSize: size) }
            
            let resultBuffer = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
            Device.Engine.dynamicSubscriptRead(values: self.values, result: resultBuffer, index: index)
            
            var result = Tensor(using: resultBuffer, context: nil)
            
            if requiresGradient {
                let sourceShape = self.shape
                
                result.context = TensorContext(
                    tag: "index",
                    sources: [self],
                    backpropagate: [{ resultGrad in
                        var target = Tensor<Element, Device>(repeating: 0, shape: sourceShape)
                        target[dynamicElements] = resultGrad
                        return target
                    }]
                )
            }
            
            return result
        }
        
        set (newValue) {
            let index = dynamicElements.map {$0?.get() ?? SubscriptElement<Device>.span}
            
            if !requiresGradient && !newValue.requiresGradient {
                Device.Engine.dynamicSubscriptWrite(values: newValue.values, result: values, index: index)
                return
            }
            
            if !requiresGradient && newValue.requiresGradient {
                Device.Engine.dynamicSubscriptWrite(values: newValue.values, result: values, index: index)
                self.context = TensorContext(
                    tag: "index",
                    sources: [newValue],
                    backpropagate: [{ resultGrad in
                        resultGrad[dynamicElements]
                    }]
                )
            }
            
            let source = self
            
            let resultBuffer = Device.Memory.allocateBuffer(withShape: self.shape, type: Element.self)
            Device.Memory.assign(from: source.values.values, to: resultBuffer.values, count: self.count)
            Device.Engine.dynamicSubscriptWrite(values: newValue.values, result: resultBuffer, index: index)
            
            let sourceShape = source.shape
            
            let result = Tensor(
                using: resultBuffer,
                context: TensorContext(
                    tag: "index",
                    sources: [source, newValue],
                    backpropagate: [
                        { resultGrad in
                            var mask = Tensor<Element, Device>(repeating: 1, shape: sourceShape)
                            mask[dynamicElements] = 0
                            return mask * resultGrad
                        },
                        { resultGrad in
                            resultGrad[dynamicElements]
                        }
                    ]
                )
            )
            
            self = result
        }
    }
    
    subscript(dynamicElements: SubscriptElementProtocol...) -> Self {
        get {self[dynamicElements]}
        set {self[dynamicElements] = newValue}
    }
}

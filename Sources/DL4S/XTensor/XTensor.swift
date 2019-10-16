//
//  XTensor.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
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

struct XTensorContext<Element: NumericType, Device: DeviceType> {
    var tag: String?
    var sources: [XTensor<Element, Device>]
    var backpropagate: [(XTensor<Element, Device>) -> XTensor<Element, Device>]
    #if DEBUG
    var operationStack = OperationGroup.operationStack
    #endif
    
    init(tag: String?, sources: [XTensor<Element, Device>], backpropagate: [(XTensor<Element, Device>) -> XTensor<Element, Device>]) {
        self.tag = tag
        self.sources = sources
        self.backpropagate = backpropagate
    }
}

class XTensorHandle<Element, Device: DeviceType> {
    var values: Buffer<Element, Device>
    var parent: XTensorHandle<Element, Device>?
    
    init(values: Buffer<Element, Device>, parent: XTensorHandle<Element, Device>? = nil) {
        self.values = values
        self.parent = parent
    }
    
    deinit {
        if parent == nil {
            Device.Memory.free(self.values)
        }
    }
}

public struct XTensor<Element: NumericType, Device: DeviceType> {
    var handle: XTensorHandle<Element, Device>
    var context: XTensorContext<Element, Device>? = nil
    
    /// Identifies the tensor during backpropagation.
    ///
    /// The id will vary across different runs and different tensors with the same values.
    let backpropID: UInt64 = UInt64.random(in: 0 ... UInt64.max)
    
    public let shape: [Int]
    public var requiresGradient: Bool
    
    #if DEBUG
    public var tag: String? = nil
    #endif
    
    var values: ShapedBuffer<Element, Device> {
        ShapedBuffer(values: handle.values, shape: shape)
    }
    
    public var count: Int {
        values.count
    }
    
    public var dim: Int {
        shape.count
    }
    
    public init(repeating value: Element, shape: Int..., requiresGradient: Bool = false) {
        self.init(repeating: value, shape: shape, requiresGradient: requiresGradient)
    }
    
    public init(repeating value: Element, shape: [Int], requiresGradient: Bool = false) {
        let values = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.fill(value: value, result: values.values, count: values.count)
        handle = XTensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = shape
    }
    
    public init(_ v: [Element], requiresGradient: Bool = false) {
        let values = Device.Memory.allocateBuffer(withShape: [v.count], type: Element.self)
        v.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: values.values, count: v.count)
        }
        handle = XTensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = [v.count]
    }
    
    public init(_ v: [Element], shape: [Int], requiresGradient: Bool = false) {
        precondition(v.count == shape.reduce(1, *), "Number of elements must match number of elements specified by shape")
        
        let values = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        v.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: values.values, count: v.count)
        }
        handle = XTensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = shape
    }
    
    init(using values: ShapedBuffer<Element, Device>, context: XTensorContext<Element, Device>?) {
        handle = XTensorHandle(values: values.values)
        self.context = context
        self.requiresGradient = context != nil
        self.shape = values.shape
    }
    
    init(handle: XTensorHandle<Element, Device>, shape: [Int], context: XTensorContext<Element, Device>?) {
        self.handle = handle
        self.context = context
        self.requiresGradient = context != nil
        self.shape = shape
    }
    
    @_specialize(where Element == Float, Device == CPU)
    @inline(__always)
    static func operationOrder(from initialTensor: XTensor<Element, Device>) -> [XTensor<Element, Device>] {
        var stack: [(XTensor<Element, Device>, Int)] = []
        var sorting: [XTensor<Element, Device>] = []
        var visited: Set<UInt64> = []
        
        stack.append((initialTensor, 0))
        
        while let (current, idx) = stack.last {
            if visited.contains(current.backpropID) {
                stack.removeLast()
                continue
            }
            
            if let context = current.context, context.sources.indices ~= idx {
                stack.removeLast()
                stack.append((current, idx + 1))
                stack.append((context.sources[idx], 0))
            } else {
                visited.insert(current.backpropID)
                sorting.append(current)
                stack.removeLast()
            }
        }
        
        return sorting
    }
    
    
    /// Performs backpropagation and returns the gradients for the given tensors
    /// - Parameters:
    ///   - tensors: Tensors to differentiate for
    ///   - retainGraph: Whether to store the graph for the backwards pass. If enabled, higher order gradients can be computed.
    public func gradients(of tensors: [XTensor<Element, Device>], retainBackwardsGraph retainGraph: Bool = false) -> [XTensor<Element, Device>] {
        OperationGroup.push("Backpropagate")
        
        let result = self
        let operationOrder = XTensor.operationOrder(from: result)
        
        var grads: [UInt64: XTensor<Element, Device>] = [
            result.backpropID: XTensor(repeating: 1, shape: result.shape)
        ]
        
        for tensor in operationOrder.reversed() {
            guard let grad = grads[tensor.backpropID] else {
                continue
            }
            guard let ctx = tensor.context else {
                continue
            }
            for (src, fn) in zip(ctx.sources, ctx.backpropagate) {
                guard src.requiresGradient else {
                    continue
                }
                
                let srcGrad: XTensor<Element, Device>
                if retainGraph {
                    srcGrad = fn(grad)
                } else {
                    srcGrad = fn(grad).detached()
                }
                
                assert(srcGrad.shape == src.shape)
                
                if let existingGrad = grads[src.backpropID] {
                    grads[src.backpropID] = existingGrad + srcGrad
                } else {
                    grads[src.backpropID] = srcGrad
                }
            }
        }
        
        let targetGrads = tensors.map {
            grads[$0.backpropID] ?? XTensor(repeating: 0, shape: $0.shape)
        }
        
        #if DEBUG
        if tensors.contains(where: {
            !grads.keys.contains($0.backpropID)
        }) {
            print("[WARNING]: No gradient given for tensor.")
        }
        #endif
        
        OperationGroup.pop()
        return targetGrads
    }
    
    mutating func ensureOwnership() {
        if isKnownUniquelyReferenced(&handle) {
            return
        }
        
        let replacementHandle = XTensorHandle(values:
            Device.Memory.allocateBuffer(withShape: shape, type: Element.self).values
        )
        Device.Memory.assign(from: handle.values, to: replacementHandle.values, count: count)
        self.handle = replacementHandle
    }
    
    public mutating func discardContext() {
        self.context = nil
    }
    
    public func detached() -> XTensor<Element, Device> {
        XTensor(handle: handle, shape: shape, context: nil)
    }
}

//
//  Tensor.swift
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


final class TensorHandle<Element, Device: DeviceType> {
    var values: Buffer<Element, Device>
    var parent: TensorHandle<Element, Device>?
    
    init(values: Buffer<Element, Device>, parent: TensorHandle<Element, Device>? = nil) {
        self.values = values
        self.parent = parent
    }
    
    deinit {
        if parent == nil {
            Device.Memory.free(self.values)
        }
    }
}

//MARK: Core Tensor functionality

/// A tensor is an n-dimensional array of numbers with a given shape.
public struct Tensor<Element: NumericType, Device: DeviceType> {
    var handle: TensorHandle<Element, Device>
    var context: TensorContext<Element, Device>? = nil
    
    /// Identifies the tensor during backpropagation.
    ///
    /// The id will vary across different runs and different tensors with the same values.
    let backpropID: UInt64 = UInt64.random(in: 0 ... UInt64.max)
    
    /// Shape of the tensor.
    ///
    /// A tensor with an empty shape is a scalar.
    /// When shape.count == 1, the tensor is a vector.
    /// When shape.count == 2, the tensor is a matrix, etc.
    public let shape: [Int]
    
    /// Whether the compute graph of operations originating from this tensor should be captured.
    /// If the compute graph is captured, the resources associated with this tensor are only released
    /// after all tensors that have been derived from this tensor are released.
    ///
    /// All tensors derived from gradient requiring tensors will also require a gradient.
    ///
    /// To compute a gradient, use the `gradients(of:)` function.
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([1,2,3,4,5], requiresGradient: true)
    /// let result = a * a * a // [1, 8, 27, 64, 125]
    ///
    /// let grads = result.gradients(of: [a])
    /// let ∇a = grads[0] // [3, 12, 27, 48, 75]
    /// ```
    ///
    /// To detach a tensor from the compute graph, use `tensor.detached()`.
    public var requiresGradient: Bool
    
    #if DEBUG
    /// Debug tag for the tensor. If you use `tensor.graph()` to visualize the compute graph, the tensor is labelled with the appropriate tag.
    public var tag: String? = nil
    #endif
    
    var values: ShapedBuffer<Element, Device> {
        ShapedBuffer(values: handle.values, shape: shape)
    }
    
    /// Number of elements in the tensor.
    public var count: Int {
        values.count
    }

    /// Dimensionality of the tensor. (0: scalar, 1: vector, 2: matrix, ...)
    public var dim: Int {
        shape.count
    }
    
    /// Creates a tensor with the given shape and fills it with `value`
    /// - Parameters:
    ///   - value: Value to fill tensor with
    ///   - shape: Shape of the tensor
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    public init(repeating value: Element, shape: Int..., requiresGradient: Bool = false) {
        self.init(repeating: value, shape: shape, requiresGradient: requiresGradient)
    }
    
    /// Creates a tensor with the given shape and fills it with `value`
    /// - Parameters:
    ///   - value: Value to fill tensor with
    ///   - shape: Shape of the tensor
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    public init(repeating value: Element, shape: [Int], requiresGradient: Bool = false) {
        let values = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.fill(value: value, result: values.values, count: values.count)
        handle = TensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = shape
    }
    
    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Value to fill tensor with
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    public init(_ v: [Element], requiresGradient: Bool = false) {
        let values = Device.Memory.allocateBuffer(withShape: [v.count], type: Element.self)
        v.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: values.values, count: v.count)
        }
        handle = TensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = [v.count]
    }
    
    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Value to fill tensor with
    ///   - shape: Shape of the tensor. The number of elements in `v` must be compatible with the shape.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    public init(_ v: [Element], shape: [Int], requiresGradient: Bool = false) {
        precondition(v.count == shape.reduce(1, *), "Number of elements must match number of elements specified by shape")
        
        let values = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        v.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: values.values, count: v.count)
        }
        handle = TensorHandle(values: values.values)
        self.requiresGradient = requiresGradient
        self.shape = shape
    }
    
    /// Creates a tensor with the given shape and fills it with the given array of elements
    /// - Parameters:
    ///   - v: Value to fill tensor with
    ///   - shape: Shape of the tensor. The number of elements in `v` must be compatible with the shape.
    ///   - requiresGradient: Whether it is desired to compute gradients of the tensor.
    public init(_ v: [Element], shape: Int..., requiresGradient: Bool = false) {
        self.init(v, shape: shape, requiresGradient: requiresGradient)
    }
    
    init(using values: ShapedBuffer<Element, Device>, context: TensorContext<Element, Device>?) {
        handle = TensorHandle(values: values.values)
        self.context = context
        self.requiresGradient = context != nil
        self.shape = values.shape
    }
    
    init(handle: TensorHandle<Element, Device>, shape: [Int], context: TensorContext<Element, Device>?) {
        self.handle = handle
        self.context = context
        self.requiresGradient = context != nil
        self.shape = shape
    }
    
    @_specialize(where Element == Float, Device == CPU)
    @inline(__always)
    static func operationOrder(from initialTensor: Self) -> [Self] {
        var stack: [(Tensor<Element, Device>, Int)] = []
        var sorting: [Tensor<Element, Device>] = []
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
    
    
    /// Performs backpropagation and returns the gradients for the given tensors.
    ///
    /// Tensors for which it is desired to compute gradients must have `requiresGradient` set to `true`.
    /// If the result is not differentiable with respect to an input tensor, a tensor of zeros will be returned.
    ///
    /// Example:
    /// ```
    /// let a = Tensor<Float, CPU>([1,2,3,4,5], requiresGradient: true)
    /// let result = a * a * a // [1, 8, 27, 64, 125]
    ///
    /// let grads = result.gradients(of: [a])
    /// let ∇a = grads[0] // [3, 12, 27, 48, 75]
    /// ```
    ///
    /// To detach a tensor from the compute graph, use `tensor.detached()`.
    ///
    /// If it is desired to compute second, third, etc. derivatives, the `retainBackwardsGraph` flag must be set to true.
    /// This will record the compute graph for the backpropagation operation. A second derivative can then be computed as the gradient of a gradient.
    /// If the flag is not set, the compute graph of the backwards operation will not be captured and the result is not differentiable to any variable.
    ///
    ///
    /// - Parameters:
    ///   - tensors: Tensors to differentiate for
    ///   - retainGraph: Whether to store the graph for the backwards pass. If enabled, higher order gradients can be computed.
    public func gradients(of tensors: [Self], retainBackwardsGraph retainGraph: Bool = false) -> [Self] {
        OperationGroup.push("Backpropagate")
        
        // self is the result of a function, which is differentiated with respect to the given tensors.
        let result = self
        
        // Build the gradient tape. Each operation is represented on the tape by its result.
        // It is not possible to just recursively walk through the compute graph, as the graph is not a tree.
        // Therefore, some variables may be included in multiple operations. Backpropagating through each path
        // could therefore lead to a combinatorial explosion.
        let operationOrder = Tensor.operationOrder(from: result)
        
        // The derivative of the function wrt. itself is 1.
        var grads: [UInt64: Tensor<Element, Device>] = [
            result.backpropID: Tensor(repeating: 1, shape: result.shape)
        ]
        grads.reserveCapacity(operationOrder.count)
        
        // Perform the actual backpropagation.
        for tensor in operationOrder.reversed() {
            guard let grad = grads[tensor.backpropID] else {
                continue
            }
            guard let ctx = tensor.context else {
                continue
            }
            // Add gradients of all tensors that directly influenced the values
            // of the current tensor to their respective accumulators
            for i in ctx.sources.indices {
                let src = ctx.sources[i]
                let fn = ctx.backpropagate[i]
                
                guard src.requiresGradient else {
                    continue
                }
                
                let srcGrad = fn(grad, grads[src.backpropID])
                #if DEBUG
                assert(srcGrad.shape == src.shape)
                #endif
                
                if retainGraph {
                    grads[src.backpropID] = srcGrad
                } else {
                    grads[src.backpropID] = srcGrad.detached()
                }
            }
        }
        
        let targetGrads = tensors.map {
            grads[$0.backpropID] ?? Tensor(repeating: 0, shape: $0.shape)
        }
        
        OperationGroup.pop()
        return targetGrads
    }
    
    mutating func ensureOwnership() {
        if isKnownUniquelyReferenced(&handle) && handle.parent == nil {
            return
        }
        
        let original = self
        let replacementHandle = TensorHandle(values:
            Device.Memory.allocateBuffer(withShape: shape, type: Element.self).values
        )
        Device.Memory.assign(from: handle.values, to: replacementHandle.values, count: count)
        
        self = Tensor(
            handle: replacementHandle,
            shape: shape,
            context: TensorContext(
                tag: "identity",
                sources: [original],
                backpropagate: [{$0}]
            )
        )
    }
    
    /// In-place detaches the tensor from the compute graph.
    public mutating func discardContext() {
        self.context = nil
    }
    
    /// Detaches the tensor from the compute graph. No gradients can be computed for the resulting tensor.
    public func detached() -> Self {
        Tensor(handle: handle, shape: shape, context: nil)
    }
}

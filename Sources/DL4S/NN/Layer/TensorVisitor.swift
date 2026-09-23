//
//  TensorVisitor.swift
//  DL4S
//
//  Created by Palle Klewitz on 21.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

/// The role of a tensor in a layer.
public enum TensorRole: Hashable, Sendable {
    /// A learned tensor. It is saved with the model and updated by an optimizer.
    case weight

    /// A tensor that is saved with the model but never trained.
    case frozen
}

/// Walks the tensors and sublayers of a layer.
///
/// Layers implement ``TensorContainer/visitTensors(_:)`` by calling the methods of the visitor once for every
/// stored tensor and sublayer. The visitor tracks the ``TensorPath`` of every tensor and calls a closure with
/// the tensor, its role, and its path. Because tensors are passed `inout`, the same traversal reads and
/// replaces tensors.
///
/// Positions in a ``Sequential`` block are numbered from 0. A nested `Sequential` is one element of the outer
/// block, so its tensors get paths such as `1.0.weights`, as in PyTorch.
///
/// Use ``LayerType/parameters``, ``LayerType/update(_:)``, ``LayerType/weightPaths``, and the other
/// extension methods of ``LayerType`` for common traversals. Create a visitor directly for custom traversals,
/// such as reading all tensors of a model for a checkpoint.
public struct TensorVisitor<Element: NumericType, Device: DeviceType> {
    /// A closure that the visitor calls for every tensor.
    public typealias TensorHandler = (_ tensor: inout Tensor<Element, Device>, _ role: TensorRole, _ path: TensorPath) -> Void

    /// A closure that the visitor calls for every sublayer with its path, before it walks the sublayer.
    public typealias LayerHandler = (_ layer: inout any TensorContainer, _ path: TensorPath) -> Void

    private let handleTensor: TensorHandler
    private let handleLayer: LayerHandler?

    private var path = TensorPath()

    // One counter per sequence scope. The root scope exists from the start so that a model that is itself a
    // Sequential numbers its elements from 0.
    private var sequenceIndices = [0]

    // Greater than 0 while the visitor is inside a sublayer that is marked as frozen, so that all weights of
    // that sublayer are reported as frozen.
    private var frozenDepth = 0

    /// Creates a visitor.
    /// - Parameters:
    ///   - tensors: Closure that is called for every tensor with the tensor, its role, and its path.
    ///   - layers: Closure that is called for every sublayer and its path before the visitor walks the sublayer.
    public init(tensors: @escaping TensorHandler, layers: LayerHandler? = nil) {
        handleTensor = tensors
        handleLayer = layers
    }

    /// Reports a learned tensor.
    /// - Parameters:
    ///   - tensor: The tensor.
    ///   - name: Name of the property that stores the tensor.
    public mutating func weight(_ tensor: inout Tensor<Element, Device>, named name: String) {
        report(&tensor, role: frozenDepth > 0 ? .frozen : .weight, segment: .name(name))
    }

    /// Reports a tensor that is saved with the model but never trained.
    /// - Parameters:
    ///   - tensor: The tensor.
    ///   - name: Name of the property that stores the tensor.
    public mutating func frozen(_ tensor: inout Tensor<Element, Device>, named name: String) {
        report(&tensor, role: .frozen, segment: .name(name))
    }

    /// Reports an array of learned tensors. The elements get the paths `name.0`, `name.1`, and so on.
    /// - Parameters:
    ///   - tensors: The tensors.
    ///   - name: Name of the property that stores the array.
    public mutating func weight(_ tensors: inout [Tensor<Element, Device>], named name: String) {
        report(&tensors, role: frozenDepth > 0 ? .frozen : .weight, named: name)
    }

    /// Reports an array of tensors that are saved but never trained. The elements get the paths `name.0`,
    /// `name.1`, and so on.
    /// - Parameters:
    ///   - tensors: The tensors.
    ///   - name: Name of the property that stores the array.
    public mutating func frozen(_ tensors: inout [Tensor<Element, Device>], named name: String) {
        report(&tensors, role: .frozen, named: name)
    }

    /// Walks a sublayer.
    /// - Parameters:
    ///   - layer: The sublayer.
    ///   - name: Name of the property that stores the sublayer.
    public mutating func sublayer<Layer: LayerType>(_ layer: inout Layer, named name: String) where Layer.Parameter == Element, Layer.Device == Device {
        path.append(.name(name))
        descend(into: &layer)
        path.removeLast()
    }

    /// Walks an optional sublayer. A `nil` value is skipped.
    /// - Parameters:
    ///   - layer: The sublayer.
    ///   - name: Name of the property that stores the sublayer.
    public mutating func sublayer<Layer: LayerType>(_ layer: inout Layer?, named name: String) where Layer.Parameter == Element, Layer.Device == Device {
        guard var unwrapped = layer else {
            return
        }
        // The optional is set to nil while its value is walked, so the value is not copied on write.
        layer = nil
        sublayer(&unwrapped, named: name)
        layer = unwrapped
    }

    /// Walks an array of sublayers. The elements get the paths `name.0`, `name.1`, and so on.
    /// - Parameters:
    ///   - layers: The sublayers.
    ///   - name: Name of the property that stores the array.
    public mutating func sublayer<Layer: LayerType>(_ layers: inout [Layer], named name: String) where Layer.Parameter == Element, Layer.Device == Device {
        path.append(.name(name))
        for index in layers.indices {
            path.append(.index(index))
            descend(into: &layers[index])
            path.removeLast()
        }
        path.removeLast()
    }

    /// Walks one element of a ``Sequential`` block.
    ///
    /// The element gets the next position in the current block as its path segment, whatever its type. A
    /// nested block is one element of the outer block and numbers its own elements from 0.
    ///
    /// The element must have the same `Parameter` and `Device` types as the visitor. ``LayerBuilder`` checks this
    /// when it builds a ``Sequential`` block.
    ///
    /// - Parameter layer: The element.
    public mutating func element<Layer: LayerType>(_ layer: inout Layer) {
        // Parameter packs cannot carry a same-type requirement per element, so the match between the element
        // and the visitor is checked here. The builder makes sure that the check passes.
        guard var visitor = self as? TensorVisitor<Layer.Parameter, Layer.Device> else {
            preconditionFailure("The element \(Layer.self) has the parameter type \(Layer.Parameter.self) on \(Layer.Device.self), but the sequence has \(Element.self) on \(Device.self).")
        }
        let index = visitor.sequenceIndices[visitor.sequenceIndices.count - 1]
        visitor.path.append(.index(index))
        visitor.descend(into: &layer)
        visitor.path.removeLast()
        visitor.sequenceIndices[visitor.sequenceIndices.count - 1] = index + 1
        // The cast succeeded above, so both types are the same and the cast back cannot fail.
        // swiftlint:disable:next force_cast
        self = visitor as! TensorVisitor<Element, Device>
    }

    private mutating func report(_ tensor: inout Tensor<Element, Device>, role: TensorRole, segment: TensorPath.Segment) {
        path.append(segment)
        handleTensor(&tensor, role, path)
        path.removeLast()
    }

    private mutating func report(_ tensors: inout [Tensor<Element, Device>], role: TensorRole, named name: String) {
        path.append(.name(name))
        for index in tensors.indices {
            report(&tensors[index], role: role, segment: .index(index))
        }
        path.removeLast()
    }

    // Opens a new sequence scope for the layer, so that a Sequential inside the layer numbers its elements from 0.
    private mutating func descend<Layer: LayerType>(into layer: inout Layer) where Layer.Parameter == Element, Layer.Device == Device {
        applyLayerHandler(to: &layer)
        sequenceIndices.append(0)
        layer.visitTensors(&self)
        sequenceIndices.removeLast()
    }

    private func applyLayerHandler<Layer: LayerType>(to layer: inout Layer) {
        guard let handleLayer else {
            return
        }
        var erased: any TensorContainer = layer
        handleLayer(&erased, path)
        guard let replaced = erased as? Layer else {
            preconditionFailure("A layer handler replaced a \(Layer.self) with a \(type(of: erased)).")
        }
        layer = replaced
    }
}

// Entry points for @Layer macro code.
// Overloads on the visitor resolve at compile time after the macro executes, as it does not have any type info available.
public extension TensorVisitor {
    /// Reports a stored tensor property.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - tensor: The tensor.
    ///   - name: Name of the property.
    ///   - role: Role of the tensor.
    mutating func stored(_ tensor: inout Tensor<Element, Device>, named name: String, role: TensorRole = .weight) {
        switch role {
        case .weight: weight(&tensor, named: name)
        case .frozen: frozen(&tensor, named: name)
        }
    }

    /// Reports a stored optional tensor property. A `nil` value is skipped.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - tensor: The tensor.
    ///   - name: Name of the property.
    ///   - role: Role of the tensor.
    mutating func stored(_ tensor: inout Tensor<Element, Device>?, named name: String, role: TensorRole = .weight) {
        guard var unwrapped = tensor else {
            return
        }
        tensor = nil
        stored(&unwrapped, named: name, role: role)
        tensor = unwrapped
    }

    /// Reports a stored array of tensors. The elements get the paths `name.0`, `name.1`, and so on.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - tensors: The tensors.
    ///   - name: Name of the property.
    ///   - role: Role of the tensors.
    mutating func stored(_ tensors: inout [Tensor<Element, Device>], named name: String, role: TensorRole = .weight) {
        switch role {
        case .weight: weight(&tensors, named: name)
        case .frozen: frozen(&tensors, named: name)
        }
    }

    /// Walks a stored sublayer. With the role ``TensorRole/frozen``, all weights of the sublayer are reported as frozen.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - layer: The sublayer.
    ///   - name: Name of the property.
    ///   - role: Role of the tensors in the sublayer.
    mutating func stored<Layer: LayerType>(_ layer: inout Layer, named name: String, role: TensorRole = .weight) where Layer.Parameter == Element, Layer.Device == Device {
        withRole(role) { visitor in
            visitor.sublayer(&layer, named: name)
        }
    }

    /// Walks a stored optional sublayer. A `nil` value is skipped.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - layer: The sublayer.
    ///   - name: Name of the property.
    ///   - role: Role of the tensors in the sublayer.
    mutating func stored<Layer: LayerType>(_ layer: inout Layer?, named name: String, role: TensorRole = .weight) where Layer.Parameter == Element, Layer.Device == Device {
        withRole(role) { visitor in
            visitor.sublayer(&layer, named: name)
        }
    }

    /// Walks a stored array of sublayers.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - layers: The sublayers.
    ///   - name: Name of the property.
    ///   - role: Role of the tensors in the sublayers.
    mutating func stored<Layer: LayerType>(_ layers: inout [Layer], named name: String, role: TensorRole = .weight) where Layer.Parameter == Element, Layer.Device == Device {
        withRole(role) { visitor in
            visitor.sublayer(&layers, named: name)
        }
    }

    /// Ignores a stored property that is not a tensor and not a layer.
    ///
    /// The ``Layer()`` macro calls this method for every stored property.
    /// - Parameters:
    ///   - value: The property value.
    ///   - name: Name of the property.
    ///   - role: Ignored.
    mutating func stored<Value>(_ value: inout Value, named name: String, role: TensorRole = .weight) {}

    private mutating func withRole(_ role: TensorRole, _ body: (inout TensorVisitor) -> Void) {
        if role == .frozen {
            frozenDepth += 1
        }
        body(&self)
        if role == .frozen {
            frozenDepth -= 1
        }
    }
}

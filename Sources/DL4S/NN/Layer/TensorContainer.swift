//
//  TensorContainer.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
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

/// A type that contains tensors and reports them to a ``TensorVisitor``, such as a layer or an optimizer.
///
/// Checkpoints and training use the traversal to read and write the tensors of a container.
public protocol TensorContainer<Parameter, Device> {
    /// Element type of the tensors
    associatedtype Parameter: NumericType

    /// Device type of the tensors
    associatedtype Device: DeviceType

    /// Reports the tensors and sublayers of the container to the visitor.
    ///
    /// Calls ``TensorVisitor/weight(_:named:)-(Tensor<Element,Device>,_)`` for every trained tensor, ``TensorVisitor/frozen(_:named:)-(Tensor<Element,Device>,_)`` for
    /// every tensor that is saved but not trained, such as moving averages and other statistics, and `sublayer(_:named:)` for every layer that is stored in a
    /// property. Tensors are reported in a fixed order.
    ///
    /// - Parameter visitor: Visitor that receives the tensors.
    mutating func visitTensors(_ visitor: inout TensorVisitor<Parameter, Device>)

    /// Adjust the structure to match the provided layout.
    ///
    /// This may create, replace, and remove tensors and sublayers.
    ///
    /// The method is called on a container before it is called on its child containers, so a
    /// container can create sublayers that then adopt their own part of the layout.
    /// Containers are not required to create all containers from the layout, only ones they need.
    ///
    /// The default implementation does nothing. Containers with a fixed structure, such as most layers, may not
    /// implement it.
    ///
    /// - Parameter layout: Paths and shapes of the tensors, relative to the path of the container.
    mutating func adoptLayout(_ layout: TensorLayout)
}

public extension TensorContainer {
    mutating func adoptLayout(_ layout: TensorLayout) {}

    /// The paths and shapes of all tensors that the container reports.
    var tensorLayout: TensorLayout {
        var shapes: [TensorPath: [Int]] = [:]
        var copy = self
        var visitor = TensorVisitor<Parameter, Device>(tensors: { tensor, _, path in
            shapes[path] = tensor.shape
        })
        copy.visitTensors(&visitor)
        return TensorLayout(shapes)
    }
}

extension TensorContainer {
    /// Calls ``adoptLayout(_:)`` on the container and then on every sublayer, with the part of the layout below
    /// the path of the sublayer.
    mutating func adoptLayoutRecursively(_ layout: TensorLayout) {
        adoptLayout(layout)
        var visitor = TensorVisitor<Parameter, Device>(tensors: { _, _, _ in }, layers: { container, path in
            container.adoptLayout(layout.scoped(to: path))
        })
        visitTensors(&visitor)
    }
}

/// The paths and shapes of a set of tensors.
///
/// Use ``children(of:)`` to get the tensors below a property, for example the optimizer state stored in an array.
public struct TensorLayout: Equatable, Sendable {
    /// One tensor of a layout.
    public struct Entry: Hashable, Sendable {
        /// The path of the tensor.
        public let path: TensorPath

        /// The shape of the tensor.
        public let shape: [Int]

        /// Creates an entry.
        /// - Parameters:
        ///   - path: The path of the tensor.
        ///   - shape: The shape of the tensor.
        public init(path: TensorPath, shape: [Int]) {
            self.path = path
            self.shape = shape
        }
    }

    private let shapes: [TensorPath: [Int]]
    private let scope: TensorPath

    /// Creates a layout.
    /// - Parameter shapes: The shape of every tensor, keyed by path.
    public init(_ shapes: [TensorPath: [Int]] = [:]) {
        self.shapes = shapes
        scope = TensorPath()
    }

    private init(shapes: [TensorPath: [Int]], scope: TensorPath) {
        self.shapes = shapes
        self.scope = scope
    }

    /// All tensors of the layout, sorted by path.
    public var entries: [Entry] {
        children(of: TensorPath())
    }

    /// Returns the shape of the tensor at a path.
    /// - Parameter path: The path of the tensor.
    /// - Returns: The shape, or `nil` when the layout has no tensor at the path.
    public subscript(path: TensorPath) -> [Int]? {
        shapes[TensorPath(scope.segments + path.segments)]
    }

    /// Returns the tensors below a path, with paths relative to it.
    ///
    /// For a layout with the tensors `firstMoments.0.weights` and `firstMoments.0.bias`, `children(of: "firstMoments")`
    /// returns the paths `0.bias` and `0.weights`.
    ///
    /// - Parameter path: The path of the parent.
    /// - Returns: The tensors below the path, sorted by path. A tensor at the path itself is not included.
    public func children(of path: TensorPath) -> [Entry] {
        let prefix = TensorPath(scope.segments + path.segments)
        return shapes
            .filter { $0.key.segments.count > prefix.segments.count && $0.key.starts(with: prefix) }
            .map { Entry(path: TensorPath(Array($0.key.segments.dropFirst(prefix.segments.count))), shape: $0.value) }
            .sorted { $0.path < $1.path }
    }

    /// Returns the part of the layout below a path, with paths relative to it.
    /// - Parameter path: The path of the new root.
    /// - Returns: A layout with the tensors below the path.
    public func scoped(to path: TensorPath) -> TensorLayout {
        TensorLayout(shapes: shapes, scope: TensorPath(scope.segments + path.segments))
    }

    public static func == (lhs: TensorLayout, rhs: TensorLayout) -> Bool {
        lhs.entries == rhs.entries
    }
}

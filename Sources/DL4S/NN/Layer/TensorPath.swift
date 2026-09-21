//
//  TensorPath.swift
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

/// The position of a tensor in the layer tree of a model.
///
/// A path has one component per level of the tree. A segment is the name of a stored property, or the index of
/// an element in a `Sequential` block or in an array of layers. The text form joins the segments with dots,
/// for example `encoder.blocks.3.Wq` or `0.weights`. Checkpoint files use the text form as the key of a tensor.
public struct TensorPath: Hashable, Sendable {
    /// One component of a tensor path.
    public enum Segment: Hashable, Sendable, CustomStringConvertible {
        /// The name of a stored property.
        case name(String)

        /// The position of an element in a `Sequential` block or in an array of layers.
        case index(Int)

        public var description: String {
            switch self {
            case let .name(name): name
            case let .index(index): String(index)
            }
        }
    }

    /// Segments of the path, from the root of the model to the tensor.
    public var segments: [Segment]

    /// Creates a path from its segments.
    /// - Parameter segments: Segments of the path, from the root of the model to the tensor.
    public init(_ segments: [Segment] = []) {
        self.segments = segments
    }

    /// Creates a path from its text form.
    ///
    /// The text form joins the segments with dots. A segment that consists of digits only is an index.
    ///
    /// - Parameter description: Text form of the path, for example `encoder.blocks.3.Wq`.
    public init(_ description: String) {
        segments = description.split(separator: ".", omittingEmptySubsequences: false).map { segment in
            if let index = Int(segment), !segment.isEmpty, segment.allSatisfy(\.isNumber) {
                .index(index)
            } else {
                .name(String(segment))
            }
        }
    }

    /// Returns the path with a segment added at the end.
    /// - Parameter segment: Segment to add.
    /// - Returns: The path to a child of the current position.
    public func appending(_ segment: Segment) -> TensorPath {
        TensorPath(segments + [segment])
    }

    /// Adds a segment at the end of the path.
    /// - Parameter segment: Segment to add.
    public mutating func append(_ segment: Segment) {
        segments.append(segment)
    }

    /// Removes the last segment of the path.
    public mutating func removeLast() {
        segments.removeLast()
    }
}

extension TensorPath: CustomStringConvertible {
    public var description: String {
        segments.map(\.description).joined(separator: ".")
    }
}

extension TensorPath: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self.init(value)
    }
}

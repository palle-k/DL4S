//
//  Graph.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
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

// MARK: Compute Graph Debugging

private extension String {
    func escaped() -> String {
        replacingOccurrences(of: "\"", with: "\\\"")
    }
}

struct Digraph: Hashable, Codable {
    struct Node: Hashable, Codable {
        var id: String
        var label: String?
        var shape: String = "box"
        var attributes: [String: String] = [:]
    }

    struct Edge: Hashable, Codable {
        var source: String
        var destination: String
        var label: String?
        var attributes: [String: String] = [:]
    }

    var id: String?
    var name: String?
    var nodes: Set<Node> = []
    var edges: Set<Edge> = []
    var subgraphs: [String: Digraph] = [:]

    mutating func addNode(id: String, label: String? = nil, shape: String = "box", attributes: [String: String] = [:]) {
        nodes.insert(Node(id: id, label: label, shape: shape, attributes: attributes))
    }

    mutating func addEdge(from source: String, to destination: String, label: String? = nil, attributes: [String: String] = [:]) {
        edges.insert(Edge(source: source, destination: destination, label: label, attributes: attributes))
    }

    mutating func join(with other: Digraph) {
        nodes.formUnion(other.nodes)
        edges.formUnion(other.edges)
        subgraphs.merge(other.subgraphs, uniquingKeysWith: { a, b in
            var m = a
            m.join(with: b)
            return m
        })
    }
}

extension Digraph.Node {
    var dot: String {
        let repr = "\(id)"

        var attrs = attributes
        if let label {
            attrs["label"] = "\"\(label.escaped())\""
        }
        attrs["shape"] = shape

        let pairs = attrs.map { "\($0.key)=\($0.value)" }
        return "\(repr) [\(pairs.joined(separator: " "))];"
    }
}

extension Digraph.Edge {
    var dot: String {
        let repr = "\(source) -> \(destination)"

        var attrs = attributes
        if let label {
            attrs["label"] = "\"\(label.escaped())\""
        }
        if attrs.isEmpty {
            return "\(repr);"
        } else {
            let pairs = attrs.map { "\($0.key)=\($0.value)" }
            return "\(repr) [\(pairs.joined(separator: " "))];"
        }
    }
}

extension Digraph: CustomStringConvertible {
    fileprivate func dot(type: String = "digraph", isRoot: Bool = false) -> String {
        """
        \(type) \(id ?? "") {
        \(isRoot ? "    graph [fontname=\"helvetica\" fontsize=10 color=\"#B0B0B0\"];\n" : "")\
        \(isRoot ? "    node [fontname=\"helvetica\" fontsize=10 margin=0.03 width=0.2 height=0 color=\"#A0A0A0\"];\n" : "")\
        \(isRoot ? "    edge [fontname=\"helvetica\" fontsize=8 arrowsize=0.5 color=\"#A0A0A0\" fontcolor=\"#A0A0A0\"];\n" : "")\
        \(isRoot ? "    splines=true;\n    ranksep=0.2;\n    nodesep=0.15;\n" : "")\
        \(nodes.isEmpty ? "" : "    \(nodes.map(\.dot).joined(separator: "\n    "))\n")\
        \(edges.isEmpty ? "" : "    \(edges.map(\.dot).joined(separator: "\n    "))\n")\
        \(subgraphs.isEmpty ? "" : "    \(subgraphs.values.map { $0.dot(type: "subgraph").split(separator: "\n").joined(separator: "\n    ") }.joined(separator: "\n    "))\n")\
        \(name.map { "    label=\"\($0.escaped())\";\n    labeljust=\"l\";\n" } ?? "")\
        }
        """
    }

    var description: String {
        dot(isRoot: true)
    }
}

#if DEBUG
/// One level of the operation stack that `OperationGroup.capture(named:_:)` records.
struct OperationGroupEntry: Sendable, Hashable {
    /// Distinguishes captures with the same name.
    let id: UInt64

    /// Name that `Tensor.graph()` shows for the group.
    let name: String
}
#endif

/// OperationGroup allows the grouping of operations in the compute graph.
/// This improves the readability, when displaying the compute graph using `result.graph()`.
/// It has no effect on the way that computations are performed. When optimization is enabled,
/// operation groups are not captured.
public enum OperationGroup {
    #if DEBUG
    /// Groups that enclose the current operation, outermost first.
    ///
    /// Task-local, so parallel tasks and threads have their own stack.
    @TaskLocal static var operationStack: [OperationGroupEntry] = []
    #endif

    /// Captures a group of operations that is displayed within a box in the compute graph, when using `result.graph()`.
    /// Only applicable for debug builds. In release builds, the operation closure is executed but otherwise, the operation group has no effect.
    /// - Parameters:
    ///   - name: Name of the operation
    ///   - operations: Operations to group
    @inline(__always)
    public static func capture<Output>(named name: String, _ operations: () -> Output) -> Output {
        #if DEBUG
        let entry = OperationGroupEntry(id: UniqueID.next(), name: name)
        return $operationStack.withValue(operationStack + [entry]) {
            operations()
        }
        #else
        return operations()
        #endif
    }
}

public extension Tensor {
    /// Identifier of the node of the tensor: the operation that created it, or the tensor when it has no context.
    private var nodeID: String {
        if let context {
            "\(backpropID)\(abs(context.tag.hashValue))"
        } else {
            "\(backpropID)"
        }
    }

    /// Adds the node of the tensor and the edges from its sources to a graph.
    private func addNode(to graph: inout Digraph) {
        guard let context else {
            let label: String

            #if DEBUG
            if shape == [] {
                label = "\(item)"
            } else if let tag {
                label = tag
            } else {
                label = "shape: \(shape)"
            }
            #else
            if shape == [] {
                label = "\(item)"
            } else {
                label = "shape: \(shape)"
            }
            #endif

            graph.addNode(id: nodeID, label: label, shape: "box", attributes: requiresGradient ? ["style": "filled", "fillcolor": "\"#99ccff\""] : [:])
            return
        }
        graph.addNode(id: nodeID, label: context.tag ?? "op", attributes: ["style": "rounded"])
        for source in context.sources {
            graph.addEdge(from: source.nodeID, to: nodeID)
        }
    }

    #if DEBUG
    /// Adds the operation groups that enclose the operation of the tensor to a graph.
    private func addGroups(to graph: inout Digraph) {
        guard let context, let last = context.operationStack.last else {
            return
        }
        var initial = Digraph(id: "cluster_\(last.id)", name: last.name, nodes: [Digraph.Node(id: nodeID)])
        for source in context.sources where source.context == nil {
            initial.addNode(id: source.nodeID, shape: "box")
        }

        let group = context.operationStack.dropLast().reversed().reduce(initial) { inner, item in
            Digraph(id: "cluster_\(item.id)", name: item.name, subgraphs: [inner.id!: inner])
        }
        var groupGraph = Digraph()
        groupGraph.subgraphs[group.id!] = group
        graph.join(with: groupGraph)
    }
    #endif

    /// Prints the compute graph, from which the tensor has been derived.
    ///
    /// The graph is in graphviz format and can be rendered with command line tools such as `dot`.
    ///
    /// **Note**: When running release builds, some information about the compute graph is discarded.
    /// To obtain a detailed compute graph, compile in debug mode.
    func graph() -> String {
        // The graph is traversed with a stack instead of recursion, because the graph of a long sequence, such as
        // the unrolled steps of a recurrent network, is deeper than the call stack allows.
        var graph = Digraph()
        var visited: Set<UInt64> = [backpropID]
        var pending = [self]
        while let tensor = pending.popLast() {
            tensor.addNode(to: &graph)
            #if DEBUG
            tensor.addGroups(to: &graph)
            #endif
            for source in tensor.context?.sources ?? [] where visited.insert(source.backpropID).inserted {
                pending.append(source)
            }
        }
        return graph.description
    }
}

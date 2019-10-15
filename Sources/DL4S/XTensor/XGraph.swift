//
//  XGraph.swift
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

extension String {
    fileprivate func escaped() -> String {
        replacingOccurrences(of: "\"", with: "\\\"")
    }
}

struct Digraph: Hashable, Codable {
    struct Node: Hashable, Codable {
        var id: String
        var label: String?
        var shape: String = "box"
        var attributes: Dictionary<String, String> = [:]
    }
    
    struct Edge: Hashable, Codable {
        var source: String
        var destination: String
        var label: String?
        var attributes: Dictionary<String, String> = [:]
    }
    
    var id: String? = nil
    var name: String? = nil
    var nodes: Set<Node> = []
    var edges: Set<Edge> = []
    var subgraphs: [String: Digraph] = [:]
    
    mutating func addNode(id: String, label: String?, shape: String = "box", attributes: Dictionary<String, String> = [:]) {
        nodes.insert(Node(id: id, label: label, shape: shape, attributes: attributes))
    }
    
    mutating func addEdge(from source: String, to destination: String, label: String? = nil, attributes: Dictionary<String, String> = [:]) {
        edges.insert(Edge(source: source, destination: destination, label: label, attributes: attributes))
    }
    
    mutating func join(with other: Digraph) {
        self.nodes.formUnion(other.nodes)
        self.edges.formUnion(other.edges)
        self.subgraphs.merge(other.subgraphs, uniquingKeysWith: { a, b in
            var m = a
            m.join(with: b)
            return m
        })
    }
}

extension Digraph.Node {
    var dot: String {
        var repr = "\(id) ["
        
        if let label = self.label {
            repr += "label=\"\(label.escaped())\" "
        }
        repr += "shape=\(shape) "
        
        for (key, value) in attributes {
            repr += "\(key)=\(value) "
        }
        
        repr += "]"
        return repr
    }
}

extension Digraph.Edge {
    var dot: String {
        var repr = "\(source) -> \(destination) ["
        
        repr += "label=\"\((label ?? "").escaped())\" "
        
        for (key, value) in attributes {
            repr += "\(key)=\(value) "
        }
        
        repr += "]"
        return repr
    }
}

extension Digraph: CustomStringConvertible {
    fileprivate func dot(type: String = "digraph") -> String {
        """
        \(type) \(id ?? "") {
            \(nodes.map {$0.dot}.joined(separator: "\n\t"))
            \(edges.map {$0.dot}.joined(separator: "\n\t"))
            \(subgraphs.values.map {$0.dot(type: "subgraph").split(separator: "\n").joined(separator: "\n\t")}.joined(separator: "\n\t"))
            label="\(name?.escaped() ?? "")";
            labeljust="l";
        }
        """
    }
    
    var description: String {
        return dot()
    }
}


enum OperationGroup {
    static private(set) var operationStack: [(id: UInt64, name: String)] = []
    
    @inline(__always)
    static func push(_ name: @autoclosure () -> String) {
        #if DEBUG
        operationStack.append((id: UInt64.random(in: 0 ... UInt64.max), name: name()))
        #endif
    }
    
    @inline(__always)
    static func pop() {
        #if DEBUG
        operationStack.removeLast()
        #endif
    }
    
    @inline(__always)
    static func capture<Output>(named name: String, _ operations: () -> Output) -> Output {
        push(name)
        let result = operations()
        pop()
        return result
    }
}


public extension XTensor {
    private func follow(visited: inout Set<UInt64>) -> Digraph {
        guard !visited.contains(self.backpropID) else {
            return Digraph()
        }
        
        visited.insert(self.backpropID)
        
        var graph = Digraph()
        if let ctx = self.context {
            graph.addNode(id: "\(self.backpropID)\(abs(ctx.tag.hashValue))", label: ctx.tag ?? "op")
            
            for src in ctx.sources {
                if let srcCtx = src.context {
                    graph.addEdge(from: "\(src.backpropID)\(abs(srcCtx.tag.hashValue))", to: "\(self.backpropID)\(abs(ctx.tag.hashValue))", label: "\(src.shape)")
                } else {
                    graph.addEdge(from: "\(src.backpropID)", to: "\(self.backpropID)\(abs(ctx.tag.hashValue))", label: "\(src.shape)")
                }
                graph.join(with: src.follow(visited: &visited))
            }
        } else {
            let label: String
            
            #if DEBUG
            if shape == [] {
                label = "\(item)"
            } else if let tag = self.tag {
                label = tag
            } else {
                label = ""
            }
            #else
            if shape == [] {
                label = "\(item)"
            } else {
                label = ""
            }
            #endif
            
            graph.addNode(id: "\(self.backpropID)", label: label, shape: "circle", attributes: requiresGradient ? ["style": "filled", "fillcolor": "\"#99ccff\""] : [:])
        }
        return graph
    }
    
    #if DEBUG
    private func followGroups(visited: inout Set<UInt64>) -> Digraph {
        guard !visited.contains(self.backpropID) else {
            return Digraph()
        }
        visited.insert(backpropID)
        
        guard let ctx = self.context else {
            return Digraph()
        }
        
        let opID = "\(self.backpropID)\(abs(ctx.tag.hashValue))"

        var ctxGraph = Digraph()
        
        if let last = ctx.operationStack.last {
            let initial = Digraph(id: "cluster_\(last.id)", name: last.name, nodes: [Digraph.Node(id: opID)])
            let g = ctx.operationStack.dropLast().reversed().reduce(initial) { acc, item in
                Digraph(id: "cluster_\(item.id)", name: item.name, subgraphs: [acc.id!: acc])
            }
            ctxGraph.subgraphs[g.id!] = g
        }
        
        return ctx.sources.reduce(into: ctxGraph) { acc, src in
            acc.join(with: src.followGroups(visited: &visited))
        }
    }
    #endif
    
    func graph() -> String {
        var visited: Set<UInt64> = []
        var graph = follow(visited: &visited)
        #if DEBUG
        visited.removeAll(keepingCapacity: true)
        graph.join(with: followGroups(visited: &visited))
        #endif
        return graph.description
    }
}

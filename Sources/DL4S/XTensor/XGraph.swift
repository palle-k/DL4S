//
//  XGraph.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

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
    }
    
    struct Edge: Hashable, Codable {
        var source: String
        var destination: String
        var label: String?
    }
    
    var nodes: Set<Node> = []
    var edges: Set<Edge> = []
    
    mutating func addNode(id: String, label: String?, shape: String = "box") {
        nodes.insert(Node(id: id, label: label, shape: shape))
    }
    
    mutating func addEdge(from source: String, to destination: String, label: String? = nil) {
        edges.insert(Edge(source: source, destination: destination, label: label))
    }
    
    mutating func join(with other: Digraph) {
        self.nodes.formUnion(other.nodes)
        self.edges.formUnion(other.edges)
    }
}

extension Digraph.Node {
    var dot: String {
        var repr = "\(id) ["
        
        if let label = self.label {
            repr += "label=\"\(label.escaped())\" "
        }
        repr += "shape=\(shape)"
        
        repr += "]"
        return repr
    }
}

extension Digraph.Edge {
    var dot: String {
        return "\(source) -> \(destination) [label=\"\((label ?? "").escaped())\"]"
    }
}

extension Digraph: CustomStringConvertible {
    var description: String {
        return """
        digraph {
            \(nodes.map {$0.dot}.joined(separator: "\n\t"))
            \(edges.map {$0.dot}.joined(separator: "\n\t"))
        }
        """
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
            
            graph.addNode(id: "\(self.backpropID)", label: label, shape: "circle")
        }
        return graph
    }
    
    func graph() -> String {
        var visited: Set<UInt64> = []
        return follow(visited: &visited).description
    }
}

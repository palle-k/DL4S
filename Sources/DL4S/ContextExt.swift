//
//  ContextExt.swift
//  DL4S
//
//  Created by Palle Klewitz on 08.03.19.
//

import Foundation

private enum GraphNode<Element: NumericType> {
    case tensor(Tensor<Element>)
    case operation(AnyTensorOperation<Element>)
    
    var id: String {
        switch self {
        case .operation(let op):
            return op.id
        case .tensor(let t):
            return String(t.values.baseAddress?.hashValue ?? 0).replacingOccurrences(of: "-", with: "z")
        }
    }
    
    var label: String {
        switch self {
        case .operation(let op):
            return op.symbol
        case .tensor(let t):
            if let tag = t.tag {
                return tag
            } else if t.dim == 0 {
                return "\(t.item)"
            } else {
                return "Tensor"
            }
        }
    }
    
    var declaration: String {
        switch self {
        case .operation:
            return "\(id) [label=\"\(label)\" shape=box style=rounded]"
        case .tensor(let t):
            let bgColor: String
            if t.requiresGradient {
                bgColor = " style=filled fillcolor=\"#99ccff\""
            } else {
                bgColor = ""
            }
            return "\(id) [label=\"\(label)\" shape=box\(bgColor)]"
        }
    }
}

private struct Edge<Element: NumericType> {
    let source: GraphNode<Element>
    let destination: GraphNode<Element>
    
    var declaration: String {
        let srcId = source.id
        let dstId = destination.id
        
        return "\(srcId) -> \(dstId)"
    }
}

extension TensorOperation {
    var graph: String {
        var visited = Set<Tensor<Element>>()
        let (nodes, edges, selfNode) = collectGraph(visited: &visited)
        let nodeString = nodes.map {$0.declaration}.joined(separator: "\n\t")
        let edgeString = edges.map {$0.declaration}.joined(separator: "\n\t")
        return """
        digraph {
            \(nodeString)
            result [shape=point]
            \(edgeString)
            \(selfNode.id) -> result
        }
        """
    }
    
    fileprivate func collectGraph(visited: inout Set<Tensor<Element>>) -> ([GraphNode<Element>], [Edge<Element>], GraphNode<Element>) {
        let selfNode = GraphNode.operation(self.asAny())
        
        let sources = self.sourceTensors
        var nodes = [selfNode]
        var edges: [Edge<Element>] = []
        
//        var newVisited = visited.union(sources)
        
        for tensor in sources {
            if let ctx = tensor.context {
                if visited.contains(tensor) {
                    let ctxNode = GraphNode.operation(ctx)
                    edges.append(Edge(source: ctxNode, destination: selfNode))
                } else {
                    let (ctxNodes, ctxEdges, ctxNode) = ctx.collectGraph(visited: &visited)
                    visited.insert(tensor)
                    nodes.append(contentsOf: ctxNodes)
                    edges.append(contentsOf: ctxEdges)
                    edges.append(Edge(source: ctxNode, destination: selfNode))
                }
            } else {
                visited.insert(tensor)
                let node = GraphNode.tensor(tensor)
                let edge = Edge(source: node, destination: selfNode)
                nodes.append(node)
                edges.append(edge)
            }
        }
        
        return (nodes, edges, selfNode)
    }
    
    fileprivate var id: String {
        var hasher = Hasher()
        hasher.combine(sourceTensors)
        hasher.combine(symbol)
        let id = hasher.finalize()
        return String(id).replacingOccurrences(of: "-", with: "z")
    }
}

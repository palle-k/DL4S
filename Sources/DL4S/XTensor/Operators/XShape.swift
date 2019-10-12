//
//  XShape.swift
//  DL4S
//
//  Created by Palle Klewitz on 12.10.19.
//

import Foundation

public extension XTensor {
    func view(as shape: [Int]) -> XTensor<Element, Device> {
        precondition(shape.count(where: {$0 == -1}) <= 1, "The size of at most one dimension can be unknown (-1).")
        precondition(shape.allSatisfy {$0 >= -1}, "All dimensions must be greater than or equal to -1.")
        precondition(shape.contains(-1) || shape.reduce(1, *) == self.count, "Number of elements in result must be equal to number of elements in source")
        
        var shape = shape
        if let idx = shape.firstIndex(of: -1) {
            let remaining = count / shape.lazy.filter {$0 >= 0}.reduce(1, *)
            shape[idx] = remaining
        }
        
        return XTensor(
            handle: self.handle,
            shape: shape,
            context: requiresGradient ? XTensorContext(
                tag: "Reshape(\(shape))",
                sources: [self],
                backpropagate: [{ resultGradient in
                    resultGradient.view(as: self.shape)
                }]
            ) : nil
        )
    }
}

public extension XTensor {
    func permuted(to axisArangement: [Int]) -> XTensor<Element, Device> {
        precondition(axisArangement.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(axisArangement).count == dim, "Axis arangement must not contain duplicate axes")
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[axisArangement[i]] = shape[i]
        }
        
        let resultBuffer = Device.Memory.allocateBuffer(withShape: dstShape, type: Element.self)
        Device.Engine.permuteAxes(values: self.values, result: resultBuffer, arangement: axisArangement)
        
        return XTensor(
            using: resultBuffer,
            context: XTensorContext(
                tag: "Permute(\(axisArangement))",
                sources: [self],
                backpropagate: [{ resultGradient in
                    var invArangement = [Int](repeating: 0, count: self.dim)
                    for (i, j) in axisArangement.enumerated() {
                        invArangement[i] = j
                    }
                    return resultGradient.permuted(to: invArangement)
                }]
            )
        )
    }
    
    func transposed() -> XTensor<Element, Device> {
        permuted(to: [1, 0])
    }
}

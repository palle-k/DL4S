//
//  Permutation.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


fileprivate struct PermutationOperation<Element: NumericType>: UnaryTensorOperation {
    var symbol: String {
        return "permute"
    }
    
    var source: Tensor<Element>
    var axisArangement: [Int]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let srcGrad = source.gradient, let dstGrad = vector.gradient else {
            return
        }
        
        for index in iterate(source.shape) {
            var dstIdx = [Int](repeating: 0, count: source.dim)
            for i in dstIdx.indices {
                dstIdx[axisArangement[i]] = index[i]
            }
            
            let lsIdx = MemoryOps.linearIndex(from: index, shape: source.shape)
            let ldIdx = MemoryOps.linearIndex(from: dstIdx, shape: vector.shape)
            
            srcGrad[lsIdx] = dstGrad[ldIdx]
        }
    }
}


public extension Tensor {
    func permuted(to axisArangement: [Int]) -> Tensor<Element> {
        precondition(axisArangement.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(axisArangement).count == dim, "Axis arangement must not contain duplicate axes")
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[axisArangement[i]] = shape[i]
        }
        
        let result = Tensor<Element>(
            shape: dstShape,
            parent: nil,
            context: requiresGradient ? PermutationOperation(source: self, axisArangement: axisArangement).asAny() : nil
        )
        
        for index in iterate(shape) {
            var dstIdx = [Int](repeating: 0, count: dim)
            for i in dstIdx.indices {
                dstIdx[axisArangement[i]] = index[i]
            }
            
            let lsIdx = MemoryOps.linearIndex(from: index, shape: shape)
            let ldIdx = MemoryOps.linearIndex(from: dstIdx, shape: dstShape)
            
            result.values[ldIdx] = self.values[lsIdx]
        }
        
        return result
    }
    
    func permuted(to axisArangement: Int...) -> Tensor<Element> {
        return permuted(to: axisArangement)
    }
}

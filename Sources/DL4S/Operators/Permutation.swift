//
//  Permutation.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
//

import Foundation


fileprivate struct PermutationOperation<Element: NumericType, Device: DeviceType>: UnaryTensorOperation {
    var symbol: String {
        return "permute"
    }
    
    var source: Tensor<Element, Device>
    var axisArangement: [Int]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, Device>) {
        guard let srcGrad = source.gradient, let dstGrad = vector.gradient else {
            return
        }
        
        var invArangement = [Int](repeating: 0, count: vector.dim)
        
        for (i, j) in axisArangement.enumerated() {
            invArangement[i] = j
        }
        
        Device.Engine.permuteAxesAdd(input: dstGrad, arangement: invArangement, shape: vector.shape, add: srcGrad, destination: srcGrad)
    }
}


public extension Tensor {
    func permuted(to axisArangement: [Int]) -> Tensor<Element, Device> {
        precondition(axisArangement.count == dim, "Axis arangement must have dimensionality of source tensor")
        precondition(Set(axisArangement).count == dim, "Axis arangement must not contain duplicate axes")
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[axisArangement[i]] = shape[i]
        }
        
        let result = Tensor<Element, Device>(
            shape: dstShape,
            parent: nil,
            context: requiresGradient ? PermutationOperation(source: self, axisArangement: axisArangement).asAny() : nil
        )
        
        Device.Engine.permuteAxes(input: self.values, arangement: axisArangement, shape: self.shape, destination: result.values)
        
        return result
    }
    
    func permuted(to axisArangement: Int...) -> Tensor<Element, Device> {
        return permuted(to: axisArangement)
    }
}

//
//  ReductionArithmetic.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct SumContext<Element: NumericType>: UnaryTensorOperation {
    let source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradientItem, let sourceGradient = source.gradient else {
            return
        }
        Element.vsAdd(lhs: sourceGradient.immutable, rhs: vectorGradient, result: sourceGradient, count: source.count)
    }
    
    var symbol: String {
        return "Σ"
    }
}

public func sum<Element>(_ vector: Tensor<Element>, axis: Int? = nil) -> Tensor<Element> {
    if let axis = axis {
        var resultShape: [Int] = vector.shape
        resultShape.remove(at: axis)
        
        var result: Tensor<Element> = 0
        
        for i in 0 ..< vector.shape[axis] {
            var idx = Array(repeating: Int?.none, count: vector.dim)
            idx[axis] = i
            
            result = result + vector[idx]
        }
        
        return result
    } else {
        let result = Tensor<Element>(
            shape: [],
            parent: nil,
            context: vector.requiresGradient ? SumContext(source: vector).asAny() : nil
        )
        result.values[0] = Element.sum(val: vector.values.immutable, count: vector.count)
        return result
    }
}

private struct MaxContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    let maxI: Int
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradientItem else {
            return
        }
        sourceGradient[maxI] = sourceGradient[maxI] + vectorGradient
    }
    
    var symbol: String {
        return "max"
    }
}

private struct MaxAxisContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    let maxIdxs: [[Int]]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        
    }
    
    var symbol: String {
        return "max"
    }
}

public func max<Element>(_ vector: Tensor<Element>, axis: Int? = nil) -> Tensor<Element> {
    if let axis = axis {
        var resultShape: [Int] = vector.shape
        resultShape.remove(at: axis)
        
        var result: Tensor<Element> = Tensor(repeating: 0, shape: resultShape)
        
        for idx in iterate(resultShape) {
            var srcIdx: [Int?] = idx.map {$0}
            srcIdx.insert(nil, at: axis)
            
            let slice = vector[srcIdx]
            let (arg, max) = Element.argmax(values: slice.values.immutable, count: slice.count)
        }
        
        fatalError()
    } else {
        let (arg, max) = Element.argmax(values: vector.values.immutable, count: vector.count)
        
        let result = Tensor(max)
        result.context = vector.requiresGradient ? MaxContext(source: vector, maxI: arg).asAny() : nil
        return result
    }
}


public func argmax<Element>(_ vector: Tensor<Element>) -> Int {
    let (arg, _) = Element.argmax(values: vector.values.immutable, count: vector.count)
    return arg
}

private struct StackOperation<Element: NumericType>: TensorOperation {
    var sourceTensors: [Tensor<Element>]
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let vectorGradient = vector.gradient else {
            return
        }
        var offset = 0
        
        for src in sourceTensors {
            guard let srcGradient = src.gradient else {
                continue
            }
            let numElements = src.count
            Element.vAdd(lhs: srcGradient.immutable, rhs: vectorGradient.advanced(by: offset).immutable, result: srcGradient, count: numElements)
            offset += numElements
        }
    }
    
    var symbol: String {
        return "stack"
    }
}

public func stack<Element>(_ vectors: [Tensor<Element>]) -> Tensor<Element> {
    precondition(vectors.allSatisfy {$0.shape.dropFirst() == vectors[0].shape.dropFirst()}, "All vectors must have same shape except for first dimension")
    
    let resultShape = [vectors.reduce(0, {$0 + $1.shape[0]})] + Array(vectors[0].shape.dropFirst())
    
    let resultVector = Tensor<Element>(
        shape: resultShape,
        parent: nil,
        context: vectors.contains(where: {$0.requiresGradient}) ? StackOperation(sourceTensors: vectors).asAny() : nil
    )
    
    var offset = 0
    
    for vector in vectors {
        let numElements = vector.count
        resultVector.values.advanced(by: offset).assign(from: vector.values.immutable, count: numElements)
        offset += numElements
    }
    
    return resultVector
}

public func stack<Element>(_ vectors: Tensor<Element>...) -> Tensor<Element> {
    return stack(vectors)
}

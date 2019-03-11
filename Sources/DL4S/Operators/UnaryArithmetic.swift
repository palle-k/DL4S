//
//  UnaryOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct ExpContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        Element.vMA(lhs: vector.values.immutable, rhs: vectorGradient.immutable, add: sourceGradient.immutable, result: sourceGradient, count: source.count)
    }
    
    var symbol: String {
        return "exp"
    }
}

private struct LogContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
        
        Element.vDiv(lhs: vectorGradient.immutable, rhs: source.values.immutable, result: temp, count: source.count)
        Element.vAdd(lhs: temp.immutable, rhs: sourceGradient.immutable, result: sourceGradient, count: source.count)
        
        CPUAllocator.free(temp)
    }
    
    var symbol: String {
        return "log"
    }
}

private struct TanhContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
        
        Element.vSquare(values: vector.values.immutable, result: temp, count: source.count)
        Element.vNeg(val: temp.immutable, result: temp, count: source.count)
        Element.vsAdd(lhs: temp.immutable, rhs: 1, result: temp, count: source.count)
        Element.vMA(lhs: temp.immutable, rhs: vectorGradient.immutable, add: sourceGradient.immutable, result: sourceGradient, count: source.count)
        
        CPUAllocator.free(temp)
    }
    
    var symbol: String {
        return "tanh"
    }
}

private struct ReluContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        let temp1: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
        let temp2: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
        
        Element.fill(value: 0.5, result: temp1, count: vector.count)
        Element.fill(value: 0.5, result: temp2, count: vector.count)
        Element.copysign(values: temp1.immutable, signs: source.values.immutable, result: temp1, count: vector.count)
        
        // temp1[x] == 0 if source[x] <= 0 else temp1[x] == 1 (Relu mask)
        Element.vAdd(lhs: temp1.immutable, rhs: temp2.immutable, result: temp1, count: vector.count)
        Element.vMA(lhs: temp1.immutable, rhs: vectorGradient.immutable, add: sourceGradient.immutable, result: sourceGradient, count: source.count)
        
        CPUAllocator.free(temp1)
        CPUAllocator.free(temp2)
    }
    
    var symbol: String {
        return "relu"
    }
}

private struct SqrtContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let tmp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: vector.count)
        
        // 1/2*1/sqrt(source)
        Element.svDiv(lhs: 0.5, rhs: vector.values.immutable, result: tmp, count: vector.count)
        Element.vMA(lhs: tmp.immutable, rhs: vectorGradient.immutable, add: sourceGradient.immutable, result: sourceGradient, count: source.count)
        CPUAllocator.free(tmp)
    }
    
    var symbol: String {
        return "√"
    }
}


public func exp<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ExpContext(source: vector).asAny() : nil
    )
    
    Element.exp(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func log<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? LogContext(source: vector).asAny() : nil
    )
    
    Element.log(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func sqrt<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SqrtContext(source: vector).asAny() : nil
    )
    
    Element.sqrt(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}


public func log<Element>(_ vector: Tensor<Element>, base: Tensor<Element>) -> Tensor<Element> {
    return log(vector) / log(base)
}

public func tanh<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? TanhContext(source: vector).asAny() : nil
    )
    
    Element.tanh(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}

public func sigmoid<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    return 1 / (1 + exp(-vector))
}

public func relu<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ReluContext(source: vector).asAny() : nil
    )
    
    Element.relu(val: vector.values.immutable, result: result.values, count: result.count)
    
    return result
}


public func binaryCrossEntropy<Element: NumericType>(expected: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element> {
    let e = expected.view(as: -1)
    let a = actual.view(as: -1)
    
    let p1 = e * log(a)
    let p2 = (1 - e) * log(1 - a)
    return mean(-(p1 + p2))
}

public func categoricalCrossEntropy<Element: NumericType>(expected: Tensor<Int32>, actual: Tensor<Element>) -> Tensor<Element> {
    let expectedView = expected.view(as: -1)
    let actualView = actual.view(as: expectedView.shape[0], -1)
    
    var result = Tensor<Element>(0)
    
    for i in 0 ..< expectedView.shape[0] {
        // expected is not included in compute graph
        result = result - log(actualView[i, Int(expectedView[i].item)])
    }
    
    return result / Tensor<Element>(Element(expected.count))
}

public func meanSquaredError<Element>(expected: Tensor<Element>, actual: Tensor<Element>) -> Tensor<Element> {
    let diff = expected - actual
    let s = sum(diff * diff)
    return s  / Tensor(Element(expected.dim > 1 ? expected.shape[0] : 1))
}

public func mean<Element>(_ vector: Tensor<Element>, axis: Int? = nil) -> Tensor<Element> {
    let s = sum(vector, axis: axis)
    return s / Tensor(Element(axis.map {vector.shape[$0]} ?? vector.count))
}

public func variance<Element>(_ vector: Tensor<Element>, axis: Int? = nil) -> Tensor<Element> {
    let m = mean(vector, axis: axis)
    return mean(vector * vector, axis: axis) - m * m
}

public func l2loss<Element>(_ vector: Tensor<Element>, loss: Element) -> Tensor<Element> {
    return mean(vector * vector) * Tensor(loss)
}

public func l1loss<Element>(_ vector: Tensor<Element>, loss: Element) -> Tensor<Element> {
    // abs(x) = sqrt(x * x)
    return mean(sqrt(vector * vector)) * Tensor(loss)
}


private struct SoftmaxContext<Element: NumericType>: UnaryTensorOperation {
    var source: Tensor<Element>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        precondition(vector.dim == 2)
        
        let diag = Tensor<Element>.diagonal(size: vector.shape[1], value: 1)
        let stride = vector.strides[0]
        let temp: UnsafeMutableBufferPointer<Element> = CPUAllocator.allocate(count: stride)
        
        for i in 0 ..< vector.shape[0] {
            let filled = diag * vector[i]
            let outer = mmul(vector.view(as: -1, 1), vector.view(as: 1, -1))
            let jac = filled - outer
            
            Element.matMul(
                lhs: vectorGradient.advanced(by: i * stride).immutable,
                rhs: jac.values.immutable,
                result: temp,
                lhsRows: 1,
                lhsCols: jac.shape[0],
                rhsCols: jac.shape[1]
            )
            
            Element.vAdd(lhs: sourceGradient.advanced(by: stride * i).immutable, rhs: temp.immutable, result: sourceGradient.advanced(by: stride * i), count: stride)
        }
        
        CPUAllocator.free(temp)
    }
    
    var symbol: String {
        return "softmax"
    }
}

public func softmax<Element>(_ vector: Tensor<Element>) -> Tensor<Element> {
    let result = Tensor<Element>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SoftmaxContext(source: vector).asAny() : nil
    )
    
    let (_, max) = Element.argmax(values: vector.values.immutable, count: result.count)
    Element.vsAdd(lhs: vector.values.immutable, rhs: -max, result: result.values, count: result.count)
    Element.exp(val: result.values.immutable, result: result.values, count: result.count)
    
    let stride = vector.strides[0]
    for i in 0 ..< vector.shape[0] {
        let sum = Element.sum(val: result.values.advanced(by: stride * i).immutable, count: stride)
        Element.vsMul(lhs: result.values.advanced(by: stride * i).immutable, rhs: 1 / sum, result: result.values.advanced(by: stride * i), count: stride)
    }
    
    return result
}

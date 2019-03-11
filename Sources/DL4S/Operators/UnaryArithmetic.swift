//
//  UnaryOps.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


private struct ExpContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        DeviceType.EngineType.vMA(lhs: vector.values, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
    }
    
    var symbol: String {
        return "exp"
    }
}

private struct LogContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        DeviceType.EngineType.vDiv(lhs: vectorGradient, rhs: source.values, result: temp, count: source.count)
        DeviceType.EngineType.vAdd(lhs: temp, rhs: sourceGradient, result: sourceGradient, count: source.count)
        
        DeviceType.MemoryOperatorType.free(temp)
    }
    
    var symbol: String {
        return "log"
    }
}

private struct TanhContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let temp = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        DeviceType.EngineType.vSquare(values: vector.values, result: temp, count: source.count)
        DeviceType.EngineType.vNeg(val: temp, result: temp, count: source.count)
        DeviceType.EngineType.vsAdd(lhs: temp, rhs: 1, result: temp, count: source.count)
        DeviceType.EngineType.vMA(lhs: temp, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        
        DeviceType.MemoryOperatorType.free(temp)
    }
    
    var symbol: String {
        return "tanh"
    }
}

private struct ReluContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        
        let temp1 = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
        let temp2 = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        DeviceType.EngineType.fill(value: 0.5, result: temp1, count: vector.count)
        DeviceType.EngineType.fill(value: 0.5, result: temp2, count: vector.count)
        DeviceType.EngineType.copysign(values: temp1, signs: source.values, result: temp1, count: vector.count)
        
        // temp1[x] == 0 if source[x] <= 0 else temp1[x] == 1 (Relu mask)
        DeviceType.EngineType.vAdd(lhs: temp1, rhs: temp2, result: temp1, count: vector.count)
        DeviceType.EngineType.vMA(lhs: temp1, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        
        DeviceType.MemoryOperatorType.free(temp1)
        DeviceType.MemoryOperatorType.free(temp2)
    }
    
    var symbol: String {
        return "relu"
    }
}

private struct SqrtContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        let tmp = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: vector.count, type: Element.self)
        
        // 1/2*1/sqrt(source)
        DeviceType.EngineType.svDiv(lhs: 0.5, rhs: vector.values, result: tmp, count: vector.count)
        DeviceType.EngineType.vMA(lhs: tmp, rhs: vectorGradient, add: sourceGradient, result: sourceGradient, count: source.count)
        DeviceType.MemoryOperatorType.free(tmp)
    }
    
    var symbol: String {
        return "√"
    }
}


public func exp<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ExpContext(source: vector).asAny() : nil
    )
    
    DeviceType.EngineType.exp(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func log<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? LogContext(source: vector).asAny() : nil
    )
    
    DeviceType.EngineType.log(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func sqrt<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SqrtContext(source: vector).asAny() : nil
    )
    
    DeviceType.EngineType.sqrt(val: vector.values, result: result.values, count: result.count)
    
    return result
}


public func log<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>, base: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    return log(vector) / log(base)
}

public func tanh<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? TanhContext(source: vector).asAny() : nil
    )
    
    DeviceType.EngineType.tanh(val: vector.values, result: result.values, count: result.count)
    
    return result
}

public func sigmoid<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    return 1 / (1 + exp(-vector))
}

public func relu<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? ReluContext(source: vector).asAny() : nil
    )
    
    DeviceType.EngineType.relu(val: vector.values, result: result.values, count: result.count)
    
    return result
}


public func binaryCrossEntropy<Element: NumericType, DeviceType: Device>(expected: Tensor<Element, DeviceType>, actual: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let e = expected.view(as: -1)
    let a = actual.view(as: -1)
    
    let p1 = e * log(a)
    let p2 = (1 - e) * log(1 - a)
    return mean(-(p1 + p2))
}

public func categoricalCrossEntropy<Element: NumericType, DeviceType: Device>(expected: Tensor<Int32>, actual: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let expectedView = expected.view(as: -1)
    let actualView = actual.view(as: expectedView.shape[0], -1)
    
    var result = Tensor<Element, DeviceType>(0)
    
    for i in 0 ..< expectedView.shape[0] {
        // expected is not included in compute graph
        result = result - log(actualView[i, Int(expectedView[i].item)])
    }
    
    return result / Tensor<Element, DeviceType>(Element(expected.count))
}

public func meanSquaredError<Element, DeviceType>(expected: Tensor<Element, DeviceType>, actual: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let diff = expected - actual
    let s = sum(diff * diff)
    return s  / Tensor(Element(expected.dim > 1 ? expected.shape[0] : 1))
}

public func mean<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>, axis: Int? = nil) -> Tensor<Element, DeviceType> {
    let s = sum(vector, axis: axis)
    return s / Tensor(Element(axis.map {vector.shape[$0]} ?? vector.count))
}

public func variance<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>, axis: Int? = nil) -> Tensor<Element, DeviceType> {
    let m = mean(vector, axis: axis)
    return mean(vector * vector, axis: axis) - m * m
}

public func l2loss<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>, loss: Element) -> Tensor<Element, DeviceType> {
    return mean(vector * vector) * Tensor(loss)
}

public func l1loss<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>, loss: Element) -> Tensor<Element, DeviceType> {
    // abs(x) = sqrt(x * x)
    return mean(sqrt(vector * vector)) * Tensor(loss)
}


private struct SoftmaxContext<Element: NumericType, DeviceType: Device>: UnaryTensorOperation {
    var source: Tensor<Element, DeviceType>
    
    func fillSourceGradients(fromResultGradients vector: Tensor<Element, DeviceType>) {
        guard let sourceGradient = source.gradient, let vectorGradient = vector.gradient else {
            return
        }
        precondition(vector.dim == 2)
        
        let diag = Tensor<Element, DeviceType>.diagonal(size: vector.shape[1], value: 1)
        let stride = vector.strides[0]
        let temp = DeviceType.MemoryOperatorType.allocateBuffer(withCapacity: stride, type: Element.self)
        
        for i in 0 ..< vector.shape[0] {
            let filled = diag * vector[i]
            let outer = mmul(vector.view(as: -1, 1), vector.view(as: 1, -1))
            let jac = filled - outer
            
            DeviceType.EngineType.matMul(
                lhs: vectorGradient.advanced(by: i * stride),
                rhs: jac.values,
                result: temp,
                lhsRows: 1,
                lhsCols: jac.shape[0],
                rhsCols: jac.shape[1]
            )
            
            DeviceType.EngineType.vAdd(lhs: sourceGradient.advanced(by: stride * i), rhs: temp, result: sourceGradient.advanced(by: stride * i), count: stride)
        }
        
        DeviceType.MemoryOperatorType.free(temp)
    }
    
    var symbol: String {
        return "softmax"
    }
}

public func softmax<Element, DeviceType>(_ vector: Tensor<Element, DeviceType>) -> Tensor<Element, DeviceType> {
    let result = Tensor<Element, DeviceType>(
        shape: vector.shape,
        parent: nil,
        context: vector.requiresGradient ? SoftmaxContext(source: vector).asAny() : nil
    )
    
    let (_, max) = DeviceType.EngineType.argmax(values: vector.values, count: result.count)
    DeviceType.EngineType.vsAdd(lhs: vector.values, rhs: -max, result: result.values, count: result.count)
    DeviceType.EngineType.exp(val: result.values, result: result.values, count: result.count)
    
    let stride = vector.strides[0]
    for i in 0 ..< vector.shape[0] {
        let sum = DeviceType.EngineType.sum(val: result.values.advanced(by: stride * i), count: stride)
        DeviceType.EngineType.vsMul(lhs: result.values.advanced(by: stride * i), rhs: 1 / sum, result: result.values.advanced(by: stride * i), count: stride)
    }
    
    return result
}

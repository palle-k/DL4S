//
//  XTensor.swift
//  DL4S
//
//  Created by Palle Klewitz on 03.10.19.
//

import Foundation

struct XTensorContext<Element: NumericType, Device: DeviceType> {
    var sources: [XTensor<Element, Device>]
    var backpropagate: (XTensor<Element, Device>) -> [XTensor<Element, Device>]
}

fileprivate class XTensorHandle<Element, Device: DeviceType> {
    var values: ShapedBuffer<Element, Device>
    
    init(values: ShapedBuffer<Element, Device>) {
        self.values = values
    }
    
    deinit {
        Device.Memory.free(self.values)
    }
}

public struct XTensor<Element: NumericType, Device: DeviceType> {
    fileprivate var handle: XTensorHandle<Element, Device>
    
    var values: ShapedBuffer<Element, Device> {
        handle.values
    }
    var context: XTensorContext<Element, Device>? = nil
    
    public var shape: [Int] {
        values.shape
    }
    
    public var count: Int {
        values.count
    }
    
    public var dim: Int {
        values.dim
    }
    
    let backpropID: UInt64 = UInt64.random(in: 0 ... UInt64.max)
    
    public init(repeating value: Element, shape: Int...) {
        self.init(repeating: value, shape: shape)
    }
    
    public init(repeating value: Element, shape: [Int]) {
        let values = Device.Memory.allocateBuffer(withShape: shape, type: Element.self)
        Device.Engine.fill(value: value, result: values.values, count: values.count)
        handle = XTensorHandle(values: values)
    }
    
    public init(_ v: [Element]) {
        let values = Device.Memory.allocateBuffer(withShape: [v.count], type: Element.self)
        v.withUnsafeBufferPointer { ptr in
            Device.Memory.assign(from: ptr, to: values.values, count: v.count)
        }
        handle = XTensorHandle(values: values)
    }
    
    init(owning values: ShapedBuffer<Element, Device>, context: XTensorContext<Element, Device>) {
        handle = XTensorHandle(values: values)
        self.context = context
    }
    
    public static func + (lhs: XTensor<Element, Device>, rhs: XTensor<Element, Device>) -> XTensor<Element, Device> {
        let resultShape = shapeForBroadcastedOperands(lhs.shape, rhs.shape)
        let resultValues = Device.Memory.allocateBuffer(withShape: resultShape, type: Element.self)
        
        Device.Engine.broadcastAdd(lhs: lhs.values, rhs: rhs.values, result: resultValues)
        
        let resultContext = XTensorContext<Element, Device>(
            sources: [lhs, rhs],
            backpropagate: { vectorGradient in
                
                let lhsGradient = { () -> XTensor<Element, Device> in
                    let lhsGradient = XTensor<Element, Device>(repeating: 0, shape: lhs.shape)
                    
                    let tmp = Device.Memory.allocateBuffer(withShape: lhs.shape, type: Element.self)
                    defer {
                        Device.Memory.free(tmp)
                    }
                    
                    let lhsPadded = Array(repeating: 1, count: vectorGradient.dim - lhs.dim) + lhs.shape
                    let lhsReducedAxes = zip(lhsPadded, vectorGradient.shape).enumerated()
                        .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                    
                    var tmpReducedShape = lhsPadded
                    
                    for a in lhsReducedAxes.reversed() {
                        tmpReducedShape.remove(at: a)
                    }
                    
                    Device.Engine.reduceSum(values: vectorGradient.values, result: tmp.reshaped(to: tmpReducedShape), axes: lhsReducedAxes)
                    Device.Engine.broadcastAdd(lhs: tmp, rhs: lhsGradient.values, result: lhsGradient.values)
                    
                    return lhsGradient
                }()
                
                let rhsGradient = { () -> XTensor<Element, Device> in
                    let rhsGradient = XTensor<Element, Device>(repeating: 0, shape: lhs.shape)
                
                    let tmp = Device.Memory.allocateBuffer(withShape: rhs.shape, type: Element.self)
                    defer {
                        Device.Memory.free(tmp)
                    }
                    
                    let rhsPadded = Array(repeating: 1, count: vectorGradient.dim - rhs.dim) + rhs.shape
                    let rhsReducedAxes = zip(rhsPadded, vectorGradient.shape).enumerated()
                        .filter {$1.0 == 1 && $1.1 > 1}.map {$0.offset}
                    
                    var tmpReducedShape = rhsPadded
                    
                    for a in rhsReducedAxes.reversed() {
                        tmpReducedShape.remove(at: a)
                    }
                    
                    Device.Engine.reduceSum(values: vectorGradient.values, result: tmp.reshaped(to: tmpReducedShape), axes: rhsReducedAxes)
                    Device.Engine.broadcastAdd(lhs: tmp, rhs: rhsGradient.values, result: rhsGradient.values)
                    
                    return rhsGradient
                }()
                
                return [lhsGradient, rhsGradient]
            }
        )
        
        return XTensor(owning: resultValues, context: resultContext)
    }
    
    @_specialize(where Element == Float, Device == CPU)
    @inline(__always)
    static func operationOrder(from initialTensor: XTensor<Element, Device>) -> [XTensor<Element, Device>] {
        var stack: [(XTensor<Element, Device>, Int)] = []
        var sorting: [XTensor<Element, Device>] = []
        var visited: Set<UInt64> = []
        
        stack.append((initialTensor, 0))
        
        while let (current, idx) = stack.last {
            if visited.contains(current.backpropID) {
                stack.removeLast()
                continue
            }
            
            if let context = current.context, context.sources.indices ~= idx {
                stack.removeLast()
                stack.append((current, idx + 1))
                stack.append((context.sources[idx], 0))
            } else {
                visited.insert(current.backpropID)
                sorting.append(current)
                stack.removeLast()
            }
        }
        
        return sorting
    }
    
    func backpropagate(for tensors: [XTensor<Element, Device>]) -> [XTensor<Element, Device>] {
        let result = self
        let operationOrder = XTensor.operationOrder(from: result)
        
        var grads: [UInt64: XTensor<Element, Device>] = [
            result.backpropID: XTensor(repeating: 1, shape: result.shape)
        ]
        
        for tensor in operationOrder.reversed() {
            guard let grad = grads[tensor.backpropID] else {
                continue
            }
            guard let ctx = tensor.context else {
                continue
            }
            let sourceGrads = ctx.backpropagate(grad)
            grads.merge(zip(ctx.sources, sourceGrads).map {($0.backpropID, $1)}, uniquingKeysWith: +)
        }
        
        let targetGrads = tensors.map {
            grads[$0.backpropID] ?? XTensor(repeating: 0, shape: $0.shape)
        }
        
        return targetGrads
    }
}

extension XTensor: CustomStringConvertible {
    public var description: String {
        values.description
    }
}

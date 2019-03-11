//
//  Random.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


public protocol RandomizableType: NumericType {
    static func random(in range: ClosedRange<Self>) -> Self
}

extension Int32: RandomizableType {}
extension Float: RandomizableType {}
extension Double: RandomizableType {}


func randNormal<T: RandomizableType>(stdev: T, mean: T) -> (T, T) {
    let a = T.random(in: 0 ... 1)
    let b = T.random(in: 0 ... 1)
    
    let scale = (-2 * a.log()).sqrt() * stdev
    
    let twoPiB = 2 * 3.141592653589 * b
    
    return (scale * twoPiB.sin() + mean, scale * twoPiB.cos() + mean)
}

public enum Random {
    public static func fill<Element: RandomizableType, DeviceType: Device>(_ vector: Tensor<Element, DeviceType>, a: Element, b: Element) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        for i in 0 ..< vector.count {
            buffer[i] = Element.random(in: a ... b)
        }
        DeviceType.MemoryOperatorType.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }
    
    public static func fillNormal<Element: RandomizableType, DeviceType: Device>(_ vector: Tensor<Element, DeviceType>, mean: Element = 0, stdev: Element = 1) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        for i in stride(from: 0, to: vector.count, by: 2) {
            let (a, b) = randNormal(stdev: stdev, mean: mean)
            buffer[i] = a
            buffer[i+1] = b
        }
        
        if vector.count % 2 == 1 {
            let (a, _) = randNormal(stdev: stdev, mean: mean)
            buffer[vector.count-1] = a
        }
        DeviceType.MemoryOperatorType.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }
    
    public static func minibatch<Element: NumericType, DeviceType: Device>(from dataset: Tensor<Element, DeviceType>, count: Int) -> Tensor<Element, DeviceType> {
        let n = dataset.shape[0]
        
        let sampleShape = [1] + Array(dataset.shape.dropFirst())
        
        return stack(
            (0 ..< count)
                .map {_ in Int.random(in: 0 ..< n)}
                .map {dataset[$0].view(as: sampleShape)}
        )
    }
    
    public static func minibatch<E1: NumericType, E2: NumericType, D1: Device, D2: Device>(from dataset: Tensor<E1, D1>, labels: Tensor<E2, D2>, count: Int) -> (Tensor<E1, D1>, Tensor<E2, D2>) {
        let n = dataset.shape[0]
        
        // let sampleShape = [1] + Array(dataset.shape.dropFirst())
        // let labelShape = [1] + Array(labels.shape.dropFirst())
        
        let indices = (0 ..< count).map {_ in Int.random(in: 0 ..< n)}
        
        let randomSamples = stack(indices.map {dataset[$0].unsqueeze(at: 0)})
        let randomLabels = stack(indices.map {labels[$0].unsqueeze(at: 0)})
        
        return (randomSamples, randomLabels)
    }
    
    // TODO: Make this an in-place operation
    public static func bernoulli<Element, Device>(p: Float, shape: [Int]) -> Tensor<Element, Device> {
        let count = shape.reduce(1, *)
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0 ..< count {
            buffer[i] = Float.random(in: 0 ... 1) <= p ? 1 : 0
        }
        
        let result = Tensor<Element, Device>(repeating: 0, shape: shape)
        Device.MemoryOperatorType.assign(from: buffer.immutable, to: result.values, count: count)
        buffer.deallocate()
        return result
    }
}

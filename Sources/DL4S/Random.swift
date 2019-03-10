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
    public static func fill<Element: RandomizableType>(_ vector: Tensor<Element>, a: Element, b: Element) {
        for i in 0 ..< vector.count {
            vector.values[i] = Element.random(in: a ... b)
        }
    }
    
    public static func fillNormal<Element: RandomizableType>(_ vector: Tensor<Element>, mean: Element = 0, stdev: Element = 1) {
        for i in stride(from: 0, to: vector.count, by: 2) {
            let (a, b) = randNormal(stdev: stdev, mean: mean)
            vector.values[i] = a
            vector.values[i+1] = b
        }
        
        if vector.count % 2 == 1 {
            let (a, _) = randNormal(stdev: stdev, mean: mean)
            vector.values[vector.count-1] = a
        }
    }
    
    public static func minibatch<Element: NumericType>(from dataset: Tensor<Element>, count: Int) -> Tensor<Element> {
        let n = dataset.shape[0]
        
        let sampleShape = [1] + Array(dataset.shape.dropFirst())
        
        return stack(
            (0 ..< count)
                .map {_ in Int.random(in: 0 ..< n)}
                .map {dataset[$0].view(as: sampleShape)}
        )
    }
    
    public static func minibatch<E1: NumericType, E2: NumericType>(from dataset: Tensor<E1>, labels: Tensor<E2>, count: Int) -> (Tensor<E1>, Tensor<E2>) {
        let n = dataset.shape[0]
        
        // let sampleShape = [1] + Array(dataset.shape.dropFirst())
        // let labelShape = [1] + Array(labels.shape.dropFirst())
        
        let indices = (0 ..< count).map {_ in Int.random(in: 0 ..< n)}
        
        let randomSamples = stack(indices.map {dataset[$0].unsqueeze(at: 0)})
        let randomLabels = stack(indices.map {labels[$0].unsqueeze(at: 0)})
        
        return (randomSamples, randomLabels)
    }
    
    public static func bernoulli<Element>(p: Float, shape: [Int]) -> Tensor<Element> {
        let result = Tensor<Element>(repeating: 0, shape: shape)
        for i in 0 ..< result.count {
            result.values[i] = Float.random(in: 0 ... 1) <= p ? 1 : 0
        }
        return result
    }
}

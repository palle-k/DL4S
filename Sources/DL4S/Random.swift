//
//  Random.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//

import Foundation


public protocol RandomizableType: NumericType, Comparable {
    static func random(in range: ClosedRange<Self>) -> Self
}

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
    public static func fill<Element: RandomizableType>(_ vector: Vector<Element>, a: Element, b: Element) {
        for i in 0 ..< vector.count {
            vector.values[i] = Element.random(in: a ... b)
        }
    }
    
    public static func fillNormal<Element: RandomizableType>(_ vector: Vector<Element>, mean: Element = 0, stdev: Element = 1) {
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
}

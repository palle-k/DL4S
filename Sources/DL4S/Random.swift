//
//  Random.swift
//  DL4S
//
//  Created by Palle Klewitz on 27.02.19.
//  Copyright (c) 2019 - Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import Foundation


public protocol RandomizableType: NumericType {
    static func random<Generator: RandomNumberGenerator>(in range: ClosedRange<Self>, using rng: inout Generator) -> Self
}

extension Int32: RandomizableType {}
extension Float: RandomizableType {}
extension Double: RandomizableType {}

struct WyHash: RandomNumberGenerator {
    fileprivate static var shared = WyHash(seed: 0)
    
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0xa0761d6478bd642f
        let (high, low) = state.multipliedFullWidth(by: state ^ 0xe7037ed1a0b428db)
        return high ^ low
    }
    
    mutating func next<T>() -> T where T : FixedWidthInteger, T : UnsignedInteger {
        let n: UInt64 = next()
        return T(clamping: n)
    }
    
    mutating func next<T>(upperBound: T) -> T where T : FixedWidthInteger, T : UnsignedInteger {
        return next() % upperBound
    }
}

func randNormal<T: RandomizableType>(stdev: T, mean: T) -> (T, T) {
    let a = T.random(in: 0 ... 1, using: &WyHash.shared)
    let b = T.random(in: 0 ... 1, using: &WyHash.shared)
    
    let scale = (-2 * a.log()).sqrt() * stdev
    
    let twoPiB = 2 * 3.141592653589 * b
    
    let (x, y) = (scale * twoPiB.sin() + mean, scale * twoPiB.cos() + mean)
    
    if x.isFinite && !x.isNaN && y.isFinite && !y.isNaN {
        return (x, y)
    } else {
        return randNormal(stdev: stdev, mean: mean)
    }
}

public enum Random {
    @_specialize(where Element == Float, Device == CPU)
    @_specialize(where Element == Double, Device == CPU)
    @_specialize(where Element == Int32, Device == CPU)
    public static func fill<Element: RandomizableType, Device>(_ vector: ShapedBuffer<Element, Device>, a: Element, b: Element) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        let range = a ... b
        for i in 0 ..< vector.count {
            buffer[i] = Element.random(in: range, using: &WyHash.shared)
        }
        Device.Memory.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }
    
    @_specialize(where Element == Float, Device == CPU)
    @_specialize(where Element == Double, Device == CPU)
    @_specialize(where Element == Int32, Device == CPU)
    public static func fillNormal<Element: RandomizableType, Device>(_ vector: ShapedBuffer<Element, Device>, mean: Element = 0, stdev: Element = 1) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        for i in stride(from: 0, to: vector.count - 1, by: 2) {
            let (a, b) = randNormal(stdev: stdev, mean: mean)
            buffer[i] = a
            buffer[i+1] = b
        }
        
        if vector.count % 2 == 0 {
            let (a, _) = randNormal(stdev: stdev, mean: mean)
            buffer[vector.count-1] = a
        }
        Device.Memory.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }
    
    /// Samples a random minibatch of tensors from the given data set with shape [sample count, sample_dim1, ..., sample_dim_n]
    /// - Parameters:
    ///   - dataset: Dataset to sample a batch from
    ///   - count: Number of elements to include in the batch
    public static func minibatch<Element: NumericType, Device: DeviceType>(from dataset: Tensor<Element, Device>, count: Int) -> Tensor<Element, Device> {
        let n = dataset.shape[0]
        
        let sampleShape = [1] + Array(dataset.shape.dropFirst())
        
        return Tensor(
            stacking: (0 ..< count)
                .map {_ in Int.random(in: 0 ..< n)}
                .map {dataset[$0].view(as: sampleShape)},
            along: 0
        )
    }
    
    /// Samples a random minibatch of tensors from the given data set with shape [sample count, sample_dim1, ..., sample_dim_n] and their corresponding expected output vectors.
    /// - Parameters:
    ///   - dataset: Dataset to sample a batch from
    ///   - labels: Corresponding expected output vectors
    ///   - count: Number of elements to include in the batch
    public static func minibatch<E1: NumericType, E2: NumericType, D1: DeviceType, D2: DeviceType>(from dataset: Tensor<E1, D1>, labels: Tensor<E2, D2>, count: Int) -> (Tensor<E1, D1>, Tensor<E2, D2>) {
        let n = dataset.shape[0]
        
        let indices = (0 ..< count).map {_ in Int.random(in: 0 ..< n)}
        
        let randomSamples = Tensor(stacking: indices.map {dataset[$0].unsqueezed(at: 0)}, along: 0)
        let randomLabels = Tensor(stacking: indices.map {labels[$0].unsqueezed(at: 0)}, along: 0)
        
        return (randomSamples, randomLabels)
    }
    
    public static func bernoulli<Element: NumericType, Device>(_ values: ShapedBuffer<Element, Device>, p: Float) {
        let count = values.shape.reduce(1, *)
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0 ..< count {
            buffer[i] = Float.random(in: 0 ... 1) <= p ? 1 : 0
        }
        
        Device.Memory.assign(from: buffer.immutable, to: values.values, count: count)
        buffer.deallocate()
    }
}

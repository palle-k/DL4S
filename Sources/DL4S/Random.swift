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
extension Float: FloatingPointNumericType {}
extension Double: FloatingPointNumericType {}

/// A small, fast pseudo random number generator.
///
/// Use `init(seed:)` together with the `using:` overloads of random tensor and layer initializers for reproducible initialization.
///
/// ```
/// var generator = WyHash(seed: 42)
/// let weights = Tensor<Float, CPU>(xavierNormalWithShape: [16, 8], using: &generator)
/// let layer = Dense<Float, CPU>(inputSize: 16, outputSize: 8, using: &generator)
/// ```
public struct WyHash: RandomNumberGenerator, Sendable {
    private var state: UInt64

    /// Creates a generator that determistically generates randomness based on the provided seed.
    /// - Parameter seed: Initial state of the generator.
    public init(seed: UInt64) {
        state = seed
    }

    /// Creates a generator with a seed from the system random number generator.
    public init() {
        var systemGenerator = SystemRandomNumberGenerator()
        self.init(seed: systemGenerator.next())
    }

    public mutating func next() -> UInt64 {
        state &+= 0xA076_1D64_78BD_642F
        let (high, low) = state.multipliedFullWidth(by: state ^ 0xE703_7ED1_A0B4_28DB)
        return high ^ low
    }
}

/// Draws two independent values from a normal distribution with the Box-Muller transform.
func randNormal<T: FloatingPointNumericType, Generator: RandomNumberGenerator>(stdev: T, mean: T, using generator: inout Generator) -> (T, T) {
    let a = T.random(in: 0 ... 1, using: &generator)
    let b = T.random(in: 0 ... 1, using: &generator)

    let scale = (-2 * a.log()).sqrt() * stdev

    let twoPiB = T(2 * 3.141592653589) * b

    let (x, y) = (scale * twoPiB.sin() + mean, scale * twoPiB.cos() + mean)

    if x.isFinite, !x.isNaN, y.isFinite, !y.isNaN {
        return (x, y)
    } else {
        return randNormal(stdev: stdev, mean: mean, using: &generator)
    }
}

public enum Random {
    /// Fills the buffer with values from a uniform distribution in `a ... b`.
    /// - Parameters:
    ///   - vector: Buffer to fill
    ///   - a: Lower bound of the distribution
    ///   - b: Upper bound of the distribution
    ///   - generator: Random number generator that provides the values
    @_specialize(where Element == Float, Device == CPU, Generator == WyHash)
    @_specialize(where Element == Double, Device == CPU, Generator == WyHash)
    @_specialize(where Element == Int32, Device == CPU, Generator == WyHash)
    public static func fill<Element: RandomizableType, Device, Generator: RandomNumberGenerator>(_ vector: MutableShapedBuffer<Element, Device>, a: Element, b: Element, using generator: inout Generator) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        let range = a ... b
        for i in 0 ..< vector.count {
            buffer[i] = Element.random(in: range, using: &generator)
        }
        Device.Memory.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }

    /// Fills the buffer with values from a normal distribution.
    /// - Parameters:
    ///   - vector: Buffer to fill
    ///   - mean: Mean of the distribution
    ///   - stdev: Standard deviation of the distribution
    ///   - generator: Random number generator that provides the values
    @_specialize(where Element == Float, Device == CPU, Generator == WyHash)
    @_specialize(where Element == Double, Device == CPU, Generator == WyHash)
    public static func fillNormal<Element: FloatingPointNumericType, Device, Generator: RandomNumberGenerator>(_ vector: MutableShapedBuffer<Element, Device>, mean: Element = 0, stdev: Element = 1, using generator: inout Generator) {
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: vector.count)
        for i in stride(from: 0, to: vector.count - 1, by: 2) {
            let (a, b) = randNormal(stdev: stdev, mean: mean, using: &generator)
            buffer[i] = a
            buffer[i + 1] = b
        }

        // The Box-Muller transform yields pairs. An odd count needs one more value for the last element.
        if vector.count % 2 == 1 {
            let (a, _) = randNormal(stdev: stdev, mean: mean, using: &generator)
            buffer[vector.count - 1] = a
        }
        Device.Memory.assign(from: buffer.immutable, to: vector.values, count: vector.count)
        buffer.deallocate()
    }

    /// Fills the buffer with ones and zeros. Each element is 1 with probability `p`.
    ///
    /// The values come from a new generator with a random seed. Use `bernoulli(_:p:using:)` for reproducible values.
    /// - Parameters:
    ///   - values: Buffer to fill
    ///   - p: Probability of a 1
    @_specialize(where Element == Float, Device == CPU)
    @_specialize(where Element == Int32, Device == CPU)
    @_specialize(where Element == Double, Device == CPU)
    public static func bernoulli<Element: NumericType, Device>(_ values: MutableShapedBuffer<Element, Device>, p: Float) {
        var generator = WyHash()
        bernoulli(values, p: p, using: &generator)
    }

    /// Fills the buffer with ones and zeros. Each element is 1 with probability `p`.
    /// - Parameters:
    ///   - values: Buffer to fill
    ///   - p: Probability of a 1
    ///   - generator: Random number generator to draw from
    @_specialize(where Element == Float, Device == CPU, Generator == WyHash)
    @_specialize(where Element == Int32, Device == CPU, Generator == WyHash)
    @_specialize(where Element == Double, Device == CPU, Generator == WyHash)
    public static func bernoulli<Element: NumericType, Device, Generator: RandomNumberGenerator>(_ values: MutableShapedBuffer<Element, Device>, p: Float, using generator: inout Generator) {
        let count = values.shape.reduce(1, *)
        let buffer = UnsafeMutableBufferPointer<Element>.allocate(capacity: count)
        for i in 0 ..< count {
            buffer[i] = Float.random(in: 0 ... 1, using: &generator) <= p ? 1 : 0
        }

        Device.Memory.assign(from: buffer.immutable, to: values.values, count: count)
        buffer.deallocate()
    }
}

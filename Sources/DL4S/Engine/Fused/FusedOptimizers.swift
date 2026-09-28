//
//  FusedOptimizers.swift
//  DL4S
//
//  Created by Palle Klewitz on 23.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

// MARK: Default implementations

public extension FusedOperationsType {
    static func adamUpdate<N: NumericType>(
        parameter: ShapedBuffer<N, Device>,
        gradient: ShapedBuffer<N, Device>,
        firstMoment: MutableShapedBuffer<N, Device>,
        secondMoment: MutableShapedBuffer<N, Device>,
        secondMomentMax: MutableShapedBuffer<N, Device>?,
        learningRate: N,
        beta1: N,
        beta2: N,
        epsilon: N,
        beta1Power: N,
        beta2Power: N,
        result: MutableShapedBuffer<N, Device>,
    ) {
        let math = BufferMath<N, Device>()
        defer {
            math.release()
        }
        let scaled = math.temporary(gradient.shape)
        math.multiply(firstMoment, beta1, into: firstMoment)
        math.multiply(gradient, 1 - beta1, into: scaled)
        math.add(firstMoment, scaled, into: firstMoment)
        math.multiply(secondMoment, beta2, into: secondMoment)
        math.multiply(gradient, gradient, into: scaled)
        math.multiply(scaled, 1 - beta2, into: scaled)
        math.add(secondMoment, scaled, into: secondMoment)
        var normalizer = secondMoment
        if let secondMomentMax {
            math.maximum(secondMomentMax, secondMoment, into: secondMomentMax)
            normalizer = secondMomentMax
        }
        // parameter - learningRate * m / (sqrt(v) + epsilon) with the corrected moments m and v
        let divisor = scaled
        math.multiply(normalizer, 1 / (1 - beta2Power), into: divisor)
        math.sqrt(divisor, into: divisor)
        math.add(divisor, epsilon, into: divisor)
        math.multiply(firstMoment, learningRate / (1 - beta1Power), into: result)
        math.divide(result, divisor, into: result)
        math.subtract(parameter, result, into: result)
    }
}

//
//  FusedLayers.swift
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
    static func linear<N: NumericType>(input: ShapedBuffer<N, Device>, weights: ShapedBuffer<N, Device>, bias: ShapedBuffer<N, Device>?, result: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        math.multiplyMatrices(input, weights, into: result)
        if let bias {
            math.add(result, bias, into: result)
        }
    }

    static func linearBackward<N: NumericType>(
        input: ShapedBuffer<N, Device>,
        weights: ShapedBuffer<N, Device>,
        bias: ShapedBuffer<N, Device>?,
        outputGradient: ShapedBuffer<N, Device>,
        inputGradient: GradientBuffer<N, Device>?,
        weightGradient: GradientBuffer<N, Device>?,
        biasGradient: GradientBuffer<N, Device>?,
    ) {
        let math = BufferMath<N, Device>()
        // The products are added to the gradients with GEMMs, so a weight that is used in several steps needs no temporary gradient.
        math.writeProduct(outputGradient, weights, rhsTransposed: true, into: inputGradient)
        math.writeProduct(input, outputGradient, lhsTransposed: true, into: weightGradient)
        math.write(biasGradient) { db in
            math.sum(outputGradient, along: [0], into: db)
        }
    }

    static func dropout<N: FloatingPointNumericType>(input: ShapedBuffer<N, Device>, rate: Float, result: MutableShapedBuffer<N, Device>, mask: MutableShapedBuffer<N, Device>) {
        let math = BufferMath<N, Device>()
        // Without dropped elements, the mask needs no random numbers.
        guard rate > 0 else {
            Device.Engine.fill(value: 1, result: mask.values, count: mask.count)
            math.copy(input, into: result)
            return
        }
        Random.bernoulli(mask, p: 1 - rate)
        math.multiply(input, mask, into: result)
    }

    static func dropoutBackward<N: FloatingPointNumericType>(mask: ShapedBuffer<N, Device>, outputGradient: ShapedBuffer<N, Device>, inputGradient: GradientBuffer<N, Device>?) {
        let math = BufferMath<N, Device>()
        math.write(inputGradient) { dx in
            math.multiply(outputGradient, mask, into: dx)
        }
    }
}

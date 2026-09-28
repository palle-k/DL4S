//
//  CPUFusedDropout.swift
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

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func dropout<N: NumericType>(input: ShapedBuffer<N, CPU>, rate: Float, result: MutableShapedBuffer<N, CPU>, mask: MutableShapedBuffer<N, CPU>) {
        let (x, y, m) = (input.elementPointer, result.elementPointer, mask.elementPointer)
        let probability = Double(1 - rate)

        // An element is kept when a uniform 64-bit random number is below the threshold, which happens with the given probability.
        if probability >= 1 {
            CPUKernels.fill(m, with: 1, count: input.count)
            y.update(from: x, count: input.count)
            return
        }
        let threshold = probability <= 0 ? 0 : UInt64(Swift.min(probability, 1 - 0x1p-53) * 0x1p64)
        var generator = WyHash()
        for i in 0 ..< input.count {
            let factor: N = generator.next() < threshold ? 1 : 0
            let value = x[i]
            m[i] = factor
            y[i] = value * factor
        }
    }
}

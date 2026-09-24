//
//  CPUFusedActivation.swift
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

// Every kernel processes the elements in blocks of `CPUKernels.blockSize`. Transcendental functions go through the
// vectorized primitives into scratch buffers, and the remaining arithmetic is an element-wise loop over the block.
// The loops index pointers to the block with the loop variable only and load every operand before a comparison,
// so that the compiler vectorizes them: checked index arithmetic and loads in only one branch of a condition prevent that.
// Shapes that the kernels do not support, such as broadcast parameters, use the default implementations.

public extension CPUFusedOperations {
    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func tanhBackward<N: NumericType>(output: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard output.count > 0, output.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.tanhBackward(output: output, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: output.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let y = output.elementPointer
        let g = outputGradient.elementPointer
        for i in 0 ..< output.count {
            dx[i] = (1 - y[i] * y[i]) * g[i]
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func reluBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.reluBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        for i in 0 ..< input.count {
            let (value, gradient) = (x[i], g[i])
            dx[i] = value > 0 ? gradient : 0
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func sigmoid<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.sigmoid(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let half = N(0.5)
        CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { t in
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                // sigmoid(x) = tanh(x / 2) / 2 + 1 / 2 does not overflow for large magnitudes.
                for i in 0 ..< length {
                    t[i] = xb[i] * half
                }
                CPUKernels.tanh(t, into: yb, count: length)
                for i in 0 ..< length {
                    yb[i] = yb[i] * half + half
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func sigmoidBackward<N: NumericType>(output: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard output.count > 0, output.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.sigmoidBackward(output: output, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: output.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let y = output.elementPointer
        let g = outputGradient.elementPointer
        for i in 0 ..< output.count {
            dx[i] = y[i] * (1 - y[i]) * g[i]
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func leakyRelu<N: NumericType>(input: Tensor<N, CPU>, leakage: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0, leakage.count == 1, leakage.dim <= input.dim else {
            return DefaultFusedOperations<CPU>.leakyRelu(input: input, leakage: leakage)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let a = leakage.elementPointer[0]
        for i in 0 ..< input.count {
            let value = x[i]
            y[i] = value > 0 ? value : a * value
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func leakyReluBackward<N: NumericType>(input: Tensor<N, CPU>, leakage: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradients: inout (input: Tensor<N, CPU>?, leakage: Tensor<N, CPU>?)) {
        guard input.count > 0, leakage.count == 1, leakage.dim <= input.dim, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.leakyReluBackward(input: input, leakage: leakage, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        let a = leakage.elementPointer[0]

        var inputGradient: Tensor<N, CPU>?
        if input.requiresGradient {
            let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
            // The slope at 0 is 0, as for the rectified linear unit.
            for i in 0 ..< input.count {
                let (value, gradient) = (x[i], g[i])
                dx[i] = value > 0 ? gradient : (value < 0 ? a * gradient : 0)
            }
            inputGradient = result
        }
        var leakageGradient: Tensor<N, CPU>?
        if leakage.requiresGradient {
            let total = CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { t in
                var total: N = 0
                CPUKernels.forEachBlock(count: input.count) { offset, length in
                    let (xb, gb) = (x + offset, g + offset)
                    for i in 0 ..< length {
                        let (value, gradient) = (xb[i], gb[i])
                        t[i] = value < 0 ? value * gradient : 0
                    }
                    total += CPUKernels.sum(t, count: length)
                }
                return total
            }
            leakageGradient = Tensor([total], shape: leakage.shape)
        }
        let computed: (input: Tensor<N, CPU>?, leakage: Tensor<N, CPU>?) = (inputGradient, leakageGradient)
        Tensor.accumulate(computed.input, into: &gradients.input)
        Tensor.accumulate(computed.leakage, into: &gradients.leakage)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gelu<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.gelu(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let half = N(0.5)
        let scale = N(1.702 * 0.5)
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, s) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                for i in 0 ..< length {
                    t[i] = xb[i] * scale
                }
                CPUKernels.tanh(t, into: s, count: length)
                for i in 0 ..< length {
                    yb[i] = xb[i] * (s[i] * half + half)
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func geluBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.geluBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        let half = N(0.5)
        let slope = N(1.702)
        let scale = N(1.702 * 0.5)
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, s) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb, dxb) = (x + offset, g + offset, dx + offset)
                for i in 0 ..< length {
                    t[i] = xb[i] * scale
                }
                CPUKernels.tanh(t, into: s, count: length)
                for i in 0 ..< length {
                    let sigmoid = s[i] * half + half
                    dxb[i] = (sigmoid + slope * xb[i] * sigmoid * (1 - sigmoid)) * gb[i]
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func swish<N: NumericType>(input: Tensor<N, CPU>, beta: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0, let rowLength = swishRowLength(input: input, beta: beta) else {
            return DefaultFusedOperations<CPU>.swish(input: input, beta: beta)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let half = N(0.5)
        withBetaRow(beta, rowLength: rowLength) { betaRow in
            CPUKernels.withScratch(N.self, count: 2 * Swift.max(CPUKernels.blockSize, rowLength)) { scratch in
                let blockCapacity = Swift.max(CPUKernels.blockSize, rowLength)
                let (t, s) = (scratch, scratch + blockCapacity)
                CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                    let (xb, yb) = (x + firstRow * rowLength, y + firstRow * rowLength)
                    let length = rowCount * rowLength
                    scaleRows(xb, by: betaRow, factor: half, into: t, rows: rowCount, rowLength: rowLength)
                    CPUKernels.tanh(t, into: s, count: length)
                    for i in 0 ..< length {
                        yb[i] = xb[i] * (s[i] * half + half)
                    }
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func swishBackward<N: NumericType>(input: Tensor<N, CPU>, beta: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradients: inout (input: Tensor<N, CPU>?, beta: Tensor<N, CPU>?)) {
        guard input.count > 0, input.shape == outputGradient.shape, let rowLength = swishRowLength(input: input, beta: beta) else {
            DefaultFusedOperations<CPU>.swishBackward(input: input, beta: beta, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        let half = N(0.5)
        let (inputGradient, dx): (Tensor<N, CPU>?, UnsafeMutablePointer<N>?) = input.requiresGradient ? {
            let (tensor, pointer) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
            return (tensor, pointer)
        }() : (nil, nil)
        let computesBeta = beta.requiresGradient

        let betaRowGradient: [N] = withBetaRow(beta, rowLength: rowLength) { betaRow in
            let blockCapacity = Swift.max(CPUKernels.blockSize, rowLength)
            return CPUKernels.withScratch(N.self, count: 3 * blockCapacity + rowLength) { scratch in
                let (t, s, products) = (scratch, scratch + blockCapacity, scratch + 2 * blockCapacity)
                let betaSums = scratch + 3 * blockCapacity
                CPUKernels.fill(betaSums, with: 0, count: rowLength)
                CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
                    let (xb, gb) = (x + firstRow * rowLength, g + firstRow * rowLength)
                    let length = rowCount * rowLength
                    scaleRows(xb, by: betaRow, factor: half, into: t, rows: rowCount, rowLength: rowLength)
                    CPUKernels.tanh(t, into: s, count: length)
                    // s becomes the slope of the sigmoid, sigmoid * (1 - sigmoid), and t the sigmoid.
                    for i in 0 ..< length {
                        let sigmoid = s[i] * half + half
                        t[i] = sigmoid
                        s[i] = sigmoid * (1 - sigmoid)
                    }
                    if let dx {
                        let dxb = dx + firstRow * rowLength
                        for row in 0 ..< rowCount {
                            let start = row * rowLength
                            let (xr, gr, tr, sr, dxr) = (xb + start, gb + start, t + start, s + start, dxb + start)
                            for j in 0 ..< rowLength {
                                dxr[j] = (tr[j] + betaRow[j] * xr[j] * sr[j]) * gr[j]
                            }
                        }
                    }
                    if computesBeta {
                        for i in 0 ..< length {
                            products[i] = xb[i] * xb[i] * s[i] * gb[i]
                        }
                        for row in 0 ..< rowCount {
                            let productRow = products + row * rowLength
                            for j in 0 ..< rowLength {
                                betaSums[j] += productRow[j]
                            }
                        }
                    }
                }
                return computesBeta ? Array(UnsafeBufferPointer(start: betaSums, count: rowLength)) : []
            }
        }

        var betaGradient: Tensor<N, CPU>?
        if computesBeta {
            // A scalar beta was repeated along the row, so its gradient is the sum of the row.
            let values = beta.count == 1 ? [betaRowGradient.reduce(0, +)] : betaRowGradient
            betaGradient = Tensor(values, shape: beta.shape)
        }
        let computed: (input: Tensor<N, CPU>?, beta: Tensor<N, CPU>?) = (inputGradient, betaGradient)
        Tensor.accumulate(computed.input, into: &gradients.input)
        Tensor.accumulate(computed.beta, into: &gradients.beta)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func mish<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.mish(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, e) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                mishFactors(xb, scratch: t, exponentials: e, into: t, count: length)
                for i in 0 ..< length {
                    yb[i] = xb[i] * t[i]
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func mishBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.mishBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, e) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb, dxb) = (x + offset, g + offset, dx + offset)
                mishFactors(xb, scratch: t, exponentials: e, into: t, count: length)
                for i in 0 ..< length {
                    let factor = t[i]
                    // sigmoid(x) = 1 - 1 / (1 + exp(x)) is the derivative of log(1 + exp(x)).
                    let sigmoid = 1 - 1 / (1 + e[i])
                    dxb[i] = (factor + xb[i] * (1 - factor * factor) * sigmoid) * gb[i]
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func lisht<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.lisht(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { t in
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                CPUKernels.tanh(xb, into: t, count: length)
                for i in 0 ..< length {
                    yb[i] = xb[i] * t[i]
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func lishtBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.lishtBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { t in
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb, dxb) = (x + offset, g + offset, dx + offset)
                CPUKernels.tanh(xb, into: t, count: length)
                for i in 0 ..< length {
                    dxb[i] = (t[i] + xb[i] * (1 - t[i] * t[i])) * gb[i]
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func elu<N: NumericType>(input: Tensor<N, CPU>, alpha: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0, alpha.count == 1, alpha.dim <= input.dim else {
            return DefaultFusedOperations<CPU>.elu(input: input, alpha: alpha)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let a = alpha.elementPointer[0]
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (e, t) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                exponentialOfNegativePart(xb, scratch: t, into: e, count: length)
                for i in 0 ..< length {
                    let (value, exponential) = (xb[i], e[i])
                    yb[i] = value > 0 ? value : a * (exponential - 1)
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func eluBackward<N: NumericType>(input: Tensor<N, CPU>, alpha: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradients: inout (input: Tensor<N, CPU>?, alpha: Tensor<N, CPU>?)) {
        guard input.count > 0, alpha.count == 1, alpha.dim <= input.dim, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.eluBackward(input: input, alpha: alpha, outputGradient: outputGradient, accumulating: &gradients)
            return
        }
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        let a = alpha.elementPointer[0]
        let (inputGradient, dx): (Tensor<N, CPU>?, UnsafeMutablePointer<N>?) = input.requiresGradient ? {
            let (tensor, pointer) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
            return (tensor, pointer)
        }() : (nil, nil)
        let computesAlpha = alpha.requiresGradient

        let alphaTotal = CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (e, t) = (scratch, scratch + CPUKernels.blockSize)
            var total: N = 0
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb) = (x + offset, g + offset)
                exponentialOfNegativePart(xb, scratch: t, into: e, count: length)
                if let dxb = dx.map({ $0 + offset }) {
                    for i in 0 ..< length {
                        let (value, slope) = (xb[i], a * e[i])
                        dxb[i] = (value > 0 ? 1 : slope) * gb[i]
                    }
                }
                if computesAlpha {
                    for i in 0 ..< length {
                        t[i] = (e[i] - 1) * gb[i]
                    }
                    total += CPUKernels.sum(t, count: length)
                }
            }
            return total
        }
        let computed: (input: Tensor<N, CPU>?, alpha: Tensor<N, CPU>?) = (inputGradient, computesAlpha ? Tensor([alphaTotal], shape: alpha.shape) : nil)
        Tensor.accumulate(computed.input, into: &gradients.input)
        Tensor.accumulate(computed.alpha, into: &gradients.alpha)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softplus<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.softplus(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { e in
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                CPUKernels.exp(x + offset, into: e, count: length)
                for i in 0 ..< length {
                    e[i] += 1
                }
                CPUKernels.log(e, into: y + offset, count: length)
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softplusBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.softplusBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        CPUKernels.withScratch(N.self, count: CPUKernels.blockSize) { e in
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (gb, dxb) = (g + offset, dx + offset)
                CPUKernels.exp(x + offset, into: e, count: length)
                for i in 0 ..< length {
                    dxb[i] = (1 - 1 / (1 + e[i])) * gb[i]
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func squareplus<N: NumericType>(input: Tensor<N, CPU>) -> Tensor<N, CPU> {
        guard input.count > 0 else {
            return DefaultFusedOperations<CPU>.squareplus(input: input)
        }
        let (result, y) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let half = N(0.5)
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, r) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, yb) = (x + offset, y + offset)
                for i in 0 ..< length {
                    t[i] = xb[i] * xb[i] + 4
                }
                CPUKernels.sqrt(t, into: r, count: length)
                for i in 0 ..< length {
                    yb[i] = (xb[i] + r[i]) * half
                }
            }
        }
        return result
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func squareplusBackward<N: NumericType>(input: Tensor<N, CPU>, outputGradient: Tensor<N, CPU>, accumulating gradient: inout Tensor<N, CPU>?) {
        guard input.count > 0, input.shape == outputGradient.shape else {
            DefaultFusedOperations<CPU>.squareplusBackward(input: input, outputGradient: outputGradient, accumulating: &gradient)
            return
        }
        let (result, dx) = CPUKernels.makeTensor(shape: input.shape) as (Tensor<N, CPU>, UnsafeMutablePointer<N>)
        let x = input.elementPointer
        let g = outputGradient.elementPointer
        let half = N(0.5)
        CPUKernels.withScratch(N.self, count: 2 * CPUKernels.blockSize) { scratch in
            let (t, r) = (scratch, scratch + CPUKernels.blockSize)
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb, dxb) = (x + offset, g + offset, dx + offset)
                for i in 0 ..< length {
                    t[i] = xb[i] * xb[i] + 4
                }
                CPUKernels.sqrt(t, into: r, count: length)
                for i in 0 ..< length {
                    dxb[i] = (1 + xb[i] / r[i]) * half * gb[i]
                }
            }
        }
        Tensor.accumulate(result, into: &gradient)
    }
}

extension CPUFusedOperations {
    /// Length of the rows that repeat beta, or nil when beta has a shape that the kernels do not support.
    ///
    /// Beta is either a single value or has the shape of the trailing axes of the input.
    static func swishRowLength<N>(input: Tensor<N, CPU>, beta: Tensor<N, CPU>) -> Int? {
        guard beta.dim <= input.dim else {
            return nil
        }
        if beta.count == 1 {
            // A long row lets the element-wise loops vectorize.
            let rowLength = Swift.max(input.shape.last ?? 1, 1)
            return input.count.isMultiple(of: rowLength) ? rowLength : 1
        }
        return Array(input.shape.suffix(beta.dim)) == beta.shape ? beta.count : nil
    }

    /// Calls `body` with a row of beta of the given length. A single value is repeated along the row.
    static func withBetaRow<N: NumericType, Result>(_ beta: Tensor<N, CPU>, rowLength: Int, _ body: (UnsafePointer<N>) -> Result) -> Result {
        if beta.count == rowLength {
            return withExtendedLifetime(beta) {
                body(beta.elementPointer)
            }
        }
        return CPUKernels.withScratch(N.self, count: rowLength) { row in
            CPUKernels.fill(row, with: beta.elementPointer[0], count: rowLength)
            return body(row)
        }
    }

    /// Writes `values * beta * factor` for rows that repeat beta.
    @inline(__always)
    static func scaleRows<N: NumericType>(_ values: UnsafePointer<N>, by beta: UnsafePointer<N>, factor: N, into result: UnsafeMutablePointer<N>, rows: Int, rowLength: Int) {
        for row in 0 ..< rows {
            let (valueRow, resultRow) = (values + row * rowLength, result + row * rowLength)
            for j in 0 ..< rowLength {
                resultRow[j] = beta[j] * valueRow[j] * factor
            }
        }
    }

    /// Computes `tanh(log(1 + exp(x)))` with a single exponential, and writes `exp(x)` to `exponentials`.
    ///
    /// With `e = exp(x)` and `n = e * (e + 2)`, `tanh(log(1 + e)) = n / (n + 2)`. The input is limited to 20 before the
    /// exponential, so that `n` does not overflow. The factor already rounds to 1 there, in single and in double precision.
    /// `scratch` and `result` can be the same buffer.
    @inline(__always)
    static func mishFactors<N: NumericType>(_ x: UnsafePointer<N>, scratch: UnsafeMutablePointer<N>, exponentials: UnsafeMutablePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        let limit = N(20)
        for i in 0 ..< count {
            scratch[i] = Swift.min(x[i], limit)
        }
        CPUKernels.exp(scratch, into: exponentials, count: count)
        for i in 0 ..< count {
            let e = exponentials[i]
            let n = e * (e + 2)
            result[i] = n / (n + 2)
        }
    }

    /// Computes `exp(min(x, 0))`, which does not overflow for large inputs. `scratch` holds `count` elements.
    @inline(__always)
    static func exponentialOfNegativePart<N: NumericType>(_ x: UnsafePointer<N>, scratch: UnsafeMutablePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        for i in 0 ..< count {
            scratch[i] = Swift.min(x[i], 0)
        }
        CPUKernels.exp(scratch, into: result, count: count)
    }
}

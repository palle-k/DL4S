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
    static func tanhBackward<N: NumericType>(output: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        let (y, g) = (output.elementPointer, outputGradient.elementPointer)
        inputGradient.writeBlocks { offset, length, dx in
            let (yb, gb) = (y + offset, g + offset)
            for i in 0 ..< length {
                dx[i] = (1 - yb[i] * yb[i]) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func reluBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        inputGradient.writeBlocks { offset, length, dx in
            let (xb, gb) = (x + offset, g + offset)
            for i in 0 ..< length {
                let (value, gradient) = (xb[i], gb[i])
                dx[i] = value > 0 ? gradient : 0
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func sigmoid<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        CPUKernels.map(input, into: result) { x, y, length in
            CPUKernels.sigmoid(x, into: y, count: length)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func sigmoidBackward<N: NumericType>(output: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == output.shape, "The gradient of the result must have the shape of the result.")
        let (y, g) = (output.elementPointer, outputGradient.elementPointer)
        inputGradient.writeBlocks { offset, length, dx in
            let (yb, gb) = (y + offset, g + offset)
            for i in 0 ..< length {
                dx[i] = yb[i] * (1 - yb[i]) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func leakyRelu<N: NumericType>(input: ShapedBuffer<N, CPU>, leakage: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        precondition(CPUKernels.broadcasts(leakage.shape, to: input.shape), "The leakage must be broadcastable to the shape of the input.")
        // The kernel supports a scalar leakage.
        guard leakage.count == 1 else {
            DefaultFusedOperations<CPU>.leakyRelu(input: input, leakage: leakage, result: result)
            return
        }
        let a = leakage.elementPointer[0]
        CPUKernels.map(input, into: result) { x, y, length in
            for i in 0 ..< length {
                let value = x[i]
                y[i] = value > 0 ? value : a * value
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func leakyReluBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        leakage: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        inputGradient: GradientBuffer<N, CPU>?,
        leakageGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(CPUKernels.broadcasts(leakage.shape, to: input.shape), "The leakage must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel supports a scalar leakage.
        guard leakage.count == 1 else {
            DefaultFusedOperations<CPU>.leakyReluBackward(input: input, leakage: leakage, outputGradient: outputGradient, inputGradient: inputGradient, leakageGradient: leakageGradient)
            return
        }
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let a = leakage.elementPointer[0]
        if let inputGradient {
            inputGradient.writeBlocks { offset, length, dx in
                let (xb, gb) = (x + offset, g + offset)
                // The slope at 0 is 0, as for the rectified linear unit.
                for i in 0 ..< length {
                    let (value, gradient) = (xb[i], gb[i])
                    dx[i] = value > 0 ? gradient : (value < 0 ? a * gradient : 0)
                }
            }
        }
        if let leakageGradient {
            let products = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
            defer {
                products.deallocate()
            }
            var total: N = 0
            CPUKernels.forEachBlock(count: input.count) { offset, length in
                let (xb, gb) = (x + offset, g + offset)
                for i in 0 ..< length {
                    let (value, gradient) = (xb[i], gb[i])
                    products[i] = value < 0 ? value * gradient : 0
                }
                total += CPUKernels.sum(products, count: length)
            }
            withUnsafePointer(to: total) { leakageGradient.write($0) }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func gelu<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        let half = N(0.5)
        let t = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            t.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            CPUKernels.tanhOfHalf(x, scale: N(1.702), into: t, count: length)
            for i in 0 ..< length {
                y[i] = x[i] * (t[i] * half + half)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func geluBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let (half, slope) = (N(0.5), N(1.702))
        let t = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            t.deallocate()
        }
        inputGradient.writeBlocks { offset, length, dx in
            let (xb, gb) = (x + offset, g + offset)
            CPUKernels.tanhOfHalf(xb, scale: slope, into: t, count: length)
            for i in 0 ..< length {
                let s = t[i] * half + half
                dx[i] = (s + slope * xb[i] * s * (1 - s)) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func swish<N: NumericType>(input: ShapedBuffer<N, CPU>, beta: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        precondition(CPUKernels.broadcasts(beta.shape, to: input.shape), "The beta must be broadcastable to the shape of the input.")
        // The kernel supports a scalar beta and a beta with the shape of the trailing axes of the input.
        guard let rowLength = swishRowLength(input: input, beta: beta) else {
            DefaultFusedOperations<CPU>.swish(input: input, beta: beta, result: result)
            return
        }
        let (x, y) = (input.elementPointer, result.elementPointer)
        let half = N(0.5)
        let t = UnsafeMutablePointer<N>.allocate(capacity: Swift.max(CPUKernels.blockSize, rowLength))
        defer {
            t.deallocate()
        }
        let betaRow = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        defer {
            betaRow.deallocate()
        }
        fillBetaRow(beta, into: betaRow, rowLength: rowLength)
        CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
            let (xb, yb) = (x + firstRow * rowLength, y + firstRow * rowLength)
            let length = rowCount * rowLength
            // tanh(beta * x / 2), from which sigmoid(beta * x) follows.
            scaleRows(xb, by: betaRow, factor: half, into: t, rows: rowCount, rowLength: rowLength)
            CPUKernels.tanh(t, into: t, count: length)
            for i in 0 ..< length {
                yb[i] = xb[i] * (t[i] * half + half)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func swishBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        beta: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        inputGradient: GradientBuffer<N, CPU>?,
        betaGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(CPUKernels.broadcasts(beta.shape, to: input.shape), "The beta must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel supports a scalar beta and a beta with the shape of the trailing axes of the input.
        guard let rowLength = swishRowLength(input: input, beta: beta) else {
            DefaultFusedOperations<CPU>.swishBackward(input: input, beta: beta, outputGradient: outputGradient, inputGradient: inputGradient, betaGradient: betaGradient)
            return
        }
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let dx = inputGradient?.elementsToWrite()
        let computesBeta = (betaGradient != nil)
        let half = N(0.5)
        let blockCapacity = Swift.max(CPUKernels.blockSize, rowLength)
        let sigmoid = UnsafeMutablePointer<N>.allocate(capacity: blockCapacity)
        let slope = UnsafeMutablePointer<N>.allocate(capacity: blockCapacity)
        let products = UnsafeMutablePointer<N>.allocate(capacity: blockCapacity)
        let betaSums = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        defer {
            sigmoid.deallocate()
            slope.deallocate()
            products.deallocate()
            betaSums.deallocate()
        }
        CPUKernels.fill(betaSums, with: 0, count: rowLength)

        let betaRow = UnsafeMutablePointer<N>.allocate(capacity: rowLength)
        defer {
            betaRow.deallocate()
        }
        fillBetaRow(beta, into: betaRow, rowLength: rowLength)
        CPUKernels.forEachRowBlock(rows: input.count / rowLength, rowLength: rowLength) { firstRow, rowCount in
            let (xb, gb) = (x + firstRow * rowLength, g + firstRow * rowLength)
            let length = rowCount * rowLength
            // tanh(beta * x / 2), from which sigmoid(beta * x) follows.
            scaleRows(xb, by: betaRow, factor: half, into: sigmoid, rows: rowCount, rowLength: rowLength)
            CPUKernels.tanh(sigmoid, into: sigmoid, count: length)
            // The sigmoid replaces the tanh, and the slope of the sigmoid is sigmoid * (1 - sigmoid).
            for i in 0 ..< length {
                let s = sigmoid[i] * half + half
                sigmoid[i] = s
                slope[i] = s * (1 - s)
            }
            if let (dx, beta) = dx {
                // dx = (sigmoid + beta * x * sigmoid * (1 - sigmoid)) * g
                for row in 0 ..< rowCount {
                    let start = row * rowLength
                    let (xr, gr, sr, pr, dr) = (xb + start, gb + start, sigmoid + start, slope + start, products + start)
                    for j in 0 ..< rowLength {
                        dr[j] = (sr[j] + betaRow[j] * xr[j] * pr[j]) * gr[j]
                    }
                }
                CPUKernels.store(products, into: dx + firstRow * rowLength, beta: beta, count: length)
            }
            if computesBeta {
                for i in 0 ..< length {
                    products[i] = xb[i] * xb[i] * slope[i] * gb[i]
                }
                for row in 0 ..< rowCount {
                    let productRow = products + row * rowLength
                    for j in 0 ..< rowLength {
                        betaSums[j] += productRow[j]
                    }
                }
            }
        }
        if computesBeta {
            // A scalar beta was repeated along the row, so its gradient is the sum of the row.
            let values = beta.count == 1 ? [CPUKernels.sum(betaSums, count: rowLength)] : Array(UnsafeBufferPointer(start: betaSums, count: rowLength))
            values.withUnsafeBufferPointer { betaGradient?.write($0.baseAddress!) }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func mish<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        let factors = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            factors.deallocate()
            exponentials.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            mishFactors(x, exponentials: exponentials, into: factors, count: length)
            for i in 0 ..< length {
                y[i] = x[i] * factors[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func mishBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let factors = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            factors.deallocate()
            exponentials.deallocate()
        }
        inputGradient.writeBlocks { offset, length, dx in
            let (xb, gb) = (x + offset, g + offset)
            mishFactors(xb, exponentials: exponentials, into: factors, count: length)
            for i in 0 ..< length {
                let factor = factors[i]
                // sigmoid(x) = 1 - 1 / (1 + exp(x)) is the derivative of log(1 + exp(x)).
                let sigmoid = 1 - 1 / (1 + exponentials[i])
                dx[i] = (factor + xb[i] * (1 - factor * factor) * sigmoid) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func lisht<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        let t = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            t.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            CPUKernels.tanh(x, into: t, count: length)
            for i in 0 ..< length {
                y[i] = x[i] * t[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func lishtBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let t = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            t.deallocate()
        }
        inputGradient.writeBlocks { offset, length, dx in
            let (xb, gb) = (x + offset, g + offset)
            CPUKernels.tanh(xb, into: t, count: length)
            for i in 0 ..< length {
                dx[i] = (t[i] + xb[i] * (1 - t[i] * t[i])) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func elu<N: NumericType>(input: ShapedBuffer<N, CPU>, alpha: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        precondition(CPUKernels.broadcasts(alpha.shape, to: input.shape), "The alpha must be broadcastable to the shape of the input.")
        // The kernel supports a scalar alpha.
        guard alpha.count == 1 else {
            DefaultFusedOperations<CPU>.elu(input: input, alpha: alpha, result: result)
            return
        }
        let a = alpha.elementPointer[0]
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            exponentials.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            exponentialOfNegativePart(x, into: exponentials, count: length)
            for i in 0 ..< length {
                let (value, exponential) = (x[i], exponentials[i])
                y[i] = value > 0 ? value : a * (exponential - 1)
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func eluBackward<N: NumericType>(
        input: ShapedBuffer<N, CPU>,
        alpha: ShapedBuffer<N, CPU>,
        outputGradient: ShapedBuffer<N, CPU>,
        inputGradient: GradientBuffer<N, CPU>?,
        alphaGradient: GradientBuffer<N, CPU>?,
    ) {
        precondition(CPUKernels.broadcasts(alpha.shape, to: input.shape), "The alpha must be broadcastable to the shape of the input.")
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        // The kernel supports a scalar alpha.
        guard alpha.count == 1 else {
            DefaultFusedOperations<CPU>.eluBackward(input: input, alpha: alpha, outputGradient: outputGradient, inputGradient: inputGradient, alphaGradient: alphaGradient)
            return
        }
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let a = alpha.elementPointer[0]
        let dx = inputGradient?.elementsToWrite()
        let computesAlpha = (alphaGradient != nil)
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        let block = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            exponentials.deallocate()
            block.deallocate()
        }
        var alphaTotal: N = 0
        CPUKernels.forEachBlock(count: input.count) { offset, length in
            let (xb, gb) = (x + offset, g + offset)
            exponentialOfNegativePart(xb, into: exponentials, count: length)
            if let (dx, beta) = dx {
                for i in 0 ..< length {
                    let (value, slope) = (xb[i], a * exponentials[i])
                    block[i] = (value > 0 ? 1 : slope) * gb[i]
                }
                CPUKernels.store(block, into: dx + offset, beta: beta, count: length)
            }
            if computesAlpha {
                for i in 0 ..< length {
                    block[i] = (exponentials[i] - 1) * gb[i]
                }
                alphaTotal += CPUKernels.sum(block, count: length)
            }
        }
        if computesAlpha {
            withUnsafePointer(to: alphaTotal) { alphaGradient?.write($0) }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softplus<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            exponentials.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            CPUKernels.exp(x, into: exponentials, count: length)
            for i in 0 ..< length {
                exponentials[i] += 1
            }
            CPUKernels.log(exponentials, into: y, count: length)
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func softplusBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let exponentials = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            exponentials.deallocate()
        }
        inputGradient.writeBlocks { offset, length, dx in
            let gb = g + offset
            CPUKernels.exp(x + offset, into: exponentials, count: length)
            // sigmoid(x) = 1 - 1 / (1 + exp(x)) is the derivative of log(1 + exp(x)).
            for i in 0 ..< length {
                dx[i] = (1 - 1 / (1 + exponentials[i])) * gb[i]
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func squareplus<N: NumericType>(input: ShapedBuffer<N, CPU>, result: MutableShapedBuffer<N, CPU>) {
        let half = N(0.5)
        let roots = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            roots.deallocate()
        }
        CPUKernels.map(input, into: result) { x, y, length in
            for i in 0 ..< length {
                roots[i] = x[i] * x[i] + 4
            }
            CPUKernels.sqrt(roots, into: roots, count: length)
            for i in 0 ..< length {
                y[i] = (x[i] + roots[i]) * half
            }
        }
    }

    @_specialize(where N == Float)
    @_specialize(where N == Double)
    static func squareplusBackward<N: NumericType>(input: ShapedBuffer<N, CPU>, outputGradient: ShapedBuffer<N, CPU>, inputGradient: GradientBuffer<N, CPU>?) {
        guard let inputGradient else {
            return
        }
        precondition(outputGradient.shape == input.shape, "The gradient of the result must have the shape of the input.")
        let (x, g) = (input.elementPointer, outputGradient.elementPointer)
        let half = N(0.5)
        let roots = UnsafeMutablePointer<N>.allocate(capacity: CPUKernels.blockSize)
        defer {
            roots.deallocate()
        }
        inputGradient.writeBlocks { offset, length, dx in
            let (xb, gb) = (x + offset, g + offset)
            for i in 0 ..< length {
                roots[i] = xb[i] * xb[i] + 4
            }
            CPUKernels.sqrt(roots, into: roots, count: length)
            for i in 0 ..< length {
                dx[i] = (1 + xb[i] / roots[i]) * half * gb[i]
            }
        }
    }
}

extension CPUFusedOperations {
    /// Length of the rows that repeat beta, or nil when beta has a shape that the kernels do not support.
    ///
    /// The kernels support a single value and a beta with the shape of the trailing axes of the input.
    static func swishRowLength<N>(input: ShapedBuffer<N, CPU>, beta: ShapedBuffer<N, CPU>) -> Int? {
        if beta.count == 1 {
            // A long row lets the element-wise loops vectorize.
            let rowLength = Swift.max(input.shape.last ?? 1, 1)
            return input.count.isMultiple(of: rowLength) ? rowLength : 1
        }
        return Array(input.shape.suffix(beta.dim)) == beta.shape ? beta.count : nil
    }

    /// Writes a row of beta of the given length. A single value is repeated along the row.
    static func fillBetaRow<N: NumericType>(_ beta: ShapedBuffer<N, CPU>, into row: UnsafeMutablePointer<N>, rowLength: Int) {
        if beta.count == rowLength {
            row.update(from: beta.elementPointer, count: rowLength)
        } else {
            CPUKernels.fill(row, with: beta.elementPointer[0], count: rowLength)
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
    @inline(__always)
    static func mishFactors<N: NumericType>(_ x: UnsafePointer<N>, exponentials: UnsafeMutablePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        let limit = N(20)
        for i in 0 ..< count {
            result[i] = Swift.min(x[i], limit)
        }
        CPUKernels.exp(result, into: exponentials, count: count)
        for i in 0 ..< count {
            let e = exponentials[i]
            let n = e * (e + 2)
            result[i] = n / (n + 2)
        }
    }

    /// Computes `exp(min(x, 0))`, which does not overflow for large inputs.
    @inline(__always)
    static func exponentialOfNegativePart<N: NumericType>(_ x: UnsafePointer<N>, into result: UnsafeMutablePointer<N>, count: Int) {
        for i in 0 ..< count {
            result[i] = Swift.min(x[i], 0)
        }
        CPUKernels.exp(result, into: result, count: count)
    }
}

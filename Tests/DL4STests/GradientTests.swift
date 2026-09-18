//
//  GradientTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.10.19.
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

import DL4S
import Testing

struct GradientTests {
    @Test func testSecondDerivative() {
        let t = Tensor<Float, CPU>([1, 2, 3, 4], requiresGradient: true)

        let result = t * t * t
        let grad = result.gradients(of: [t], retainBackwardsGraph: true)[0]
        let secondGrad = grad.gradients(of: [t], retainBackwardsGraph: true)[0]
        let thirdGrad = secondGrad.gradients(of: [t], retainBackwardsGraph: true)[0]

        expectClose(grad, 3 * t * t)
        expectClose(secondGrad, 6 * t)
        expectClose(thirdGrad, Tensor(repeating: 6, shape: [4]))
    }

    /// First and second derivatives of common operations must match central difference estimates.
    @Test func testSecondDerivative2() {
        let functions: [(Tensor<Double, CPU>) -> Tensor<Double, CPU>] = [
            DL4S.exp,
            DL4S.log,
            DL4S.sum,
            DL4S.mean,
            DL4S.tanh,
            DL4S.sin,
            DL4S.cos,
            DL4S.relu,
            DL4S.sqrt,
            { DL4S.softmax($0, axis: 1) },
            { $0 * 2 },
            { $0 * $0 },
            { 1 / $0 },
            { Tensor(stacking: [$0, $0], along: 1) },
            { logSoftmax($0, axis: 1) },
        ]

        for (index, function) in functions.enumerated() {
            let t = Tensor<Double, CPU>([[2, 3, 4, 5]], requiresGradient: true)
            let result = function(t)

            let grad = result.gradients(of: [t], retainBackwardsGraph: true)[0]
            #expect(grad.requiresGradient, "function \(index)")
            expectClose(grad, numericalGradient(of: function, at: t))

            let firstDerivative: (Tensor<Double, CPU>) -> Tensor<Double, CPU> = { point in
                var point = point
                point.requiresGradient = true
                return function(point).gradients(of: [point])[0]
            }
            let secondGrad = grad.gradients(of: [t], retainBackwardsGraph: true)[0]
            expectClose(secondGrad, numericalGradient(of: firstDerivative, at: t))
        }
    }

    @Test func testSecondDerivative3() {
        let net = Concat<Float, CPU>()

        let input1 = Tensor<Float, CPU>(uniformlyDistributedWithShape: 1, 16, requiresGradient: true)
        let input2 = Tensor<Float, CPU>(uniformlyDistributedWithShape: 1, 16, requiresGradient: true)
        let result = net([input1, input2])
        let loss = meanSquaredError(expected: 1, actual: result)

        #expect(result.shape == [1, 32])

        // The loss is the sum of squared differences to 1, so the gradient of an input is 2 * (input - 1).
        let grad = loss.gradients(of: [input1], retainBackwardsGraph: true)[0]
        expectClose(grad, 2 * (input1 - 1))

        let secondGrad = grad.reduceSum().gradients(of: [input1])[0]
        expectClose(secondGrad, Tensor(repeating: 2, shape: [1, 16]))
    }

    @Test func testGradient() {
        let t2 = Tensor<Float, CPU>([1, 2, 3, 4], requiresGradient: true)
        let r2 = 1 / t2

        let grad = r2.gradients(of: [t2])[0]

        let expectedValues: [Float] = [1, 1 / 2, 1 / 3, 1 / 4]
        let expectedGrad: [Float] = [-1, -1 / 4, -1 / 9, -1 / 16]
        expectClose(r2, Tensor(expectedValues))
        expectClose(grad, Tensor(expectedGrad))
    }

    @Test func testMatMul() {
        let lhs = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6],
        ], requiresGradient: true)

        let rhs = Tensor<Float, CPU>([
            [1, 1],
            [2, 2],
            [3, 3],
        ], requiresGradient: true)

        let result = lhs.matrixMultiplied(with: rhs)
        let grads = result.gradients(of: [lhs, rhs], retainBackwardsGraph: true)

        let expectedLhsGrad = Tensor<Float, CPU>([[2, 4, 6], [2, 4, 6]])
        let expectedRhsGrad = Tensor<Float, CPU>([[5, 5], [7, 7], [9, 9]])
        #expect(grads[0] == expectedLhsGrad)
        #expect(grads[1] == expectedRhsGrad)

        // The sum of the lhs gradient is 2 * sum(rhs), so it changes with rhs only. The rhs gradient behaves the same way.
        let lhsGradGrads = grads[0].reduceSum().gradients(of: [lhs, rhs])
        #expect(lhsGradGrads[0] == Tensor(repeating: 0, shape: [2, 3]))
        #expect(lhsGradGrads[1] == Tensor(repeating: 2, shape: [3, 2]))

        let rhsGradGrads = grads[1].reduceSum().gradients(of: [lhs, rhs])
        #expect(rhsGradGrads[0] == Tensor(repeating: 2, shape: [2, 3]))
        #expect(rhsGradGrads[1] == Tensor(repeating: 0, shape: [3, 2]))
    }

    @Test func testXRNN() {
        let model = LSTM<Float, CPU>(inputSize: 32, hiddenSize: 32)
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [1, 4, 32], requiresGradient: true)

        let result = model(input).0.hiddenState
        let inputGrad = result.gradients(of: [input], retainBackwardsGraph: true)[0]

        #expect(inputGrad.shape == input.shape)
        #expect(inputGrad.elements.allSatisfy { $0.isFinite })
        #expect(inputGrad.elements.contains { $0 != 0 })
        #expect(!inputGrad.graph().isEmpty)
    }

    @Test func testRepeatedSubscriptReadAccumulatesGradient() {
        let x = Tensor<Float, CPU>([[1, 2], [3, 4]], requiresGradient: true)
        let y = (x[0] * 2 + x[0] * 3 + x[1 ..< 2] * 4 + x[1 ..< 2] * 5).reduceSum()

        #expect(y.gradients(of: [x])[0] == Tensor([[5, 5], [9, 9]]))
    }

    /// Broadcasting reads the same weight slice once per batch element, so the weight gradient must sum over the batch.
    @Test func testBroadcastMatMulWeightGradientSumsOverBatch() {
        let batchSize = 4
        let x = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 3, 5])
        let w = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 2], requiresGradient: true)
        let scale = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 3, 2])

        let broadcastGrad = (x.broadcastMatrixMultiplied(with: w) * scale).reduceSum().gradients(of: [w])[0]
        let explicitGrad = (0 ..< batchSize)
            .map { (x[$0].matrixMultiplied(with: w) * scale[$0]).reduceSum() }
            .reduce(Tensor(0), +)
            .gradients(of: [w])[0]

        expectClose(broadcastGrad, explicitGrad, tolerance: 1e-8)
    }
}

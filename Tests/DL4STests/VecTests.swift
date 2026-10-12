//
//  VecTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 26.02.19.
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
import Foundation
import Testing

/// An arithmetic operator with broadcasting, whose gradients ``VecTests`` checks.
enum ArithmeticOperator: String, CaseIterable, CustomTestStringConvertible, Sendable {
    case add
    case subtract
    case multiply
    case divide

    var testDescription: String {
        rawValue
    }

    func apply<Element: NumericType, Device: DeviceType>(_ lhs: Tensor<Element, Device>, _ rhs: Tensor<Element, Device>) -> Tensor<Element, Device> {
        switch self {
        case .add: lhs + rhs
        case .subtract: lhs - rhs
        case .multiply: lhs * rhs
        case .divide: lhs / rhs
        }
    }
}

struct VecTests {
    @Test func testVectorWriteItem() {
        var vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)

        vector[0, 0] = 2

        #expect(vector[0, 0].item == 2)

        #expect(vector[0, 1].item == 1)
        #expect(vector[1, 0].item == 2)
        #expect(vector[1, 1].item == 3)
        #expect(vector[2, 0].item == 4)
        #expect(vector[2, 1].item == 5)
    }

    @Test func testVectorWriteItem2() {
        var vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)

        vector[2, 1] = 10

        #expect(vector[2, 1].item == 10)

        #expect(vector[0, 0].item == 0)
        #expect(vector[0, 1].item == 1)
        #expect(vector[1, 0].item == 2)
        #expect(vector[1, 1].item == 3)
        #expect(vector[2, 0].item == 4)
    }

    @Test func testVectorReadSlice() {
        let v: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)

        #expect(v[nil, 0 ..< 2] == v)
        #expect(v[nil, 0 ..< 1] == Tensor([[0], [2], [4]]))
        #expect(v[nil, 1 ..< 2] == Tensor([[1], [3], [5]]))
    }

    @Test func testVectorWrite() {
        var v: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)
        v[2, 1] = 20

        #expect(v == Tensor([[0, 1], [2, 3], [4, 20]]))
    }

    @Test func testVecOps2() {
        var input: Tensor<Double, CPU> = 0
        input.requiresGradient = true
        let result = sigmoid(input)

        let grad = result.gradients(of: [input])[0]

        #expect(grad.item == 0.25)
        #expect(result.item == 0.5)
    }

    @Test func testMMul1x1() {
        let a = Tensor<Float, CPU>([1, 2, 3])
        let b = Tensor<Float, CPU>([4, 5, 6])

        let result = matMul(a, b)

        #expect(result.dim == 0)
        #expect(result.item == 32)
    }

    @Test func testMMul2x1() {
        let a = Tensor<Float, CPU>([1, 2, 3])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])

        let result = matMul(c, a)

        #expect(result.dim == 1)
        #expect(result.shape[0] == 2)
        #expect(result[0].item == 14)
        #expect(result[1].item == 32)
    }

    @Test func testMMul1x2() {
        let d = Tensor<Float, CPU>([1, 2])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])

        let result = matMul(d, c)

        #expect(result.dim == 1)
        #expect(result.shape[0] == 3)
        #expect(result[0].item == 9)
        #expect(result[1].item == 12)
        #expect(result[2].item == 15)
    }

    @Test func testMMul2x2() {
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])

        let result = matMul(c.T, c)

        #expect(result.shape == [3, 3])

        let expected: [[Float]] = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]

        for r in 0 ..< 3 {
            for c in 0 ..< 3 {
                #expect(result[r, c].item == expected[r][c])
            }
        }
    }

    @Test func testMMul2x2_2() {
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])

        let result = matMul(c, c.T)

        #expect(result.shape == [2, 2])

        let expected: [[Float]] = [[14, 32], [32, 77]]

        for r in 0 ..< 2 {
            for c in 0 ..< 2 {
                #expect(result[r, c].item == expected[r][c])
            }
        }
    }

    @Test func testLog() {
        let x = Tensor<Float, CPU>(uniformlyDistributedWithShape: 10, 10, min: -5, max: 5)

        let result = log(exp(x))

        for r in 0 ..< 10 {
            for c in 0 ..< 10 {
                expectEqual(result[r, c].item, x[r, c].item, accuracy: 0.0001)
            }
        }
    }

    @Test func testGradientAddMul() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([[4, 5, 6], [7, 8, 9]], requiresGradient: true)
        let c = Tensor<Float, CPU>([[1, 1, 1], [2, 2, 2]], requiresGradient: true)

        let result = (a + b) * c
        let grads = result.gradients(of: [a, b, c])

        #expect(grads[0] == c.detached())
        #expect(grads[1] == c.detached())
        #expect(grads[2] == (a + b).detached())
    }

    @Test func testGradientExp() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [0, -1, -2]], requiresGradient: true)

        let result = exp(a) * 2
        let aGrad = result.gradients(of: [a])[0]

        let e = Float(M_E)

        let expected: [[Float]] = [
            [e * 2, e * e * 2, e * e * e * 2],
            [2, 2 / e, 2 / (e * e)],
        ]

        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                expectEqual(aGrad[r, c].item, expected[r][c], accuracy: 0.0001)
            }
        }
    }

    @Test func testGradientLog() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [10, 20, 30]], requiresGradient: true)

        let result = log(a) * 4
        let aGrad = result.gradients(of: [a])[0]

        let expected: [[Float]] = [
            [4, 2, 4.0 / 3.0],
            [4.0 / 10.0, 4.0 / 20.0, 4.0 / 30.0],
        ]

        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                expectEqual(aGrad[r, c].item, expected[r][c], accuracy: 0.0001)
            }
        }
    }

    @Test func testGradientMatmul() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)

        let result = matMul(c, a) * 2
        #expect(result == Tensor([28, 64]))

        // The gradient of a sums the rows of c, the gradient of c repeats a in every row. Both are scaled by 2.
        let grads = result.gradients(of: [a, c])
        #expect(grads[0] == Tensor([10, 14, 18]))
        #expect(grads[1] == Tensor([[2, 4, 6], [2, 4, 6]]))
    }

    @Test func testNeg() {
        let a = Tensor<Float, CPU>([1, 2, 3, 4, 5])

        let result = -a

        #expect(result == Tensor([-1, -2, -3, -4, -5]))
    }

    @Test func testSigmoid() {
        let a = Tensor<Float, CPU>(normalDistributedWithShape: 10)

        let elements = (0 ..< 10).map { (x: Int) in a[x].item }

        let ref = elements.map { 1 / (1 + exp(-$0)) }
        let result = 1 / (1 + exp(-a))

        for i in 0 ..< 10 {
            expectEqual(result[i].item, ref[i], accuracy: 0.0001)
        }
    }

    /// Shapes of the operands of the arithmetic gradient tests: equal shapes, broadcasting of either operand, and broadcasting of both.
    static let broadcastShapes: [(lhs: [Int], rhs: [Int])] = [
        ([2], [2]),
        ([3, 2], [2]),
        ([2], [3, 2]),
        ([3, 2], [3, 1]),
        ([3, 1], [1, 2]),
        ([], [3, 2]),
        ([3, 2], []),
    ]

    @Test(arguments: ArithmeticOperator.allCases, broadcastShapes)
    func testArithmeticGradients(operation: ArithmeticOperator, shapes: (lhs: [Int], rhs: [Int])) {
        // The right operand is the divisor, so its elements are away from zero.
        let lhs = Tensor<Double, CPU>.uniform(shapes.lhs, seed: 1, requiresGradient: true)
        let rhs = Tensor<Double, CPU>.uniform(shapes.rhs, min: 0.5, max: 2, seed: 2, requiresGradient: true)
        let weights = Tensor<Double, CPU>.uniform(operation.apply(lhs, rhs).shape, min: 0.5, max: 1.5, seed: 3)
        expectGradientsMatchCentralDifferences(of: { operation.apply($0[0], $0[1]) * weights }, at: [lhs, rhs])

        // The kernels of Float differ from the kernels of Double, so the Float gradients are checked against the Double gradients.
        let expected = (operation.apply(lhs, rhs) * weights).reduceSum().gradients(of: [lhs, rhs])
        let floatSources = [lhs, rhs].map { Tensor<Float, CPU>($0.elements.map(Float.init), shape: $0.shape, requiresGradient: true) }
        let floatWeights = Tensor<Float, CPU>(weights.elements.map(Float.init), shape: weights.shape)
        let gradients = (operation.apply(floatSources[0], floatSources[1]) * floatWeights).reduceSum().gradients(of: floatSources)
        for (gradient, expectedGradient) in zip(gradients, expected) {
            expectClose(gradient, Tensor(expectedGradient.elements.map(Float.init), shape: expectedGradient.shape), tolerance: 1e-8)
        }
    }

    @Test func testAxisSum() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])

        let result = a.reduceSum(along: [0])

        #expect(result == Tensor([5, 7, 9]))
    }

    @Test func testNegativeIndices() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8]])

        #expect(a[nil, -3] == Tensor([2, 6]))
    }

    @Test func testPadding() {
        let a = Tensor<Float, CPU>(repeating: 1, shape: 1, 28, 28)
        let padded = a.padded(padding: [0, 2, 2])

        #expect(padded.shape == [1, 32, 32])
        #expect(padded.reduceSum().item == 28 * 28)
        #expect(padded[0][0 ..< 2] == Tensor(repeating: 0, shape: [2, 32]))
        #expect(padded[0][2 ..< 30][nil, 2 ..< 30] == Tensor(repeating: 1, shape: [28, 28]))
    }
}

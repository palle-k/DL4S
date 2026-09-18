//
//  EngineV2Tests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 16.03.19.
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

@testable import DL4S
import Testing

struct EngineV2Tests {
    @Test func testScatterZeroFillsLargerResult() {
        // The result buffer is larger than the source tensor and starts with placeholder values.
        // Scatter must zero the full result buffer.
        let values = Tensor<Double, CPU>([3, 1, 4])
        let context = Tensor<Int32, CPU>([0, 1, 2])

        let result = CPU.Memory.allocateBuffer(withShape: [5, 3], type: Double.self)
        defer { CPU.Memory.free(result) }
        CPU.Engine.fill(value: 42, result: result.values, count: result.count)

        CPU.Engine.scatter(reduced: values.values, context: context.values, result: result, axis: 0, ignoreIndex: -1)

        let expected: [Double] = [
            3, 0, 0,
            0, 1, 0,
            0, 0, 4,
            0, 0, 0,
            0, 0, 0,
        ]
        #expect(Buffer(result.values).array == expected)
    }

    @Test func testBroadcast1() {
        let lhs = Tensor<Float, CPU>([1, 2, 3, 4])
        let rhs = Tensor<Float, CPU>([2, 4, 6, 8])

        #expect(lhs + rhs == Tensor([3, 6, 9, 12]))
    }

    @Test func testBroadcast2() {
        let x = Tensor<Float, CPU>([1, 0.5, 0]).view(as: -1, 1)
        let result = 1 - x

        #expect(result == Tensor([[0], [0.5], [1]]))
    }

    @Test func testBroadcast3() {
        let lhs = Tensor<Float, CPU>([[1, 2], [3, 4], [5, 6]])
        let rhs = Tensor<Float, CPU>([1, 2, 3]).view(as: -1, 1)

        var result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)

        CPUEngine.broadcastAdd(lhs: lhs.values, rhs: rhs.values, result: result.mutableValues)

        #expect(result == Tensor([[2, 3], [5, 6], [8, 9]]))
    }

    @Test func testBroadcast4() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        let b = Tensor<Float, CPU>([1, 3, 3, 7])

        let result = a.unsqueezed(at: 2) + b.view(as: -1, 1, 1)

        #expect(result.shape == [4, 4, 1])
        #expect(result.squeezed() == Tensor([[2, 3, 4, 5], [8, 9, 10, 11], [12, 13, 14, 15], [20, 21, 22, 23]]))
    }

    @Test func testBroadcast5() {
        let a = Tensor<Float, CPU>(repeating: 0, shape: 16, 16)
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: 16, 1, min: 0, max: 1)

        let result = a + b

        #expect(result.shape == [16, 16])
        #expect(result == b * Tensor(repeating: 1, shape: 16, 16))
    }

    @Test func testReduceSum1() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        let v = a.values
        var result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.mutableValues

        CPU.Engine.reduceSum(values: v, result: r, axis: 0)

        #expect(result == Tensor([28, 32, 36, 40]))
    }

    @Test func testReduceSum2() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        let v = a.values
        var result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.mutableValues

        CPU.Engine.reduceSum(values: v, result: r, axis: 1)

        #expect(result == Tensor([10, 26, 42, 58]))
    }

    @Test func testReduceSum3() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        let v = a.values
        var result = Tensor<Float, CPU>(repeating: Float(0), shape: [])
        let r = result.mutableValues

        CPU.Engine.reduceSum(values: v, result: r, axes: [0, 1])

        #expect(result.item == 136)
    }

    /// Gradients of broadcast operations must have the shape of their operand and match a central difference estimate.
    @Test func testReduceOps() {
        let a = Tensor<Double, CPU>([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], requiresGradient: true)
        let b = Tensor<Double, CPU>([1, 2, 3, 4], shape: 4, 1, requiresGradient: true)

        let operations: [(Tensor<Double, CPU>, Tensor<Double, CPU>) -> Tensor<Double, CPU>] = [
            { a, b in a + b },
            { a, b in a - b },
            { a, b in b - a },
            { a, b in a * b },
            { a, b in a / b },
            { a, b in b / a },

            { a, b in a + b.T },
            { a, b in a - b.T },
            { a, b in b.T - a },
            { a, b in a * b.T },
            { a, b in a / b.T },
            { a, b in b.T / a },

            { a, b in a + b.squeezed() },
            { a, b in a - b.squeezed() },
            { a, b in b.squeezed() - a },
            { a, b in a * b.squeezed() },
            { a, b in a / b.squeezed() },
            { a, b in b.squeezed() / a },
        ]

        for (index, operation) in operations.enumerated() {
            let result = operation(a, b)
            #expect(result.shape == [4, 4], "operation \(index)")

            let grads = result.gradients(of: [a, b])
            #expect(grads[0].shape == a.shape, "operation \(index)")
            #expect(grads[1].shape == b.shape, "operation \(index)")
            expectClose(grads[0], numericalGradient(of: { operation($0, b) }, at: a))
            expectClose(grads[1], numericalGradient(of: { operation(a, $0) }, at: b))
        }
    }

    @Test func testScatter1() {
        let a = Tensor<Float, CPU>([1, 2, 3])
        let c = Tensor<Int32, CPU>([0, 1, 2])

        let result = a.scatter(using: c, alongAxis: 1, withSize: 3)
        #expect(result == Tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))

        let gathered = result.gather(using: c, alongAxis: 1)
        #expect(gathered == a)
    }

    @Test func testScatter2() {
        let a = Tensor<Float, CPU>([[1, 2, 3, 4], [5, 6, 7, 8]])
        let c = Tensor<Int32, CPU>([[0, 1, 0, 1], [1, 0, 1, 0]])

        let result = a.scatter(using: c, alongAxis: 1, withSize: 2)
        #expect(result == Tensor([
            [[1, 0, 3, 0],
             [0, 2, 0, 4]],
            [[0, 6, 0, 8],
             [5, 0, 7, 0]],
        ]))

        let gathered = result.gather(using: c, alongAxis: 1)
        #expect(gathered == a)
    }

    @Test func testBroadcastMatrixMultiply() {
        let a = Tensor<Float, CPU>([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]],
        ])
        var lhs = a.view(as: 2, 1, 2, 2)
        lhs.requiresGradient = true
        var rhs = a.view(as: 1, 2, 2, 2)
        rhs.requiresGradient = true

        let bmm = lhs.broadcastMatrixMultiplied(with: rhs)
        let ref = Tensor(stacking: [
            lhs[0, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[0, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
        ]).view(as: 2, 2, 2, 2)

        expectClose(bmm, ref)

        let grads = bmm.gradients(of: [lhs, rhs])
        let refGrads = ref.gradients(of: [lhs, rhs])
        expectClose(grads[0], refGrads[0])
        expectClose(grads[1], refGrads[1])

        let ref2 = [
            lhs[0, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[0, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
        ].reduce(0, +)
        let ref2Grads = ref2.gradients(of: [lhs, rhs])
        expectClose(grads[0], ref2Grads[0])
        expectClose(grads[1], ref2Grads[1])
    }

    @Test func testSubscriptSlice() {
        let a = Tensor<Int32, CPU>([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])

        let expected1 = Tensor<Int32, CPU>([
            [0, 1],
            [3, 4],
            [6, 7],
        ])
        let expected2 = Tensor<Int32, CPU>([
            [1, 2],
            [4, 5],
            [7, 8],
        ])
        let expected3 = Tensor<Int32, CPU>([
            [0, 1, 2],
            [3, 4, 5],
        ])
        let expected4 = Tensor<Int32, CPU>([
            [3, 4, 5],
            [6, 7, 8],
        ])

        #expect(a[nil, 0 ..< 2] == expected1)
        #expect(a[nil, 1 ..< 3] == expected2)
        #expect(a[0 ..< 2] == expected3)
        #expect(a[1 ..< 3] == expected4)
    }

    @Test func testSubscriptSliceWrite() {
        var result = Tensor<Int32, CPU>(repeating: 0, shape: [3, 3])
        let src1 = Tensor<Int32, CPU>([[0, 1], [3, 4], [6, 7]])

        result[nil, 0 ..< 2] = src1
        let expected1 = Tensor<Int32, CPU>([
            [0, 1, 0],
            [3, 4, 0],
            [6, 7, 0],
        ])
        #expect(result == expected1)

        result = Tensor<Int32, CPU>(repeating: 0, shape: [3, 3])
        let expected2 = Tensor<Int32, CPU>([
            [0, 0, 1],
            [0, 3, 4],
            [0, 6, 7],
        ])
        result[nil, 1 ..< 3] = src1
        #expect(result == expected2)
    }

    @Test func testElementwiseMinMax() {
        let x: Tensor<Float, CPU> = Tensor([1, 2, 3, 4, 5, 6], requiresGradient: true)
        let y: Tensor<Float, CPU> = Tensor([6, 5, 4, 3, 2, 1], requiresGradient: true)

        let result1 = Tensor.max(x, y) * 2
        let grads1 = result1.gradients(of: [x, y])

        #expect(grads1[0] == Tensor([0, 0, 0, 2, 2, 2]))
        #expect(grads1[1] == Tensor([2, 2, 2, 0, 0, 0]))

        let result2 = Tensor.min(x, y) * 2
        let grads2 = result2.gradients(of: [x, y])

        #expect(grads2[0] == Tensor([2, 2, 2, 0, 0, 0]))
        #expect(grads2[1] == Tensor([0, 0, 0, 2, 2, 2]))
    }

    /// Fused transposed products must have the same gradients as products of explicitly transposed operands.
    @Test func testTransposedMatmul() {
        let x = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6],
        ], requiresGradient: true)

        let y = Tensor<Float, CPU>([
            [7, 8, 9],
            [9, 10, 12],
        ], requiresGradient: true)

        let fused1 = x.matrixMultiplied(with: y, transposeSelf: true, transposeOther: false) + x.matrixMultiplied(with: y, transposeSelf: true, transposeOther: false)
        let explicit1 = x.transposed().matrixMultiplied(with: y) + x.transposed().matrixMultiplied(with: y)

        let fused2 = x.matrixMultiplied(with: y, transposeSelf: false, transposeOther: true) + x.matrixMultiplied(with: y, transposeSelf: false, transposeOther: true)
        let explicit2 = x.matrixMultiplied(with: y.transposed()) + x.matrixMultiplied(with: y.transposed())

        let fused3 = x.matrixMultiplied(with: y.transposed(), transposeSelf: true, transposeOther: true) + x.matrixMultiplied(with: y.transposed(), transposeSelf: true, transposeOther: true)
        let explicit3 = x.transposed().matrixMultiplied(with: y) + x.transposed().matrixMultiplied(with: y)

        for (fused, explicit) in [(fused1, explicit1), (fused2, explicit2), (fused3, explicit3)] {
            #expect(fused == explicit.detached())
            let fusedGrads = fused.gradients(of: [x, y])
            let explicitGrads = explicit.gradients(of: [x, y])
            #expect(fusedGrads[0] == explicitGrads[0])
            #expect(fusedGrads[1] == explicitGrads[1])
        }
    }

    @Test func testReduce() {
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: 10, 10, requiresGradient: true)
        #expect(a.reduceMax(along: 1) == a.detached().reduceMax(along: 1))
    }

    @Test func testDiagonal() {
        let a = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        let diag = a.diagonalElements()
        let expected = Tensor<Float, CPU>([1, 5, 9])
        #expect(diag == expected)
    }

    @Test func testDiagonalGeneration() {
        let b = Tensor<Float, CPU>([1, 5, 9])
        let diag = b.diagonalMatrix()

        let expected = Tensor<Float, CPU>([
            [1, 0, 0],
            [0, 5, 0],
            [0, 0, 9],
        ])
        #expect(diag == expected)
    }

    @Test func testConstantDiagonal() {
        let a = Tensor<Float, CPU>(fillingDiagonalWith: 3, size: 4)
        let expected = Tensor<Float, CPU>([
            [3, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 3],
        ])
        #expect(a == expected)
    }
}

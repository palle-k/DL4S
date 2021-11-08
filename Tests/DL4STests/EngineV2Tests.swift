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

import XCTest
@testable import DL4S

class EngineV2Tests: XCTestCase {
    func testBroadcast1() {
        // let lhs = Tensor<Float, CPU>([[1,2],[3,4],[5,6]])
        // let rhs = Tensor<Float, CPU>([1,2])
        
        // let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        // CPUEngine.broadcastMul(lhs: rhs.values, rhs: lhs.values, result: result.values)
        
        // print(result)
        let lhs = Tensor<Float, CPU>([1,2,3,4])
        let rhs = Tensor<Float, CPU>([2,4,6,8])

        print(lhs + rhs)
    }
    
    func testBroadcast2() {
        let x = Tensor<Float, CPU>([1, 0.5, 0]).view(as: -1, 1)
        let result = 1 - x
        print(result)
    }
    
    func testBroadcast3() {
        let lhs = Tensor<Float, CPU>([[1,2],[3,4],[5,6]])
        let rhs = Tensor<Float, CPU>([1,2,3]).view(as: -1, 1)
        
        let result = Tensor<Float, CPU>(repeating: 0, shape: 3, 2)
        
        CPUEngine.broadcastAdd(lhs: lhs.values, rhs: rhs.values, result: result.values)
        
        print(result)
    }
    
    func testBroadcast4() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let b = Tensor<Float, CPU>([1,3,3,7])
        
        let result = a.unsqueezed(at: 2) + b.view(as: -1, 1, 1)
        
        print(result.squeezed())
    }
    
    func testBroadcast5() {
        let a = Tensor<Float, CPU>(repeating: 0, shape: 16, 16)
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: 16, 1, min: 0, max: 1)
        
        let result = a + b
        
        print(result)
    }
    
    func testReduceSum1() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.values
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.values
        
        CPU.Engine.reduceSum(values: v, result: r, axis: 0)
        
        print(result)
    }
    
    func testReduceSum2() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.values
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: 4)
        let r = result.values
        
        CPU.Engine.reduceSum(values: v, result: r, axis: 1)
        
        print(result)
    }
    
    func testReduceSum3() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        let v = a.values
        let result = Tensor<Float, CPU>(repeating: Float(0), shape: [])
        let r = result.values
        
        CPU.Engine.reduceSum(values: v, result: r, axes: [0, 1])
        
        print(result)
    }
    
    func testReduceOps() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2,3,4], shape: 4, 1, requiresGradient: true)
        
        let s = a + b
        let d1 = a - b
        let d2 = b - a
        let p = a * b
        let q1 = a / b
        let q2 = b / a
        
        let st = a + b.T
        let dt1 = a - b.T
        let dt2 = b.T - a
        let pt = a * b.T
        let qt1 = a / b.T
        let qt2 = b.T / a
        
        let bf = b.squeezed()
        
        let sf = a + bf
        let df1 = a - bf
        let df2 = bf - a
        let pf = a * bf
        let qf1 = a / bf
        let qf2 = bf / a
        
        XCTAssertEqual(s.shape, [4, 4])
        XCTAssertEqual(d1.shape, [4, 4])
        XCTAssertEqual(d2.shape, [4, 4])
        XCTAssertEqual(p.shape, [4, 4])
        XCTAssertEqual(q1.shape, [4, 4])
        XCTAssertEqual(q2.shape, [4, 4])
        
        XCTAssertEqual(st.shape, [4, 4])
        XCTAssertEqual(dt1.shape, [4, 4])
        XCTAssertEqual(dt2.shape, [4, 4])
        XCTAssertEqual(pt.shape, [4, 4])
        XCTAssertEqual(qt1.shape, [4, 4])
        XCTAssertEqual(qt2.shape, [4, 4])
        
        XCTAssertEqual(sf.shape, [4, 4])
        XCTAssertEqual(df1.shape, [4, 4])
        XCTAssertEqual(df2.shape, [4, 4])
        XCTAssertEqual(pf.shape, [4, 4])
        XCTAssertEqual(qf1.shape, [4, 4])
        XCTAssertEqual(qf2.shape, [4, 4])
        
        for x in [s, d1, d2, p, q1, q2, st, dt1, dt2, pt, qt1, qt2, sf, df1, df2, pf, qf1, qf2] {
            let grads = x.gradients(of: [a, b])
            
            print(grads[0])
            print(grads[1])
            print()
        }
    }
    
    func testScatter1() {
        let a = Tensor<Float, CPU>([1,2,3])
        let c = Tensor<Int32, CPU>([0,1,2])

        let result = a.scatter(using: c, alongAxis: 1, withSize: 3)
        print(result)
        
        let gathered = result.gather(using: c, alongAxis: 1)
        print(gathered)
    }
    
    func testScatter2() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8]])
        let c = Tensor<Int32, CPU>([[0,1,0,1],[1,0,1,0]])
        
        let result = a.scatter(using: c, alongAxis: 1, withSize: 2)
        print(result)
        
        let gathered = result.gather(using: c, alongAxis: 1)
        print(gathered)
    }
    
    func testBroadcastMatrixMultiply() {
        let a = Tensor<Float, CPU>([
            [[1, 2],
             [3, 4]],
            [[5, 6],
             [7, 8]]
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
        
        print(bmm, terminator: "\n\n")
        print(ref, terminator: "\n\n")
        
        let grads = bmm.gradients(of: [lhs, rhs])
        print("Gradients:")
        print(grads[0], grads[1], separator: "\n", terminator: "\n\n")
        
        print("Reference Gradients:")
        let refGrads = ref.gradients(of: [lhs, rhs])
        print(refGrads[0], refGrads[1], separator: "\n", terminator: "\n\n")
        
        print("Ref2 Gradients:")
        let ref2 = [
            lhs[0, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[0, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 0]).unsqueezed(at: 0),
            lhs[1, 0].matrixMultiplied(with: rhs[0, 1]).unsqueezed(at: 0),
        ].reduce(0, +)
        let ref2Grads = ref2.gradients(of: [lhs, rhs])
        print(ref2Grads[0], ref2Grads[1], separator: "\n", terminator: "\n\n")
        
    }
    
    func testSubscriptSlice() {
        let a = Tensor<Int32, CPU>([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])
        
        let expected1 = Tensor<Int32, CPU>([
            [0, 1],
            [3, 4],
            [6, 7]
        ])
        let expected2 = Tensor<Int32, CPU>([
            [1, 2],
            [4, 5],
            [7, 8]
        ])
        let expected3 = Tensor<Int32, CPU>([
            [0, 1, 2],
            [3, 4, 5]
        ])
        let expected4 = Tensor<Int32, CPU>([
            [3, 4, 5],
            [6, 7, 8]
        ])
        
        XCTAssertEqual(a[nil, 0 ..< 2], expected1)
        XCTAssertEqual(a[nil, 1 ..< 3], expected2)
        XCTAssertEqual(a[0 ..< 2], expected3)
        XCTAssertEqual(a[1 ..< 3], expected4)
    }
    
    func testSubscriptSliceWrite() {
        var result = Tensor<Int32, CPU>(repeating: 0, shape: [3, 3])
        let src1 = Tensor<Int32, CPU>([[0, 1], [3, 4], [6, 7]])
        
        result[nil, 0 ..< 2] = src1
        let expected1 = Tensor<Int32, CPU>([
            [0, 1, 0],
            [3, 4, 0],
            [6, 7, 0]
        ])
        XCTAssertEqual(result, expected1)
        
        result = Tensor<Int32, CPU>(repeating: 0, shape: [3, 3])
        let expected2 = Tensor<Int32, CPU>([
            [0, 0, 1],
            [0, 3, 4],
            [0, 6, 7]
        ])
        result[nil, 1 ..< 3] = src1
        XCTAssertEqual(result, expected2)
    }
    
    func testElementwiseMinMax() {
        let x: Tensor<Float, CPU> = Tensor([1,2,3,4,5,6], requiresGradient: true)
        let y: Tensor<Float, CPU> = Tensor([6,5,4,3,2,1], requiresGradient: true)
        
        let result1 = Tensor.max(x, y) * 2
        let grads1 = result1.gradients(of: [x, y])
        
        XCTAssertEqual(grads1[0], Tensor([0, 0, 0, 2, 2, 2]))
        XCTAssertEqual(grads1[1], Tensor([2, 2, 2, 0, 0, 0]))
        
        let result2 = Tensor.min(x, y) * 2
        let grads2 = result2.gradients(of: [x, y])
        
        XCTAssertEqual(grads2[0], Tensor([2, 2, 2, 0, 0, 0]))
        XCTAssertEqual(grads2[1], Tensor([0, 0, 0, 2, 2, 2]))
    }
    
    func testTransposedMatmul() {
        let x = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6]
        ], requiresGradient: true)
        
        let y = Tensor<Float, CPU>([
            [7, 8, 9],
            [9, 10, 12]
        ], requiresGradient: true)
        
        let result1 = x.matrixMultiplied(with: y, transposeSelf: true, transposeOther: false) + x.matrixMultiplied(with: y, transposeSelf: true, transposeOther: false)
        _ = result1.gradients(of: [x, y])
        
        let result2 = x.matrixMultiplied(with: y, transposeSelf: false, transposeOther: true) + x.matrixMultiplied(with: y, transposeSelf: false, transposeOther: true)
        _ = result2.gradients(of: [x, y])
        
        let result3 = x.matrixMultiplied(with: y.transposed(), transposeSelf: true, transposeOther: true) + x.matrixMultiplied(with: y.transposed(), transposeSelf: true, transposeOther: true)
        _ = result3.gradients(of: [x, y])
    }
    
    func testRandomPerformance() {
        measure {
            _ = Tensor<Float, CPU>(uniformlyDistributedWithShape: 30, 50, 50, 50)
        }
    }
    
    func testReduce() {
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: 10, 10, requiresGradient: true)
        XCTAssertEqual(a.reduceMax(along: 1), a.detached().reduceMax(along: 1))
    }
    
    func testDiagonal() {
        let a = Tensor<Float, CPU>([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        let diag = a.diagonalElements()
        let expected = Tensor<Float, CPU>([1, 5, 9])
        XCTAssertEqual(diag, expected)
    }
    
    func testDiagonalGeneration() {
        let b = Tensor<Float, CPU>([1, 5, 9])
        let diag = b.diagonalMatrix()
        
        let expected = Tensor<Float, CPU>([
            [1, 0, 0],
            [0, 5, 0],
            [0, 0, 9]
        ])
        XCTAssertEqual(diag, expected)
    }
    
    func testConstantDiagonal() {
        let a = Tensor<Float, CPU>(fillingDiagonalWith: 3, size: 4)
        let expected = Tensor<Float, CPU>([
            [3, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 3]
        ])
        XCTAssertEqual(a, expected)
    }
}

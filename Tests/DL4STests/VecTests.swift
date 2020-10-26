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

import XCTest
import DL4S

class VecTests: XCTestCase {
    func testVectorWriteItem() {
        var vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
        vector[0, 0] = 2
        print(vector)
        
        XCTAssertEqual(vector[0,0].item, 2)
        
        XCTAssertEqual(vector[0,1].item, 1)
        XCTAssertEqual(vector[1,0].item, 2)
        XCTAssertEqual(vector[1,1].item, 3)
        XCTAssertEqual(vector[2,0].item, 4)
        XCTAssertEqual(vector[2,1].item, 5)
    }
    
    func testVectorWriteItem2() {
        var vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
        vector[2, 1] = 10
        print(vector)
        
        XCTAssertEqual(vector[2, 1].item, 10)
        
        XCTAssertEqual(vector[0,0].item, 0)
        XCTAssertEqual(vector[0,1].item, 1)
        XCTAssertEqual(vector[1,0].item, 2)
        XCTAssertEqual(vector[1,1].item, 3)
        XCTAssertEqual(vector[2,0].item, 4)
    }
    
    func testVectorReadSlice() {
        let v: Tensor<Float, CPU> = Tensor([0,1,2,3,4,5], shape:3,2)
        print(v)
        print(v[nil, 0 ..< 2])
        print(v[nil, 0 ..< 1])
    }
    
    func testVectorWrite() {
        var v: Tensor<Float, CPU> = Tensor([0,1,2,3,4,5], shape:3,2)
        // v[0,0] = 10
        v[2,1] = 20
        print(v)
    }
    
    func testVecOps2() {
        var input: Tensor<Double, CPU> = 0
        input.requiresGradient = true
        let result = sigmoid(input)
        
        let grad = result.gradients(of: [input])[0]
        
        debugPrint(result)
        XCTAssertEqual(grad.item, 0.25)
        XCTAssertEqual(result.item, 0.5)
    }
    
    func testMMul1x1() {
        let a = Tensor<Float, CPU>([1,2,3])
        let b = Tensor<Float, CPU>([4,5,6])
        
        let result = matMul(a, b)
        
        XCTAssertEqual(result.dim, 0)
        XCTAssertEqual(result.item, 32)
    }
    
    func testMMul2x1() {
        let a = Tensor<Float, CPU>([1,2,3])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = matMul(c, a)
        
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 2)
        XCTAssertEqual(result[0].item, 14)
        XCTAssertEqual(result[1].item, 32)
    }
    
    func testMMul1x2() {
        let d = Tensor<Float, CPU>([1,2])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = matMul(d, c)
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 3)
        XCTAssertEqual(result[0].item, 9)
        XCTAssertEqual(result[1].item, 12)
        XCTAssertEqual(result[2].item, 15)
    }
    
    func testMMul2x2() {
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = matMul(c.T, c)
        print(result)
        
        XCTAssertEqual(result.shape, [3, 3])
        
        let expected: [[Float]] = [[17, 22, 27], [22, 29, 36], [27, 36, 45]]
        
        for r in 0 ..< 3 {
            for c in 0 ..< 3 {
                XCTAssertEqual(result[r, c].item, expected[r][c])
            }
        }
    }
    
    func testMMul2x2_2() {
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = matMul(c, c.T)
        print(result)
        
        XCTAssertEqual(result.shape, [2, 2])
        
        let expected: [[Float]] = [[14, 32], [32, 77]]
        
        for r in 0 ..< 2 {
            for c in 0 ..< 2 {
                XCTAssertEqual(result[r, c].item, expected[r][c])
            }
        }
    }
    
    func testLog() {
        let x = Tensor<Float, CPU>(uniformlyDistributedWithShape: 10, 10, min: -5, max: 5)
        
        let result = log(exp(x))
        
        for r in 0 ..< 10 {
            for c in 0 ..< 10 {
                XCTAssertEqual(result[r, c].item, x[r, c].item, accuracy: 0.0001)
            }
        }
    }
    
    func testGradientAddMul() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([[4, 5, 6], [7, 8, 9]], requiresGradient: true)
        let c = Tensor<Float, CPU>([[1, 1, 1], [2, 2, 2]], requiresGradient: true)
        
        let result = (a + b) * c
        let grads = result.gradients(of: [a, b, c])
        
        print(grads[0])
        print(grads[1])
        print(grads[2])
    }
    
    func testGradientExp() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [0, -1, -2]], requiresGradient: true)
        
        let result = exp(a) * 2
        let aGrad = result.gradients(of: [a])[0]
        print(aGrad)
        
        let e = Float(M_E)
        
        let expected: [[Float]] = [
            [e * 2, e * e * 2, e * e * e * 2],
            [2, 2 / e, 2 / (e * e)]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(aGrad[r, c].item, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientLog() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [10, 20, 30]], requiresGradient: true)
        
        let result = log(a) * 4
        let aGrad = result.gradients(of: [a])[0]
        print(aGrad)
        
        let expected: [[Float]] = [
            [4, 2, 4.0 / 3.0],
            [4.0 / 10.0, 4.0 / 20.0, 4.0 / 30.0]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(aGrad[r, c].item, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientMatmul() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)
        
        let result = matMul(c, a) * 2
        print(result)
        
        let grads = result.gradients(of: [a, c])
        print(grads[0])
        print(grads[1])
    }
    
    func testNeg() {
        let a = Tensor<Float, CPU>([1,2,3,4,5])
        
        let result = -a
        print(result)
    }
    
    func testSigmoid() {
        let a = Tensor<Float, CPU>(normalDistributedWithShape: 10)
        
        let elements = (0 ..< 10).map { (x: Int) in a[x].item}
        
        let ref = elements.map {1 / (1 + exp(-$0))}
        let result = 1 / (1 + exp(-a))
        
        print(a)

        print(elements.map {1 / $0})
        print(1 / a)
                
        for i in 0 ..< 10 {
            XCTAssertEqual(result[i].item, ref[i], accuracy: 0.0001)
        }
    }
    
    func testAddBackwards() {
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b + c
        let grads = result.gradients(of: [b, c])
        
        print(grads[0])
        print(grads[1])
        
        let bExpected: [Float] = [1, 1]
        let cExpected: [Float] = [1, 1]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(grads[0][i].item, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(grads[1][i].item, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testAddBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a + b) * 2
        let grads = result.gradients(of: [a, b])
        
        print(grads[0])
        print(grads[1])
        
        let aExpected: [[Float]] = [[2,2],[2,2],[2,2]]
        let bExpected: [Float] = [6,6]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(grads[1][i].item, bExpected[i], accuracy: 0.0001)
        }
        
        for r in 0 ..< 3 {
            for c in 0 ..< 2 {
                XCTAssertEqual(grads[0][r, c].item, aExpected[r][c])
            }
        }
    }
    
    func testAddBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a + b) * 2
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testAddBackwards4() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b + a) * 2
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testSubBackwards() {
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b - c
        let grads = result.gradients(of: [b, c])
        
        print(grads[0])
        print(grads[1])
        
        let bExpected: [Float] = [1, 1]
        let cExpected: [Float] = [-1, -1]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(grads[0][i].item, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(grads[1][i].item, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testSubBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a - b) * 2
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testSubBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b - a) * 2
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[-2.0, -2.0], [-2.0, -2.0], [-2.0, -2.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testMulBackwards() {
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b * c
        
        let grads = result.gradients(of: [b, c])
        
        print(grads[0])
        print(grads[1])
        
        let bExpected: [Float] = [1, 2]
        let cExpected: [Float] = [1, 2]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(grads[0][i].item, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(grads[1][i].item, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testMulBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a * b) * 2
        
        print(a * b)
        
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testMulBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b * a) * 2
        
        let aGrad = result.gradients(of: [a])[0]
        
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testDivBackwards() {
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b / c
        let grads = result.gradients(of: [b, c])
        
        print(grads[0])
        print(grads[1])
        
        let bExpected: [Float] = [1, 0.5]
        let cExpected: [Float] = [-1, -0.5]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(grads[0][i].item, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(grads[1][i].item, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testDivBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a / b) * 2
        
        let aGrad = result.gradients(of: [a])[0]
        print(aGrad)
        
        let refGrad: [[Float]] = [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(aGrad[row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testDivBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b / a) * 2
        
        let grads = result.gradients(of: [a, b])
        
        let refGrad: [[Float]] = [[-2.0, -1.0], [-0.22222222, -0.25], [-0.08, -0.11111111]]
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(grads[0][row, column].item, refGrad[row][column], accuracy: 0.0001)
            }
        }
    }
    
    func testAxisSum() {
        let a = Tensor<Float, CPU>([[1,2,3],[4,5,6]])
        
        let result = sum(a, axes: [0])
        print(result)
    }
    
    func testNegativeIndices() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8]])
        
        print(a[nil, -3])
    }
    
    func testPadding() {
        let a = Tensor<Float, CPU>(repeating: 1, shape: 1, 28, 28)
        let padded = a.padded(padding: [0, 2, 2])
        print(padded)
    }
}

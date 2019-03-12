//
//  VecTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 26.02.19.
//

import XCTest
@testable import DL4S

class VecTests: XCTestCase {
    func testVectorWriteItem() {
        let vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
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
        let vector: Tensor<Float, CPU> = Tensor([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
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
        let v: Tensor<Float, CPU> = Tensor([0,1,2,3,4,5], shape:3,2)
        // v[0,0] = 10
        v[2,1] = 20
        print(v)
    }
    
    func testVecOps() {
        let v: Tensor<Float, CPU> = Tensor([0,1,2,3,2,1], shape:3,2)
        
        let result = log(exp(v * v))
        print(result)
        result.backwards()
        
        debugPrint(v)
    }
    
    func testVecOps2() {
        func sigmoid<Element>(_ v: Tensor<Element, CPU>) -> Tensor<Element, CPU> {
            return 1 / (1 + exp(0-v))
        }
        let input: Tensor<Double, CPU> = 0
        input.requiresGradient = true
        let result = sigmoid(input)
        
        debugPrint(result)
        XCTAssertEqual(result.gradientItem, 0)
        XCTAssertEqual(result.item, 0.5)
    }
    
    func testMMul1x1() {
        let a = Tensor<Float, CPU>([1,2,3])
        let b = Tensor<Float, CPU>([4,5,6])
        
        let result = mmul(a, b)
        
        XCTAssertEqual(result.dim, 0)
        XCTAssertEqual(result.item, 32)
    }
    
    func testMMul2x1() {
        let a = Tensor<Float, CPU>([1,2,3])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(c, a)
        
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 2)
        XCTAssertEqual(result[0].item, 14)
        XCTAssertEqual(result[1].item, 32)
    }
    
    func testMMul1x2() {
        let d = Tensor<Float, CPU>([1,2])
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(d, c)
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 3)
        XCTAssertEqual(result[0].item, 9)
        XCTAssertEqual(result[1].item, 12)
        XCTAssertEqual(result[2].item, 15)
    }
    
    func testMMul2x2() {
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(c.T, c)
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
        
        let result = mmul(c, c.T)
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
        let x = Tensor<Float, CPU>(repeating: 0, shape: 10, 10)
        Random.fill(x, a: -5, b: 5)
        
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
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        print(c.gradientDescription!)
    }
    
    func testGradientExp() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [0, -1, -2]], requiresGradient: true)
        
        let result = exp(a) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        
        let e = Float(M_E)
        
        let expected: [[Float]] = [
            [e * 2, e * e * 2, e * e * e * 2],
            [2, 2 / e, 2 / (e * e)]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(a[r, c].gradientItem!, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientLog() {
        let a = Tensor<Float, CPU>([[1, 2, 3], [10, 20, 30]], requiresGradient: true)
        
        let result = log(a) * 4
        result.backwards()
        print(a.gradientDescription!)
        
        let expected: [[Float]] = [
            [4, 2, 4.0 / 3.0],
            [4.0 / 10.0, 4.0 / 20.0, 4.0 / 30.0]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(a[r, c].gradientItem!, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientMatmul() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let c = Tensor<Float, CPU>([[1, 2, 3], [4, 5, 6]], requiresGradient: true)
        
        let result = mmul(c, a) * 2
        print(result)
        
        result.backwards()
        print(a.gradientDescription!)
        print(c.gradientDescription!)
    }
    
    func testDiv() {
        let a = Tensor<Float, CPU>([1,2,3,4,5])
        
        let result = -a
        print(result)
    }
    
    func testSigmoid() {
        let a = Tensor<Float, CPU>(repeating: 0, shape: 10)
        Random.fillNormal(a)
        
        let elements = (0 ..< 10).map { (x: Int) in Variable(value: a[x].item)}
        
        
        let ref = elements.map {1 / (1 + exp(-$0))}
        let result = 1 / (1 + exp(-a))
        
        print(a, ref)
        
        for i in 0 ..< 10 {
            XCTAssertEqual(result[i].item, ref[i].value, accuracy: 0.0001)
        }
    }
    
    func testAddBackwards() {
        //let a = Vector<Float>([[1,2],[3,4],[5,6]])
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b + c
        result.backwards()
        
        print(b.gradientDescription!)
        print(c.gradientDescription!)
        
        let bExpected: [Float] = [1, 1]
        let cExpected: [Float] = [1, 1]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(b[i].gradientItem!, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(c[i].gradientItem!, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testAddBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a + b) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let aExpected: [[Float]] = [[2,2],[2,2],[2,2]]
        let bExpected: [Float] = [6,6]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(b[i].gradientItem!, bExpected[i], accuracy: 0.0001)
        }
        
        for r in 0 ..< 3 {
            for c in 0 ..< 2 {
                XCTAssertEqual(a[r, c].gradientItem, aExpected[r][c])
            }
        }
    }
    
    func testAddBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a + b) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip($0, refB).map(+).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testAddBackwards4() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b + a) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip(refB, $0).map(+).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testSubBackwards() {
        //let a = Vector<Float>([[1,2],[3,4],[5,6]])
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b - c
        result.backwards()
        
        print(b.gradientDescription!)
        print(c.gradientDescription!)
        
        let bExpected: [Float] = [1, 1]
        let cExpected: [Float] = [-1, -1]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(b[i].gradientItem!, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(c[i].gradientItem!, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testSubBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a - b) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip($0, refB).map(-).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testSubBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b - a) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip(refB, $0).map(-).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testMulBackwards() {
        //let a = Vector<Float>([[1,2],[3,4],[5,6]])
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b * c
        result.backwards()
        
        print(b.gradientDescription!)
        print(c.gradientDescription!)
        
        let bExpected: [Float] = [1, 2]
        let cExpected: [Float] = [1, 2]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(b[i].gradientItem!, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(c[i].gradientItem!, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testMulBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a * b) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip($0, refB).map(*).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testMulBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b * a) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip(refB, $0).map(*).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testDivBackwards() {
        //let a = Vector<Float>([[1,2],[3,4],[5,6]])
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        let c = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = b / c
        result.backwards()
        
        print(b.gradientDescription!)
        print(c.gradientDescription!)
        
        let bExpected: [Float] = [1, 0.5]
        let cExpected: [Float] = [-1, -0.5]
        
        for i in 0 ..< 2 {
            XCTAssertEqual(b[i].gradientItem!, bExpected[i], accuracy: 0.0001)
            XCTAssertEqual(c[i].gradientItem!, cExpected[i], accuracy: 0.0001)
        }
    }
    
    func testDivBackwards2() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (a / b) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip($0, refB).map(/).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testDivBackwards3() {
        let a = Tensor<Float, CPU>([[1,2],[3,4],[5,6]], requiresGradient: true)
        let b = Tensor<Float, CPU>([1,2], requiresGradient: true)
        
        let result = (b / a) * 2
        result.backwards()
        
        print(a.gradientDescription!)
        print(b.gradientDescription!)
        
        let refA: [[Variable]] = [[1,2],[3,4],[5,6]]
        let refB: [Variable] = [1,2]
        
        let refResult = refA.map {zip(refB, $0).map(/).map {$0 * 2}}
        
        for row in refResult {
            for v in row {
                v.backwards()
            }
        }
        
        for row in 0 ..< 3 {
            for column in 0 ..< 2 {
                XCTAssertEqual(a[row, column].gradientItem!, refA[row][column].gradient, accuracy: 0.0001)
            }
        }
    }
    
    func testAxisSum() {
        let a = Tensor<Float, CPU>([[1,2,3],[4,5,6]])
        
        let result = sum(a, axis: 0)
        print(result)
    }
    
    func testNegativeIndices() {
        let a = Tensor<Float, CPU>([[1,2,3,4],[5,6,7,8]])
        
        print(a[nil, -3])
    }
}

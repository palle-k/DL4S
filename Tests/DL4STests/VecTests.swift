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
        let vector: Vector<Float> = Vector([0, 1, 2, 3, 4, 5], shape: 3, 2)
        
        vector[0, 0] = 2
        print(vector)
        
        XCTAssertEqual(vector[0,0].item, 2)
    }
    
    func testVectorReadSlice() {
        let v: Vector<Float> = Vector([0,1,2,3,4,5], shape:3,2)
        print(v)
        print(v[nil, 0 ..< 2])
        print(v[nil, 0 ..< 1])
    }
    
    func testVectorWrite() {
        let v: Vector<Float> = Vector([0,1,2,3,4,5], shape:3,2)
        // v[0,0] = 10
        v[2,1] = 20
        print(v)
    }
    
    func testVecOps() {
        let v: Vector<Float> = Vector([0,1,2,3,2,1], shape:3,2)
        
        let result = log(exp(v * v))
        print(result)
        result.backwards()
        
        debugPrint(v)
    }
    
    func testVecOps2() {
        func sigmoid<Element>(_ v: Vector<Element>) -> Vector<Element> {
            return 1 / (1 + exp(0-v))
        }
        let input: Vector<Double> = 0
        let result = sigmoid(input)
        
        debugPrint(result)
        XCTAssertEqual(result.gradientItem, 0)
        XCTAssertEqual(result.item, 0.5)
    }
    
    func testMMul1x1() {
        let a = Vector<Float>([1,2,3])
        let b = Vector<Float>([4,5,6])
        
        let result = mmul(a, b)
        
        XCTAssertEqual(result.dim, 0)
        XCTAssertEqual(result.item, 32)
    }
    
    func testMMul2x1() {
        let a = Vector<Float>([1,2,3])
        let c = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(c, a)
        
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 2)
        XCTAssertEqual(result[0].item, 14)
        XCTAssertEqual(result[1].item, 32)
    }
    
    func testMMul1x2() {
        let d = Vector<Float>([1,2])
        let c = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(d, c)
        print(result)
        
        XCTAssertEqual(result.dim, 1)
        XCTAssertEqual(result.shape[0], 3)
        XCTAssertEqual(result[0].item, 9)
        XCTAssertEqual(result[1].item, 12)
        XCTAssertEqual(result[2].item, 15)
    }
    
    func testMMul2x2() {
        let c = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        
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
        let c = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        
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
        let x = Vector<Float>(repeating: 0, shape: 10, 10)
        Random.fill(x, a: -5, b: 5)
        
        let result = log(exp(x))
        
        for r in 0 ..< 10 {
            for c in 0 ..< 10 {
                XCTAssertEqual(result[r, c].item, x[r, c].item, accuracy: 0.0001)
            }
        }
    }
    
    func testGradientAddMul() {
        let a = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        let b = Vector<Float>([[4, 5, 6], [7, 8, 9]])
        let c = Vector<Float>([[1, 1, 1], [2, 2, 2]])
        
        let result = (a + b) * c
        result.backwards()
        
        print(a.gradientDescription)
        print(b.gradientDescription)
        print(c.gradientDescription)
    }
    
    func testGradientExp() {
        let a = Vector<Float>([[1, 2, 3], [0, -1, -2]])
        
        let result = exp(a) * 2
        result.backwards()
        
        print(a.gradientDescription)
        
        let e = Float(M_E)
        
        let expected: [[Float]] = [
            [e * 2, e * e * 2, e * e * e * 2],
            [2, 2 / e, 2 / (e * e)]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(a[r, c].gradientItem, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientLog() {
        let a = Vector<Float>([[1, 2, 3], [10, 20, 30]])
        
        let result = log(a) * 4
        result.backwards()
        print(a.gradientDescription)
        
        let expected: [[Float]] = [
            [4, 2, 4.0 / 3.0],
            [4.0 / 10.0, 4.0 / 20.0, 4.0 / 30.0]
        ]
        
        for r in 0 ..< result.shape[0] {
            for c in 0 ..< result.shape[1] {
                XCTAssertEqual(a[r, c].gradientItem, expected[r][c], accuracy: 0.0001)
            }
        }
    }
    
    func testGradientMatmul() {
        let a = Vector<Float>([1, 2, 3])
        let c = Vector<Float>([[1, 2, 3], [4, 5, 6]])
        
        let result = mmul(c, a) * 2
        print(result)
        
        result.backwards()
        print(a.gradientDescription)
        print(c.gradientDescription)
    }
    
    func testDiv() {
        let a = Vector<Float>([1,2,3,4,5])
        
        let result = -a
        print(result)
    }
    
    func testSigmoid() {
        let a = Vector<Float>(repeating: 0, shape: 10)
        Random.fillNormal(a)
        
        let elements = (0 ..< 10).map { (x: Int) in Variable(value: a[x].item)}
        
        
        let ref = elements.map {1 / (1 + exp(-$0))}
        let result = 1 / (1 + exp(-a))
        
        print(a, ref)
        
        for i in 0 ..< 10 {
            XCTAssertEqual(result[i].item, ref[i].value, accuracy: 0.0001)
        }
    }
}

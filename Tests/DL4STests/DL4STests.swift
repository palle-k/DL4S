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

final class DL4STests: XCTestCase {
    func testExample() {
        let w0: Variable = 2.0
        let w1: Variable = -3.0
        let x0: Variable = -1.0
        let x1: Variable = -2.0
        let b: Variable = -3.0
        
        let result = 1.0 / (1.0 + exp(-(w0 * x0 + w1 * x1 + b)))
        XCTAssertEqual(result.value, 0.73, accuracy: 0.01)
        
        result.backwards()
        
        XCTAssertEqual(w0.gradient, -0.20, accuracy: 0.01)
        XCTAssertEqual(w1.gradient, -0.39, accuracy: 0.01)
        XCTAssertEqual(x0.gradient, 0.39,  accuracy: 0.01)
        XCTAssertEqual(x1.gradient, -0.59, accuracy: 0.01)
        XCTAssertEqual(b.gradient, 0.20,   accuracy: 0.01)
        
    }

    func testSimpleExample() {
        let a: Variable = 1.0
        let b: Variable = 2.0
        let c: Variable = 3.0
        let d: Variable = 4.0
        
        let result = (a * b) / (c + d)
        print(result.value)
        result.backwards()
        dump(result)
    }
    
    static var allTests = [
        ("testExample", testExample),
    ]
}

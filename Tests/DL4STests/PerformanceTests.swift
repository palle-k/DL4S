//
//  PerformanceTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 10.04.19.
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

class PerformanceTests: XCTestCase {
    func testHeaviside() {
        let src = Tensor<Float, CPU>(repeating: 0, shape: 32, 200)
        let dst = Tensor<Float, CPU>(repeating: 0, shape: 32, 200)
        
        Random.fill(src, a: -10, b: 10)
        
        let srcPtr = src.values.immutable
        let dstPtr = dst.values.pointer
        
        self.measure {
            for _ in 0 ..< 100000 {
                Float.heaviside(values: srcPtr, result: dstPtr, count: 32 * 200)
            }
        }
        
        print(sum(src) + sum(dst)) // retain src and dst
    }
    
}

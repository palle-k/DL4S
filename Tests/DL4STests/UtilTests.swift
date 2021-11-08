//
//  UtilTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 02.05.19.
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

class UtilTests: XCTestCase {
    @ThreadLocal private var x = 42
    
    func testFileReader() {
        let f = File(url: URL(fileURLWithPath: "./Package.swift"))
        
        for line in f {
            print("### \(line)")
        }
    }
    
    func testThreadLocal() {
        XCTAssertEqual(self.x, 42)
        x = 1337
        // Using sema instead of sync block, because sync block is executed on main thread.
        let sema = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            XCTAssertEqual(self.x, 42)
            self.x = 314
            XCTAssertEqual(self.x, 314)
            sema.signal()
        }
        sema.wait()
        XCTAssertEqual(self.x, 1337)
    }
}

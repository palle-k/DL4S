//
//  AutogradTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 25.02.19.
//

import XCTest
@testable import DL4S

class AutogradTests: XCTestCase {
    func testAdd() {
        let a = Variable(value: 0)
        let b = Variable(value: 2)
        
        let sum = a + b
        sum.backwards()
        
        XCTAssertEqual(a.gradient, 1)
        XCTAssertEqual(b.gradient, 1)
    }
    
    func testMul() {
        let a = Variable(value: 0)
        let b = Variable(value: 2)
        
        let prod = a * b
        prod.backwards()
        
        XCTAssertEqual(a.gradient, 2)
        XCTAssertEqual(b.gradient, 0)
    }
    
    func testSub() {
        let a = Variable(value: 2)
        let b = Variable(value: 1)
        
        let diff = a - b
        diff.backwards()
        
        XCTAssertEqual(a.gradient, 1)
        XCTAssertEqual(b.gradient, -1)
    }
    
    func testDiv() {
        let a = Variable(value: 2)
        let b = Variable(value: 3)
        
        let quot = a / b
        quot.backwards()
        
        XCTAssertEqual(a.gradient, 1.0 / 3.0)
        XCTAssertEqual(b.gradient, -2.0 / 9.0)
    }
    
    func testSum() {
        let a = Variable(value: 0)
        let b = Variable(value: 2)
        
        let sum = DL4S.sum([a, b])
        sum.backwards()
        
        XCTAssertEqual(a.gradient, 1)
        XCTAssertEqual(b.gradient, 1)
    }
    
    func testExp() {
        let a = Variable(value: 2)
        let e = exp(a)
        e.backwards()
        XCTAssertEqual(a.gradient, exp(2))
    }
    
    func testLog() {
        let a = Variable(value: 2)
        let l = log(a)
        l.backwards()
        XCTAssertEqual(a.gradient, 1.0 / 2.0)
    }
    
    func testCombinedAdd() {
        let a = Variable(value: 0)
        let b = Variable(value: 2)
        
        let sum = a + b
        sum.gradient = 2
        sum._backwards()
        
        XCTAssertEqual(a.gradient, 2)
        XCTAssertEqual(b.gradient, 2)
    }
    
    func testCombinedSub() {
        let a = Variable(value: 2)
        let b = Variable(value: 1)
        
        let diff = a - b
        diff.gradient = 2
        diff._backwards()
        
        XCTAssertEqual(a.gradient, 2)
        XCTAssertEqual(b.gradient, -2)
    }
    
    func testCombinedMul() {
        let a = Variable(value: 1)
        let b = Variable(value: 2)
        
        let prod = a * b
        prod.gradient = 2
        prod._backwards()
        
        XCTAssertEqual(a.gradient, 4)
        XCTAssertEqual(b.gradient, 2)
    }
    
    func testCombinedDiv() {
        let a = Variable(value: 2)
        let b = Variable(value: 3)
        
        let quot = a / b
        quot.gradient = 2
        quot._backwards()
        
        XCTAssertEqual(a.gradient, 2.0 / 3.0)
        XCTAssertEqual(b.gradient, -4.0 / 9.0)
    }
    
    func testCombinedSum() {
        let a = Variable(value: 0)
        let b = Variable(value: 2)
        
        let sum = DL4S.sum([a, b])
        sum.gradient = 2
        sum._backwards()
        
        XCTAssertEqual(a.gradient, 2)
        XCTAssertEqual(b.gradient, 2)
    }
    
    func testCombinedExp() {
        let a = Variable(value: 2)
        let e = exp(a)
        e.gradient = 2
        e._backwards()
        XCTAssertEqual(a.gradient, exp(2) * 2)
    }
    
    func testCombinedLog() {
        let a = Variable(value: 2)
        let l = log(a)
        l.gradient = 2
        l._backwards()
        XCTAssertEqual(a.gradient, 2.0 / 2.0)
    }
    
}

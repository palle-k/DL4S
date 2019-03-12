//
//  MemTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.03.19.
//

import XCTest
@testable import DL4S

class MemTests: XCTestCase {
    func testSliceRead() {
        let a: Tensor<Float, CPU> = Tensor((0..<16).map(Float.init), shape: 4, 4)
        
        print(a[0, nil])
        print(a[1, nil])
        print(a[2, nil])
        print(a[3, nil])
    }
}

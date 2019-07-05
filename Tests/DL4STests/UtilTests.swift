//
//  UtilTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 02.05.19.
//

import XCTest
@testable import DL4S

class UtilTests: XCTestCase {
    func testFileReader() {
        let f = File(url: URL(fileURLWithPath: "/Users/Palle/Developer/DL4S/Package.swift"))
        
        for line in f {
            print("### \(line)")
        }
    }
}

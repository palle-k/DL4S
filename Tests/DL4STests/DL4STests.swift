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
        result._backwards()
        dump(result)
    }
    
    static var allTests = [
        ("testExample", testExample),
    ]
}

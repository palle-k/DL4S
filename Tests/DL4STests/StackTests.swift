//
//  StackTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 03.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

final class StackTests: XCTestCase {
    /// A stacked tensor that lives across many backward passes must not keep the gradients of earlier passes.
    ///
    /// Each unstack pass creates 8 MB of gradient tensors. One leak would grow the process by 800 MB.
    func testLongLivedStackNodeDoesNotRetainGradients() throws {
        let sources = (0 ..< 2).map { _ in
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [1000, 1000], requiresGradient: true)
        }
        let stacked = stack(sources)
        let passCount = 100
        let leakedBytesPerPass = sources.reduce(0) { $0 + $1.count * MemoryLayout<Float>.stride }
        
        func backwardPass() -> [Tensor<Float, CPU>] {
            stacked.reduceSum().gradients(of: sources)
        }
        
        // The first pass pays for allocator warm-up, so the baseline is measured after it.
        _ = backwardPass()
        guard let baseline = ProcessMemory.residentBytes() else {
            throw XCTSkip("Resident memory is not available on this platform.")
        }
        
        for _ in 0 ..< passCount {
            let gradients = backwardPass()
            XCTAssertEqual(gradients.count, sources.count)
        }
        
        guard let final = ProcessMemory.residentBytes() else {
            throw XCTSkip("Resident memory is not available on this platform.")
        }
        let growth = final - baseline
        let tolerance = leakedBytesPerPass * passCount / 8
        XCTAssertLessThan(
            growth, tolerance,
            "Resident memory grew by \(growth / 1_000_000) MB over \(passCount) backward passes. A leak of the gradient cache grows by \(leakedBytesPerPass * passCount / 1_000_000) MB."
        )
    }
    
    /// The gradient of a stacked tensor must reach every source, also when one tensor is stacked more than once.
    func testStackGradientWithRepeatedSource() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>([4, 5, 6], requiresGradient: true)
        let weights = Tensor<Float, CPU>([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        let stacked = stack([a, b, a])
        let loss = (stacked * weights).reduceSum()
        let gradients = loss.gradients(of: [a, b])
        
        XCTAssertEqual(gradients[0], Tensor([8, 10, 12]))
        XCTAssertEqual(gradients[1], Tensor([4, 5, 6]))
    }
    
    /// A tensor that is stacked twice and also used outside the stack must receive the sum of all three contributions.
    ///
    /// The gradient from the other branch is already accumulated when the stack is visited, so this checks that
    /// the accumulator is added once, not once per repeated source.
    func testRepeatedSourceWithGradientFromOtherBranch() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let stackWeights = Tensor<Float, CPU>([1, 2, 3, 4, 5, 6])
        let otherWeights = Tensor<Float, CPU>([10, 20, 30])
        let expected = Tensor<Float, CPU>([15, 27, 39])
        
        let stackBranch = (stack([a, a]) * stackWeights).reduceSum()
        let otherBranch = (a * otherWeights).reduceSum()
        
        XCTAssertEqual((stackBranch + otherBranch).gradients(of: [a])[0], expected)
        XCTAssertEqual((otherBranch + stackBranch).gradients(of: [a])[0], expected)
        XCTAssertEqual((stackBranch + otherBranch).gradients(of: [a], retainBackwardsGraph: true)[0], expected)
    }
    
    /// A retained backward graph must give the same gradient as a plain backward pass and must allow a second derivative.
    func testStackGradientWithRetainedBackwardsGraph() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>([4, 5, 6], requiresGradient: true)
        
        let stacked = stack([a, b])
        let loss = (stacked * stacked * stacked).reduceSum()
        let firstOrder = loss.gradients(of: [a, b], retainBackwardsGraph: true)
        
        XCTAssertEqual(firstOrder[0], Tensor([3, 12, 27]))
        XCTAssertEqual(firstOrder[1], Tensor([48, 75, 108]))
        
        let secondOrder = firstOrder[0].reduceSum().gradients(of: [a])
        XCTAssertEqual(secondOrder[0], Tensor([6, 12, 18]))
    }
}

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

@testable import DL4S
import Testing

/// Holds a weak reference, so a test can check that an object was released.
private final class WeakReference<Object: AnyObject> {
    weak var value: Object?

    init(_ value: Object) {
        self.value = value
    }
}

struct StackTests {
    /// A stacked tensor that lives across many backward passes must not keep the gradients of earlier passes.
    ///
    /// The test holds weak references to the gradient storage of each pass. After the pass, the storage must be released.
    @Test func testLongLivedStackNodeDoesNotRetainGradients() {
        let sources = (0 ..< 2).map { _ in
            Tensor<Float, CPU>(uniformlyDistributedWithShape: [16, 16], requiresGradient: true)
        }
        let stacked = stack(sources)

        func backwardPass() -> [WeakReference<TensorHandle<Float, CPU>>] {
            stacked.reduceSum().gradients(of: sources).map { WeakReference($0.handle) }
        }

        for _ in 0 ..< 10 {
            let gradientStorage = backwardPass()
            #expect(gradientStorage.count == sources.count)
            #expect(gradientStorage.allSatisfy { $0.value == nil }, "The stack node keeps the gradient of a finished backward pass alive.")
        }
    }

    /// The gradient of a stacked tensor must reach every source, also when one tensor is stacked more than once.
    @Test func testStackGradientWithRepeatedSource() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>([4, 5, 6], requiresGradient: true)
        let weights = Tensor<Float, CPU>([1, 2, 3, 4, 5, 6, 7, 8, 9])

        let stacked = stack([a, b, a])
        let loss = (stacked * weights).reduceSum()
        let gradients = loss.gradients(of: [a, b])

        #expect(gradients[0] == Tensor([8, 10, 12]))
        #expect(gradients[1] == Tensor([4, 5, 6]))
    }

    /// A tensor that is stacked twice and also used outside the stack must receive the sum of all three contributions.
    ///
    /// The gradient from the other branch is already accumulated when the stack is visited, so this checks that
    /// the accumulator is added once, not once per repeated source.
    @Test func testRepeatedSourceWithGradientFromOtherBranch() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let stackWeights = Tensor<Float, CPU>([1, 2, 3, 4, 5, 6])
        let otherWeights = Tensor<Float, CPU>([10, 20, 30])
        let expected = Tensor<Float, CPU>([15, 27, 39])

        let stackBranch = (stack([a, a]) * stackWeights).reduceSum()
        let otherBranch = (a * otherWeights).reduceSum()

        #expect((stackBranch + otherBranch).gradients(of: [a])[0] == expected)
        #expect((otherBranch + stackBranch).gradients(of: [a])[0] == expected)
        #expect((stackBranch + otherBranch).gradients(of: [a], retainBackwardsGraph: true)[0] == expected)
    }

    /// A retained backward graph must give the same gradient as a plain backward pass and must allow a second derivative.
    @Test func testStackGradientWithRetainedBackwardsGraph() {
        let a = Tensor<Float, CPU>([1, 2, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>([4, 5, 6], requiresGradient: true)

        let stacked = stack([a, b])
        let loss = (stacked * stacked * stacked).reduceSum()
        let firstOrder = loss.gradients(of: [a, b], retainBackwardsGraph: true)

        #expect(firstOrder[0] == Tensor([3, 12, 27]))
        #expect(firstOrder[1] == Tensor([48, 75, 108]))

        let secondOrder = firstOrder[0].reduceSum().gradients(of: [a])
        #expect(secondOrder[0] == Tensor([6, 12, 18]))
    }
}

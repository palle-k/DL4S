//
//  OwnershipTests.swift
//  DL4S
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
@testable import DL4S


class OwnershipTests: XCTestCase {
    /// A tensor with the values of `source` and one backpropagation closure that observes the gradient flow to `source`.
    ///
    /// The closure receives the gradient of the probe and the accumulated gradient of `source`, and returns the new accumulated gradient.
    func probeBackpropagation(_ source: Tensor<Float, CPU>, _ backpropagate: @escaping (Tensor<Float, CPU>, consuming Tensor<Float, CPU>?) -> Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        Tensor(
            handle: source.handle,
            shape: source.shape,
            context: TensorContext(tag: "probe", sources: [source], backpropagateAccumulate: [backpropagate])
        )
    }
    
    func testMutableValuesCopiesSharedStorage() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        var copy = original
        CPU.Engine.fill(value: 0, result: copy.mutableValues.values, count: 3)

        XCTAssertEqual(original, Tensor([1, 2, 3]))
        XCTAssertEqual(copy, Tensor([0, 0, 0]))
    }

    func testMutableValuesCopiesViewStorage() {
        let matrix = Tensor<Float, CPU>([[1, 2], [3, 4]])
        var row = matrix[1]
        CPU.Engine.fill(value: 0, result: row.mutableValues.values, count: 2)

        XCTAssertEqual(matrix, Tensor([[1, 2], [3, 4]]))
        XCTAssertEqual(row, Tensor([0, 0]))
    }

    func testAccumulationIntoSharedBufferLeavesOtherTensorUnchanged() {
        let original = Tensor<Float, CPU>([[1, 2], [3, 4]])
        var accumulator = original
        accumulator.addingPermuted(Tensor([[10, 20], [30, 40]]), permutation: [1, 0])

        XCTAssertEqual(original, Tensor([[1, 2], [3, 4]]))
        XCTAssertEqual(accumulator, Tensor([[11, 32], [23, 44]]))
    }

    func testAccumulationIntoUniqueBufferIsInPlace() {
        var accumulator = Tensor<Float, CPU>([[1, 2], [3, 4]])
        let address = accumulator.bufferAddress
        accumulator.addingPermuted(Tensor([[10, 20], [30, 40]]), permutation: [1, 0])

        XCTAssertEqual(accumulator.bufferAddress, address)
        XCTAssertEqual(accumulator, Tensor([[11, 32], [23, 44]]))
    }

    /// A residual connection makes the gradient of the sum and the gradient of the product share one buffer.
    /// The product must not add into that buffer, or the gradient of the weight doubles.
    func testResidualConnectionGradient() {
        let a = Tensor<Float, CPU>([[1, 2], [3, 4]], requiresGradient: true)
        let w = Tensor<Float, CPU>([[1, 0], [0, 1]], requiresGradient: true)
        var sharedAddress: UnsafeMutableRawPointer?
        // The probe is the residual path. It receives the gradient of the sum and passes it on to `a` unchanged.
        let s = a.matrixMultiplied(with: w) + probeBackpropagation(a) { gradient, _ in
            sharedAddress = gradient.bufferAddress
            return gradient
        }

        let grads = s.gradients(of: [a, w])

        XCTAssertEqual(grads[1], Tensor([[4, 4], [6, 6]]))
        XCTAssertEqual(grads[0], Tensor([[2, 2], [2, 2]]))
        XCTAssertNotNil(sharedAddress)
        XCTAssertNotEqual(grads[0].bufferAddress, sharedAddress)
    }

    /// Only optimized builds accumulate without a copy. In unoptimized builds, a copy is accepted there.
    func testWeightUsedTwiceAccumulatesGradientInPlace() {
        let w = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 3], requiresGradient: true)
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let d = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        var addressAfterFirstProduct: UnsafeMutableRawPointer?
        // Backpropagation visits the products in the order b, then the probe, then a. The probe records the
        // accumulator that the first product created and hands it on without a contribution of its own.
        let probed = probeBackpropagation(w) { _, accumulator in
            addressAfterFirstProduct = accumulator?.bufferAddress
            return accumulator!
        }
        let y = (a.matrixMultiplied(with: w) + d.matrixMultiplied(with: probed)) + b.matrixMultiplied(with: w)

        let grad = y.gradients(of: [w])[0]

        let expected = (a + b).transposed().matrixMultiplied(with: Tensor(repeating: 1, shape: [5, 3]))
        XCTAssertLessThan(((grad - expected) * (grad - expected)).reduceSum().item, 1e-8)
        XCTAssertNotNil(addressAfterFirstProduct)
        #if !DEBUG
        XCTAssertEqual(grad.bufferAddress, addressAfterFirstProduct)
        #endif
    }

    func testTensorTransposedTwiceAccumulatesGradientInPlace() {
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        let c = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        let d = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        var addressAfterFirstTranspose: UnsafeMutableRawPointer?
        let probed = probeBackpropagation(a) { _, accumulator in
            addressAfterFirstTranspose = accumulator?.bufferAddress
            return accumulator!
        }
        let y = (a.transposed() * b + probed.transposed() * d) + a.transposed() * c

        let grad = y.gradients(of: [a])[0]

        let expected = (b + c).transposed()
        XCTAssertLessThan(((grad - expected) * (grad - expected)).reduceSum().item, 1e-8)
        XCTAssertNotNil(addressAfterFirstTranspose)
        #if !DEBUG
        XCTAssertEqual(grad.bufferAddress, addressAfterFirstTranspose)
        #endif
    }

    func testTransposedMatrixProductGradients() {
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4], requiresGradient: true)
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 5], requiresGradient: true)
        let scale = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 5])

        for (transposeLhs, transposeRhs) in [(false, false), (true, false), (false, true), (true, true)] {
            let lhs = transposeLhs ? a.transposed().detached() : a.detached()
            let rhs = transposeRhs ? b.transposed().detached() : b.detached()
            let lhsParameter = Tensor<Float, CPU>(lhs.elements, shape: lhs.shape, requiresGradient: true)
            let rhsParameter = Tensor<Float, CPU>(rhs.elements, shape: rhs.shape, requiresGradient: true)

            let fused = (lhsParameter.matrixMultiplied(with: rhsParameter, transposeSelf: transposeLhs, transposeOther: transposeRhs) * scale).reduceSum()
            let fusedGrads = fused.gradients(of: [lhsParameter, rhsParameter])

            let explicit = ((transposeLhs ? lhsParameter.transposed() : lhsParameter).matrixMultiplied(with: transposeRhs ? rhsParameter.transposed() : rhsParameter) * scale).reduceSum()
            let explicitGrads = explicit.gradients(of: [lhsParameter, rhsParameter])

            for (fusedGrad, explicitGrad) in zip(fusedGrads, explicitGrads) {
                XCTAssertEqual(fusedGrad.shape, explicitGrad.shape)
                XCTAssertLessThan(((fusedGrad - explicitGrad) * (fusedGrad - explicitGrad)).reduceSum().item, 1e-8, "transposeLhs: \(transposeLhs), transposeRhs: \(transposeRhs)")
            }
        }
    }

    func testCopiedTensorKeepsBackpropID() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        let copy = original
        
        XCTAssertEqual(copy.backpropID, original.backpropID)
        XCTAssertNotEqual(Tensor<Float, CPU>([1, 2, 3]).backpropID, original.backpropID)
    }
    
    func testEnsureOwnershipOnSharedBufferMintsNewBackpropID() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        var copy = original
        copy.ensureOwnership()
        
        XCTAssertNotEqual(copy.backpropID, original.backpropID)
        XCTAssertNotEqual(copy.bufferAddress, original.bufferAddress)
        XCTAssertEqual(copy, original)
    }
    
    func testEnsureOwnershipOnUniqueBufferKeepsBackpropID() {
        var tensor = Tensor<Float, CPU>([1, 2, 3])
        let id = tensor.backpropID
        let address = tensor.bufferAddress
        tensor.ensureOwnership()
        
        XCTAssertEqual(tensor.backpropID, id)
        XCTAssertEqual(tensor.bufferAddress, address)
    }
}

private extension Tensor where Device == CPU {
    var bufferAddress: UnsafeMutableRawPointer? {
        values.values.memory.baseAddress
    }

}

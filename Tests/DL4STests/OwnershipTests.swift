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

@testable import DL4S
import Synchronization
import Testing

struct OwnershipTests {
    /// A tensor with the values of `source` and one backpropagation closure that observes the gradient flow to `source`.
    ///
    /// The closure receives the gradient of the probe and the accumulated gradient of `source`, and returns the new accumulated gradient.
    func probeBackpropagation(_ source: Tensor<Float, CPU>, _ backpropagate: @escaping @Sendable (Tensor<Float, CPU>, consuming Tensor<Float, CPU>?) -> Tensor<Float, CPU>) -> Tensor<Float, CPU> {
        Tensor(
            handle: source.handle,
            shape: source.shape,
            context: TensorContext(tag: "probe", sources: [source], backpropagateAccumulate: [backpropagate]),
        )
    }

    @Test func testMutableValuesCopiesSharedStorage() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        var copy = original
        CPU.Engine.fill(value: 0, result: copy.mutableValues.values, count: 3)

        #expect(original == Tensor([1, 2, 3]))
        #expect(copy == Tensor([0, 0, 0]))
    }

    @Test func testMutableValuesCopiesViewStorage() {
        let matrix = Tensor<Float, CPU>([[1, 2], [3, 4]])
        var row = matrix[1]
        CPU.Engine.fill(value: 0, result: row.mutableValues.values, count: 2)

        #expect(matrix == Tensor([[1, 2], [3, 4]]))
        #expect(row == Tensor([0, 0]))
    }

    @Test func testAccumulationIntoSharedBufferLeavesOtherTensorUnchanged() {
        let original = Tensor<Float, CPU>([[1, 2], [3, 4]])
        var accumulator = original
        accumulator.addingPermuted(Tensor([[10, 20], [30, 40]]), permutation: [1, 0])

        #expect(original == Tensor([[1, 2], [3, 4]]))
        #expect(accumulator == Tensor([[11, 32], [23, 44]]))
    }

    @Test func testAccumulationIntoUniqueBufferIsInPlace() {
        var accumulator = Tensor<Float, CPU>([[1, 2], [3, 4]])
        let address = accumulator.bufferAddress
        accumulator.addingPermuted(Tensor([[10, 20], [30, 40]]), permutation: [1, 0])

        #expect(accumulator.bufferAddress == address)
        #expect(accumulator == Tensor([[11, 32], [23, 44]]))
    }

    /// A residual connection makes the gradient of the sum and the gradient of the product share one buffer.
    /// The product must not add into that buffer, or the gradient of the weight doubles.
    @Test func testResidualConnectionGradient() {
        let a = Tensor<Float, CPU>([[1, 2], [3, 4]], requiresGradient: true)
        let w = Tensor<Float, CPU>([[1, 0], [0, 1]], requiresGradient: true)
        let sharedAddress = Mutex<UInt?>(nil)
        // The probe is the residual path. It receives the gradient of the sum and passes it on to `a` unchanged.
        let s = a.matrixMultiplied(with: w) + probeBackpropagation(a) { gradient, _ in
            sharedAddress.withLock { $0 = gradient.bufferAddress }
            return gradient
        }

        let grads = s.gradients(of: [a, w])

        #expect(grads[1] == Tensor([[4, 4], [6, 6]]))
        #expect(grads[0] == Tensor([[2, 2], [2, 2]]))
        #expect(sharedAddress.withLock { $0 } != nil)
        #expect(grads[0].bufferAddress != sharedAddress.withLock { $0 })
    }

    /// Only optimized builds accumulate without a copy. In unoptimized builds, a copy is accepted there.
    @Test func testWeightUsedTwiceAccumulatesGradientInPlace() {
        let w = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 3], requiresGradient: true)
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let d = Tensor<Float, CPU>(uniformlyDistributedWithShape: [5, 4])
        let addressAfterFirstProduct = Mutex<UInt?>(nil)
        // Backpropagation visits the products in the order b, then the probe, then a. The probe records the
        // accumulator that the first product created and hands it on without a contribution of its own.
        let probed = probeBackpropagation(w) { _, accumulator in
            addressAfterFirstProduct.withLock { $0 = accumulator?.bufferAddress }
            return accumulator!
        }
        let y = (a.matrixMultiplied(with: w) + d.matrixMultiplied(with: probed)) + b.matrixMultiplied(with: w)

        let grad = y.gradients(of: [w])[0]

        let expected = (a + b).transposed().matrixMultiplied(with: Tensor(repeating: 1, shape: [5, 3]))
        expectClose(grad, expected, tolerance: 1e-8)
        #expect(addressAfterFirstProduct.withLock { $0 } != nil)
        #if !DEBUG
        #expect(grad.bufferAddress == addressAfterFirstProduct.withLock { $0 })
        #endif
    }

    @Test func testTensorTransposedTwiceAccumulatesGradientInPlace() {
        let a = Tensor<Float, CPU>(uniformlyDistributedWithShape: [4, 3], requiresGradient: true)
        let b = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        let c = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        let d = Tensor<Float, CPU>(uniformlyDistributedWithShape: [3, 4])
        let addressAfterFirstTranspose = Mutex<UInt?>(nil)
        let probed = probeBackpropagation(a) { _, accumulator in
            addressAfterFirstTranspose.withLock { $0 = accumulator?.bufferAddress }
            return accumulator!
        }
        let y = (a.transposed() * b + probed.transposed() * d) + a.transposed() * c

        let grad = y.gradients(of: [a])[0]

        let expected = (b + c).transposed()
        expectClose(grad, expected, tolerance: 1e-8)
        #expect(addressAfterFirstTranspose.withLock { $0 } != nil)
        #if !DEBUG
        #expect(grad.bufferAddress == addressAfterFirstTranspose.withLock { $0 })
        #endif
    }

    @Test func testTransposedMatrixProductGradients() {
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
                expectClose(fusedGrad, explicitGrad, tolerance: 1e-8)
            }
        }
    }

    @Test func testCopiedTensorKeepsBackpropID() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        let copy = original

        #expect(copy.backpropID == original.backpropID)
        #expect(Tensor<Float, CPU>([1, 2, 3]).backpropID != original.backpropID)
    }

    @Test func testEnsureOwnershipOnSharedBufferMintsNewBackpropID() {
        let original = Tensor<Float, CPU>([1, 2, 3])
        var copy = original
        copy.ensureOwnership()

        #expect(copy.backpropID != original.backpropID)
        #expect(copy.bufferAddress != original.bufferAddress)
        #expect(copy == original)
    }

    @Test func testEnsureOwnershipOnUniqueBufferKeepsBackpropID() {
        var tensor = Tensor<Float, CPU>([1, 2, 3])
        let id = tensor.backpropID
        let address = tensor.bufferAddress
        tensor.ensureOwnership()

        #expect(tensor.backpropID == id)
        #expect(tensor.bufferAddress == address)
    }
}

private extension Tensor where Device == CPU {
    /// Address of the storage as an integer, so tests can record it from `@Sendable` closures.
    var bufferAddress: UInt? {
        values.values.memory.baseAddress.map { UInt(bitPattern: $0) }
    }
}

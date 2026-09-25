//
//  GPUConcurrencyTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 24.09.26.
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import DL4S
import Foundation
import Synchronization
import Testing

/// Stress tests that record GPU work from several threads at the same time.
///
/// The GPU kernels are deterministic, so every thread must get exactly the results of a serial run.
extension GPUTests {
    @Suite(.serialized)
    struct GPUConcurrencyTests {
        private func run(threadCount: Int, _ body: @escaping @Sendable (_ threadIndex: Int) -> Void) {
            let group = DispatchGroup()
            for index in 0 ..< threadCount {
                group.enter()
                let thread = Thread {
                    body(index)
                    group.leave()
                }
                thread.name = "DL4S.GPUConcurrencyTests.\(index)"
                thread.start()
            }
            group.wait()
        }

        /// Trains a model with the given seed for a few steps and returns its predictions.
        private static func train(seed: UInt64, steps: Int) -> [Float] {
            var generator = WyHash(seed: seed)
            var model = Sequential {
                Dense<Float, GPU>(inputSize: 16, outputSize: 128, using: &generator)
                Tanh<Float, GPU>()
                Dense<Float, GPU>(inputSize: 128, outputSize: 4, using: &generator)
                Softmax<Float, GPU>()
            }
            var optimizer = Adam<Float, GPU>(learningRate: 0.01)
            let input = Tensor<Float, GPU>(uniformlyDistributedWithShape: [512, 16], min: -1, max: 1, using: &generator)
            let labels = Tensor<Int32, GPU>((0 ..< 512).map { Int32($0 % 4) })
            for _ in 0 ..< steps {
                let loss = categoricalCrossEntropy(expected: labels, actual: model(input))
                model.update { parameters in
                    optimizer.update(&parameters, along: loss.gradients(of: parameters))
                }
            }
            return model(input).elements
        }

        @Test(arguments: [2, 4, 8])
        func concurrentTrainingMatchesSerialTraining(threadCount: Int) {
            // Whether a small operation runs on the host depends on the timing of the GPU, and the host rounds some functions
            // differently. With all operations on the GPU, the results do not depend on the timing.
            GPU.hostExecutionLimit = 0
            defer { GPU.hostExecutionLimit = 4096 }
            let references = (0 ..< threadCount).map { Self.train(seed: UInt64($0), steps: 10) }
            let results = Mutex<[Int: [Float]]>([:])
            run(threadCount: threadCount) { index in
                let result = Self.train(seed: UInt64(index), steps: 10)
                results.withLock { $0[index] = result }
            }
            let collected = results.withLock { $0 }
            for index in 0 ..< threadCount {
                #expect(collected[index] == references[index], "thread \(index) differs from the serial run")
            }
        }

        @Test func concurrentTrainingWithHostOperationsMatchesSerialTraining() {
            let threadCount = 6
            let references = (0 ..< threadCount).map { Self.train(seed: UInt64($0), steps: 10) }
            let results = Mutex<[Int: [Float]]>([:])
            run(threadCount: threadCount) { index in
                let result = Self.train(seed: UInt64(index), steps: 10)
                results.withLock { $0[index] = result }
            }
            let collected = results.withLock { $0 }
            for index in 0 ..< threadCount {
                let difference = zip(collected[index] ?? [], references[index]).map { Swift.abs($0 - $1) }.max() ?? .infinity
                #expect(difference < 1e-4, "thread \(index) differs from the serial run by \(difference)")
            }
        }

        @Test func concurrentInferenceOnSharedTensorsMatchesSerialReference() {
            var generator = WyHash(seed: 3)
            let weights = Tensor<Float, GPU>(uniformlyDistributedWithShape: [256, 256], min: -0.1, max: 0.1, using: &generator)
            let input = Tensor<Float, GPU>(uniformlyDistributedWithShape: [128, 256], min: -1, max: 1, using: &generator)
            let reference = input.matrixMultiplied(with: weights).gaussianErrorLinear().softmax(axis: 1).elements
            let mismatches = Atomic<Int>(0)
            run(threadCount: 8) { _ in
                for _ in 0 ..< 20 {
                    // Some results are read at once and some later, so reads and new work of other threads interleave.
                    let result = input.matrixMultiplied(with: weights).gaussianErrorLinear().softmax(axis: 1)
                    if result.elements != reference {
                        mismatches.add(1, ordering: .relaxed)
                    }
                }
            }
            #expect(mismatches.load(ordering: .relaxed) == 0)
        }
    }
}
#endif

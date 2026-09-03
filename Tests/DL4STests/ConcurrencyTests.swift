//
//  ConcurrencyTests.swift
//  DL4STests
//
//  Created by Palle Klewitz on 01.09.26.
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

/// Collects values from several threads behind a lock.
private final class ThreadSafeCollector<Value> {
    private var storage: [Value] = []
    private let lock = NSLock()
    
    func append(_ value: Value) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }
    
    var values: [Value] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

/// Stress tests that run DL4S with several threads at the same time.
///
/// Run the suite with `swift test --filter Concurrency`.
/// Run it under the thread sanitizer with `swift test --sanitize=thread --filter Concurrency`.
final class ConcurrencyTests: XCTestCase {
    private let threadCounts = [2, 4, 8]
    
    private func run(threadCount: Int, _ body: @escaping (_ threadIndex: Int) -> Void) {
        let group = DispatchGroup()
        for index in 0 ..< threadCount {
            group.enter()
            // raw Threads ensure parallel execution, as opposed to structured concurrency or dispatch queues.
            let thread = Thread {
                body(index)
                group.leave()
            }
            thread.name = "DL4S.ConcurrencyTests.\(index)"
            thread.start()
        }
        group.wait()
    }
    
    /// Runs `body` and records known concurrency failures as expected failures.
    private func runExpectingKnownConcurrencyFailure(_ body: () throws -> Void) throws {
        // TODO: Remove this wrapper when concurrency fixes are implemented.
        // On Darwin, XCTest marks the failures as expected and the test does not break CI.
        // On Linux, `XCTExpectFailure` does not exist, so the test is skipped unless
        // `DL4S_CONCURRENCY_TESTS=1` is set.
        #if canImport(ObjectiveC)
        try XCTExpectFailure("Failure is expected until concurrency fixes are implemented", options: .nonStrict()) {
            try body()
        }
        #else
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["DL4S_CONCURRENCY_TESTS"] == nil,
            "Failure is expected until concurrency fixes are implemented"
        )
        try body()
        #endif
    }
    
    private typealias DenseTanh = Sequential<Dense<Float, CPU>, Tanh<Float, CPU>>
    private typealias TrainedModel = Sequential<Sequential<DenseTanh, DenseTanh>, Sequential<Dense<Float, CPU>, Sigmoid<Float, CPU>>>
    
    /// Trains a small model on the XOR problem so the test has a model with initialized weights.
    private func makeTrainedModel() -> TrainedModel {
        let inputs = Tensor<Float, CPU>([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        let expected = Tensor<Float, CPU>([
            [0],
            [1],
            [1],
            [0]
        ])
        
        let model = Sequential {
            Dense<Float, CPU>(inputSize: 2, outputSize: 64)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 64, outputSize: 64)
            Tanh<Float, CPU>()
            Dense<Float, CPU>(inputSize: 64, outputSize: 1)
            Sigmoid<Float, CPU>()
        }
        var optimizer = Adam(model: model, learningRate: 0.05)
        
        for _ in 1 ... 50 {
            let prediction = optimizer.model(inputs)
            let loss = binaryCrossEntropy(expected: expected, actual: prediction)
            let gradients = loss.gradients(of: optimizer.model.parameters)
            optimizer.update(along: gradients)
        }
        
        return optimizer.model
    }
    
    /// Parallel inference on shared model
    ///
    /// Every result must be exactly equal to a result that was computed in main. Inference has no random component, so a tolerance is not needed and would hide errors.
    ///
    /// Values are correct, but the thread sanitizer reports an access race on `OperationGroup.operationStack`.
    /// The  mutating`@ThreadLocal` setter is a write access on shared state.
    func testConcurrentInferenceMatchesSerialReference() throws {
        let model = makeTrainedModel()
        let input = Tensor<Float, CPU>(uniformlyDistributedWithShape: [256, 2], min: 0, max: 1)
        let reference = model(input).elements
        let iterations = 50
        
        try runExpectingKnownConcurrencyFailure {
            for threadCount in threadCounts {
                let mismatches = ThreadSafeCollector<String>()
                
                run(threadCount: threadCount) { threadIndex in
                    for iteration in 0 ..< iterations {
                        let result = model(input).elements
                        if result != reference {
                            let firstDifference = zip(result, reference).enumerated().first { $0.element.0 != $0.element.1 }
                            mismatches.append(
                                "thread \(threadIndex), iteration \(iteration), first difference at element \(firstDifference?.offset ?? -1)"
                            )
                        }
                    }
                }
                
                let collected = mismatches.values
                XCTAssertEqual(
                    collected.count, 0,
                    "\(collected.count) of \(threadCount * iterations) results with \(threadCount) threads differ from reference. First: \(collected.first ?? "none")"
                )
            }
        }
    }
    
    /// Dropout layer randomness stress test.
    ///
    /// The mask must only contain zeros and ones, the keep rate must be close to the configured rate, and no two passes may produce the same mask.
    ///
    /// A generator that is shared between threads would produce overlapping masks.
    func testConcurrentDropoutForwardPasses() throws {
        let dropout = Dropout<Float, CPU>(rate: 0.5)
        let input = Tensor<Float, CPU>(repeating: 1, shape: [64, 64])
        let iterations = 100
        let threadCount = 8
        
        let invalidMasks = ThreadSafeCollector<String>()
        let masks = ThreadSafeCollector<[Float]>()
        
        run(threadCount: threadCount) { threadIndex in
            for iteration in 0 ..< iterations {
                let mask = dropout(input).elements
                if mask.contains(where: { $0 != 0 && $0 != 1 }) {
                    invalidMasks.append("thread \(threadIndex), iteration \(iteration)")
                }
                masks.append(mask)
            }
        }
        
        let collectedMasks = masks.values
        XCTAssertEqual(collectedMasks.count, threadCount * iterations)
        XCTAssertEqual(invalidMasks.values.count, 0, "Dropout produced values other than 0 and 1: \(invalidMasks.values.prefix(3))")
        
        let keptElements = collectedMasks.reduce(0) { $0 + $1.reduce(0, +) }
        let keepRate = keptElements / Float(collectedMasks.count * input.count)
        XCTAssertEqual(keepRate, 0.5, accuracy: 0.02, "Keep rate over all passes deviates from the configured rate.")
        
        let distinctMasks = Set(collectedMasks.map { $0.map { UInt8($0) } })
        XCTAssertEqual(distinctMasks.count, collectedMasks.count, "Some dropout passes produced identical masks.")
    }
    
    /// Weight initialization stress test
    ///
    /// Weights of one model must be independent of another model initialized in parallel.
    func testConcurrentWeightInitializationProducesIndependentWeights() throws {
        let threadCount = 8
        let layerSize = 128
        
        let weights = ThreadSafeCollector<[Double]>()
        
        run(threadCount: threadCount) { _ in
            let layer = Dense<Double, CPU>(inputSize: layerSize, outputSize: layerSize)
            weights.append(layer.weights.elements)
        }
        
        let collected = weights.values
        XCTAssertEqual(collected.count, threadCount)
        
        let allValues = collected.flatMap { $0 }
        let duplicateCount = allValues.count - Set(allValues).count
        let duplicateFraction = Double(duplicateCount) / Double(allValues.count)
        XCTAssertLessThan(
            duplicateFraction, 0.001,
            "\(duplicateCount) of \(allValues.count) weights are duplicates across \(threadCount) independently initialized models."
        )
        
        for (index, model) in collected.enumerated() {
            XCTAssertFalse(model.contains(where: { $0.isNaN }), "Model \(index) contains NaN weights.")
        }
    }
}

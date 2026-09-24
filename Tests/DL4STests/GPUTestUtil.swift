//
//  GPUTestUtil.swift
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
@testable import DL4S
import Foundation
import Testing

/// Parent of the suites that use the GPU.
///
/// The suites change global settings of the GPU, such as ``GPU/hostExecutionLimit``, so they must not run at the same time.
@Suite(.enabled(if: GPU.isAvailable, "The system has no Metal device."), .serialized)
struct GPUTests {}

/// Where the GPU runs an operation in a test case.
enum GPUPlacementMode: String, CaseIterable, CustomTestStringConvertible {
    /// All operations with GPU kernels run on the GPU.
    case gpu
    /// Small operations whose operands are on the host run on the host.
    case mixed

    var testDescription: String {
        rawValue
    }

    var hostExecutionLimit: Int {
        switch self {
        case .gpu: 0
        case .mixed: 1 << 30
        }
    }
}

/// Helpers that run an operation on the CPU and on the GPU and compare the results.
enum GPUTest {
    /// Runs `body` on the CPU, then on the GPU in every placement mode, and records an issue when a result differs.
    ///
    /// - Parameters:
    ///   - tolerance: Largest difference of an element, relative to the magnitude of the expected element when it is larger than 1.
    ///   - body: Computes the results on the GPU when its argument is true, on the CPU otherwise.
    static func compare(_ name: String, tolerance: Float = 1e-3, _ body: (Bool) -> [Tensor<Float, CPU>], sourceLocation: SourceLocation = #_sourceLocation) {
        let expected = body(false)
        for mode in GPUPlacementMode.allCases {
            GPU.hostExecutionLimit = mode.hostExecutionLimit
            let actual = body(true)
            #expect(actual.count == expected.count, "\(name) [\(mode)]: result count", sourceLocation: sourceLocation)
            for (index, (a, e)) in zip(actual, expected).enumerated() {
                guard a.shape == e.shape else {
                    Issue.record("\(name) [\(mode)]: shape \(a.shape) differs from \(e.shape)", sourceLocation: sourceLocation)
                    continue
                }
                let difference = zip(a.elements, e.elements).map { Swift.abs($0 - $1) / Swift.max(1, Swift.abs($1)) }.max() ?? 0
                #expect(difference <= tolerance, "\(name) [\(mode)]: result \(index) has the relative difference \(difference)", sourceLocation: sourceLocation)
            }
        }
        GPU.hostExecutionLimit = 4096
    }

    static func random(_ shape: [Int], seed: UInt64, min: Float = -1, max: Float = 1, requiresGradient: Bool = false) -> Tensor<Float, CPU> {
        var generator = WyHash(seed: seed)
        return Tensor<Float, CPU>(uniformlyDistributedWithShape: shape, min: min, max: max, requiresGradient: requiresGradient, using: &generator)
    }

    /// Runs `cpu` on the inputs, or `gpu` on copies of the inputs on the GPU.
    static func run<Result>(on gpu: Bool, _ inputs: [Tensor<Float, CPU>], cpu: ([Tensor<Float, CPU>]) -> Result, gpu gpuBody: ([Tensor<Float, GPU>]) -> Result) -> Result {
        gpu ? gpuBody(inputs.map { Tensor<Float, GPU>($0, requiresGradient: $0.requiresGradient) }) : cpu(inputs)
    }

    /// Copies results of the GPU to the CPU.
    static func host(_ tensors: [Tensor<Float, GPU>]) -> [Tensor<Float, CPU>] {
        tensors.map { Tensor<Float, CPU>($0) }
    }
}
#endif

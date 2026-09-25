//
//  GPUMemoryTests.swift
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
import Synchronization
import Testing

extension GPUTests {
    @Suite(.serialized)
    struct GPUMemoryTests {
        /// Bytes of GPU memory that are allocated and not in the pool: the memory of live tensors, and of leaked buffers.
        private static var liveByteCount: Int {
            let context = GPUContext.current
            return context.device.currentAllocatedSize - context.cachedByteCount
        }

        /// Trains with batches of a different size in every step, on a thread whose autorelease pool is never drained,
        /// like the main thread of a command line tool, and returns the live bytes after every step.
        private static func trainOnThreadWithoutPoolDrain(steps: Int) -> [Int] {
            let result = Mutex<[Int]>([])
            let finished = DispatchSemaphore(value: 0)
            let thread = Thread {
                var generator = WyHash(seed: 5)
                var model = Sequential {
                    Dense<Float, GPU>(inputSize: 2048, outputSize: 256, using: &generator)
                    Relu<Float, GPU>()
                    Dense<Float, GPU>(inputSize: 256, outputSize: 10, using: &generator)
                    LogSoftmax<Float, GPU>()
                }
                var optimizer = Adam<Float, GPU>(learningRate: 0.001)
                var liveBytes: [Int] = []
                for step in 0 ..< steps {
                    let batchSize = 200 + (step * 37) % 300
                    let input = Tensor<Float, GPU>(uniformlyDistributedWithShape: [batchSize, 2048], min: -1, max: 1, using: &generator)
                    let labels = Tensor<Int32, GPU>((0 ..< batchSize).map { Int32($0 % 10) })
                    let loss = categoricalNegativeLogLikelihood(expected: labels, actual: model(input))
                    model.update { parameters in
                        optimizer.update(&parameters, along: loss.gradients(of: parameters))
                    }
                    // A read of the loss waits for the step, as a training loop that prints the loss does.
                    _ = loss.item
                    GPU.synchronize()
                    // Without the cache, a buffer stays allocated only while something references it.
                    GPU.clearCache()
                    liveBytes.append(liveByteCount)
                }
                result.withLock { $0 = liveBytes }
                finished.signal()
            }
            thread.start()
            finished.wait()
            return result.withLock { $0 }
        }

        @Test func trainingDoesNotLeakGPUMemory() {
            let liveBytes = Self.trainOnThreadWithoutPoolDrain(steps: 40)
            // The first steps allocate the parameters and the optimizer state, later steps must reuse or release their buffers.
            let growth = liveBytes[liveBytes.count - 1] - liveBytes[9]
            #expect(growth < 16 << 20, "The live GPU memory grew by \(growth >> 20) MiB in 30 steps.")
        }
    }
}
#endif

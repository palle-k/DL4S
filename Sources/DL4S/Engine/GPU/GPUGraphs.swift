//
//  GPUGraphs.swift
//  DL4S
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

#if canImport(Metal) && canImport(MetalPerformanceShaders) && canImport(MetalPerformanceShadersGraph)
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Synchronization

/// A compiled Metal Performance Shaders graph with the order of its inputs and results.
struct GPUGraph: @unchecked Sendable {
    // `@unchecked Sendable`: An executable does not change when it encodes, and the graph is not used after the compilation.

    let executable: MPSGraphExecutable
    /// Positions of the inputs of the executable in the order in which the caller passes them.
    let inputOrder: [Int]
    /// Positions of the results of the executable in the order in which the caller passes them.
    let resultOrder: [Int]

    /// Compiles a graph.
    ///
    /// - Parameter build: Creates the graph from its inputs and returns the results.
    init(inputShapes: [[Int]], build: (MPSGraph, [MPSGraphTensor]) -> [MPSGraphTensor]) {
        let graph = MPSGraph()
        let inputs = inputShapes.map { graph.placeholder(shape: $0.map { NSNumber(value: $0) }, dataType: .float32, name: nil) }
        let results = build(graph, inputs)
        var feeds: [MPSGraphTensor: MPSGraphShapedType] = [:]
        for input in inputs {
            feeds[input] = MPSGraphShapedType(shape: input.shape, dataType: .float32)
        }
        executable = graph.compile(with: MPSGraphDevice(mtlDevice: GPUContext.current.device), feeds: feeds, targetTensors: results, targetOperations: nil, compilationDescriptor: nil)
        let feedTensors = executable.feedTensors ?? []
        let targetTensors = executable.targetTensors ?? []
        inputOrder = inputs.map { input in feedTensors.firstIndex { $0 === input }! }
        resultOrder = results.map { result in targetTensors.firstIndex { $0 === result }! }
    }

    /// Records the graph.
    ///
    /// - Parameters:
    ///   - inputs: Buffers and shapes of the inputs, in the order of the inputs of the builder.
    ///   - results: Buffers and shapes of the results, in the order of the results of the builder.
    func encode(inputs: [(GPUBuffer, [Int])], results: [(GPUBuffer, [Int])]) {
        GPUContext.current.graph(reading: inputs.map(\.0), writing: results.map(\.0)) { commandBuffer in
            // The data objects are created while the stream is locked, because a new storage can still replace its buffer.
            var inputData = [MPSGraphTensorData?](repeating: nil, count: inputs.count)
            for (index, input) in inputs.enumerated() {
                inputData[inputOrder[index]] = Self.data(input.0, shape: input.1)
            }
            var resultData = [MPSGraphTensorData?](repeating: nil, count: results.count)
            for (index, result) in results.enumerated() {
                resultData[resultOrder[index]] = Self.data(result.0, shape: result.1)
            }
            let descriptor = MPSGraphExecutableExecutionDescriptor()
            descriptor.waitUntilCompleted = false
            executable.encode(to: commandBuffer, inputs: inputData.map { $0! }, results: resultData.map { $0! }, executionDescriptor: descriptor)
        }
    }

    private static func data(_ buffer: GPUBuffer, shape: [Int]) -> MPSGraphTensorData {
        let dimensions = shape.map { NSNumber(value: $0) }
        guard buffer.byteOffset != 0 else {
            return MPSGraphTensorData(buffer.storage.buffer, shape: dimensions, dataType: .float32)
        }
        let descriptor = MPSNDArrayDescriptor(dataType: .float32, shape: dimensions)
        return MPSGraphTensorData(MPSNDArray(buffer: buffer.storage.buffer, offset: buffer.byteOffset, descriptor: descriptor))
    }
}

/// Compiled graphs by a key that describes the operation and the shapes of its inputs.
enum GPUGraphCache {
    private struct Key: Hashable, @unchecked Sendable {
        // `@unchecked Sendable`: The wrapped key is `Sendable`, see ``GPUGraphCache/graph(for:compile:)``.
        let value: AnyHashable
    }

    private static let graphs = Mutex<[Key: GPUGraph]>([:])

    /// Returns the graph for the key, and compiles it when the cache has none.
    static func graph(for key: some Hashable & Sendable, compile: () -> GPUGraph) -> GPUGraph {
        let key = Key(value: AnyHashable(key))
        if let graph = graphs.withLock({ $0[key] }) {
            return graph
        }
        // The compilation runs without the lock. When two threads compile the same graph, the second result is kept.
        let graph = autoreleasepool(invoking: compile)
        graphs.withLock { $0[key] = graph }
        return graph
    }
}
#endif

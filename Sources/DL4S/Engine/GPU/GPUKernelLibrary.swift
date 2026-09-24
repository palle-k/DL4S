//
//  GPUKernelLibrary.swift
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

#if canImport(Metal) && canImport(MetalPerformanceShaders)
import Foundation
import Metal
import Synchronization

/// A group of kernels in Metal Shading Language, in a `.metal` file in the `Shaders` resource directory.
///
/// The package ships the kernels as source and compiles every group when one of its kernels is used first.
/// A package cannot rely on the Metal compiler at build time: it is not part of every toolchain.
struct GPUShaderSource: Sendable {
    /// Name of the `.metal` file without the extension.
    let name: String

    static let elementwise = GPUShaderSource(name: "elementwise")
    static let copy = GPUShaderSource(name: "copy")
    static let reduction = GPUShaderSource(name: "reduction")
    static let matrix = GPUShaderSource(name: "matrix")
    static let fused = GPUShaderSource(name: "fused")

    /// All kernel groups of the package.
    static var all: [GPUShaderSource] {
        [.elementwise, .copy, .reduction, .matrix, .fused]
    }

    /// Source of the group, after the declarations of `prelude.metal` that all groups share.
    func code() throws -> String {
        try Self.load("prelude") + "\n" + Self.load(name)
    }

    private static func load(_ name: String) throws -> String {
        guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders") else {
            throw GPUShaderError.missingSource(name: name)
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}

/// An error of the loading of the kernel sources.
enum GPUShaderError: Error, CustomStringConvertible {
    /// The resource bundle has no source file with the given name.
    case missingSource(name: String)

    var description: String {
        switch self {
        case let .missingSource(name):
            "The resources of DL4S contain no file Shaders/\(name).metal."
        }
    }
}

/// Compiles the kernel groups and keeps their pipeline states.
final class GPUKernelLibrary: @unchecked Sendable {
    // `@unchecked Sendable`: The mutable state is in a mutex. Metal devices, libraries, and pipeline states can be used from any thread.

    private struct State {
        var libraries: [String: any MTLLibrary] = [:]
        var pipelines: [String: any MTLComputePipelineState] = [:]
    }

    private let device: any MTLDevice
    private let state = Mutex(State())

    init(device: any MTLDevice) {
        self.device = device
    }

    /// Returns the pipeline state of the kernel with the given name in the given group.
    func pipeline(_ name: String, in source: GPUShaderSource) -> any MTLComputePipelineState {
        let key = source.name + "." + name
        return state.withLock { state in
            if let pipeline = state.pipelines[key] {
                return pipeline
            }
            let library = library(for: source, state: &state)
            guard let function = library.makeFunction(name: name) else {
                preconditionFailure("DL4S: The GPU kernel \(name) does not exist in \(source.name).")
            }
            do {
                let pipeline = try device.makeComputePipelineState(function: function)
                state.pipelines[key] = pipeline
                return pipeline
            } catch {
                preconditionFailure("DL4S: The pipeline of the GPU kernel \(name) could not be created: \(error)")
            }
        }
    }

    /// Compiles the given kernel group, or returns the error of the compiler.
    func compile(_ source: GPUShaderSource) -> (any Error)? {
        do {
            _ = try device.makeLibrary(source: source.code(), options: Self.compileOptions)
            return nil
        } catch {
            return error
        }
    }

    private static var compileOptions: MTLCompileOptions {
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        return options
    }

    private func library(for source: GPUShaderSource, state: inout State) -> any MTLLibrary {
        if let library = state.libraries[source.name] {
            return library
        }
        do {
            let library = try device.makeLibrary(source: source.code(), options: Self.compileOptions)
            state.libraries[source.name] = library
            return library
        } catch {
            preconditionFailure("DL4S: The GPU kernels \(source.name) could not be compiled: \(error)")
        }
    }
}

/// Element types that have GPU kernels.
enum GPUElement: String {
    case float
    case int

    /// The GPU element type of `N`, or nil when the GPU has no kernels for it.
    init?<N>(of _: N.Type) {
        if N.self == Float.self {
            self = .float
        } else if N.self == Int32.self {
            self = .int
        } else {
            return nil
        }
    }
}

/// Decides whether an operation runs on the host or on the GPU.
enum GPUPlacement {
    /// Whether an operation with the given number of result elements runs on the host.
    ///
    /// Small operations run on the host when the host can access all operands without waiting for the GPU.
    /// Then the host computes the result in less time than it takes to record a GPU command, and the result is available on the host.
    static func runsOnHost(elements: Int, reading: [GPUBuffer], writing: [GPUBuffer]) -> Bool {
        let context = GPUContext.current
        guard elements <= context.hostExecutionLimit else {
            return false
        }
        return context.isHostAccessible(reading: reading, writing: writing)
    }
}

/// Builds the layout of a strided region for up to three operands and merges axes that are contiguous in all operands.
struct GPULayout {
    var shape: [Int]
    var strides: [[Int]]

    init(shape: [Int], strides: [[Int]]) {
        precondition(strides.count <= 3, "A layout has at most three operands.")
        var mergedShape: [Int] = []
        var mergedStrides = [[Int]](repeating: [], count: strides.count)
        for axis in shape.indices where shape[axis] != 1 {
            // An axis merges into the axis before it when the stride of that axis spans the axis in every operand.
            let isContiguous = !mergedShape.isEmpty && strides.indices.allSatisfy { operand in
                mergedStrides[operand][mergedStrides[operand].count - 1] == strides[operand][axis] * shape[axis]
            }
            if isContiguous {
                mergedShape[mergedShape.count - 1] *= shape[axis]
                for operand in strides.indices {
                    mergedStrides[operand][mergedStrides[operand].count - 1] = strides[operand][axis]
                }
            } else {
                mergedShape.append(shape[axis])
                for operand in strides.indices {
                    mergedStrides[operand].append(strides[operand][axis])
                }
            }
        }
        self.shape = mergedShape
        self.strides = mergedStrides
    }

    /// Whether the kernels support the layout.
    var isSupported: Bool {
        shape.count <= 8
    }

    /// The layout in the memory layout of the `Layout` struct of the kernels.
    var arguments: [Int32] {
        var values = [Int32](repeating: 0, count: 1 + 8 + 3 * 8)
        values[0] = Int32(shape.count)
        for axis in shape.indices {
            values[1 + axis] = Int32(shape[axis])
            for operand in strides.indices {
                values[9 + operand * 8 + axis] = Int32(strides[operand][axis])
            }
        }
        return values
    }

    /// Row-major strides of a shape.
    static func contiguousStrides(_ shape: [Int]) -> [Int] {
        CPUMemoryOperators.strides(from: shape)
    }
}
#endif

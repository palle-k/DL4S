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

// A package cannot rely on the Metal compiler at build time: it is not part of every toolchain.
/// A group of kernels in Metal Shading Language, in a `.metal` file in the `Shaders` resource directory.
///
/// The package ships the kernels as source and compiles every group when one of its kernels is used first.
struct GPUShaderSource: Sendable {
    /// Name of the `.metal` file without the extension.
    let name: String

    static let elementwise = GPUShaderSource(name: "elementwise")
    static let copy = GPUShaderSource(name: "copy")
    static let reduction = GPUShaderSource(name: "reduction")
    static let matrix = GPUShaderSource(name: "matrix")
    static let fused = GPUShaderSource(name: "fused")
    static let convolution = GPUShaderSource(name: "convolution")
    static let attention = GPUShaderSource(name: "attention")

    /// All kernel groups of the package.
    static var all: [GPUShaderSource] {
        [.elementwise, .copy, .reduction, .matrix, .fused, .convolution, .attention]
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
        return autoreleasepool {
            state.withLock { state in
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
    }

    /// Compiles the given kernel group and throws the error of the compiler.
    func compile(_ source: GPUShaderSource) throws {
        _ = try device.makeLibrary(source: source.code(), options: Self.compileOptions)
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

#endif

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
enum GPUShaderSource: String, CaseIterable, Sendable {
    case elementwise
    case copy
    case reduction
    case matrix
    case fused
    case convolution
    case attention

    /// Name of the `.metal` file without the extension.
    var name: String {
        rawValue
    }

    /// All kernel groups of the package.
    static var all: [GPUShaderSource] {
        allCases
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

/// A kernel: a function of a kernel group.
struct GPUKernel: Hashable, Sendable {
    /// Name of the function.
    let name: String
    /// Group that contains the function.
    let source: GPUShaderSource
}

/// Compiles the kernel groups and keeps their pipeline states.
final class GPUKernelLibrary: Sendable {
    // The compilations run without the locks, so that the lookups of other threads do not wait for them. Two threads can
    // compile the same group or pipeline at the same time, and the first result is kept.

    private let device: any MTLDevice
    private let libraries = Mutex<[GPUShaderSource: any MTLLibrary]>([:])
    private let pipelines = Mutex<[GPUKernel: any MTLComputePipelineState]>([:])

    init(device: any MTLDevice) {
        self.device = device
    }

    /// Returns the pipeline state of the kernel.
    func pipeline(_ kernel: GPUKernel) -> any MTLComputePipelineState {
        if let pipeline = pipelines.withLock({ $0[kernel] }) {
            return pipeline
        }
        // The compilation autoreleases objects, see ``GPUContext``.
        let pipeline = autoreleasepool { makePipeline(kernel) }
        return pipelines.withLock { pipelines in
            if let existing = pipelines[kernel] {
                return existing
            }
            pipelines[kernel] = pipeline
            return pipeline
        }
    }

    private func makePipeline(_ kernel: GPUKernel) -> any MTLComputePipelineState {
        guard let function = library(for: kernel.source).makeFunction(name: kernel.name) else {
            preconditionFailure("DL4S: The GPU kernel \(kernel.name) does not exist in \(kernel.source.name).")
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            preconditionFailure("DL4S: The pipeline of the GPU kernel \(kernel.name) could not be created: \(error)")
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

    private func library(for source: GPUShaderSource) -> any MTLLibrary {
        if let library = libraries.withLock({ $0[source] }) {
            return library
        }
        do {
            let library = try device.makeLibrary(source: source.code(), options: Self.compileOptions)
            return libraries.withLock { libraries in
                if let existing = libraries[source] {
                    return existing
                }
                libraries[source] = library
                return library
            }
        } catch {
            preconditionFailure("DL4S: The GPU kernels \(source.name) could not be compiled: \(error)")
        }
    }
}

#endif

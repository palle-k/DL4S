//
//  GPUPlacement.swift
//  DL4S
//
//  Created by Palle Klewitz on 30.09.26.
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
    // The host then computes the result in less time than it takes to record a GPU command, and the result is available on the host.
    /// Whether an operation with the given number of result elements runs on the host: small operations whose operands the
    /// host can access without waiting for the GPU.
    static func runsOnHost(elements: Int, reading: [GPUBuffer], writing: [GPUBuffer]) -> Bool {
        let context = GPUContext.current
        guard elements <= context.hostExecutionLimit else {
            return false
        }
        return context.isHostAccessible(reading: reading, writing: writing)
    }
}
#endif

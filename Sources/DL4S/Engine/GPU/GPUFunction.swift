//
//  GPUFunction.swift
//  DL4S
//
//  Created by Palle Klewitz on 24.10.19.
//  Copyright (c) 2019 - Palle Klewitz
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

#if canImport(Metal)
import Foundation
import Metal
import MetalPerformanceShaders


extension MTLComputeCommandEncoder {
    
    /// Dispatches a compute shader with the given global size.
    ///
    /// The size will be divided into thread groups are as large as possible
    /// for the device used.
    ///
    /// This may lead to kernel invokations, which are outside of
    /// the given global size.
    ///
    /// - Parameter maxSize: Global size
    func dispatch(workSize maxSize: (width: Int, height: Int, depth: Int)) {
        let maxDeviceSize = self.device.maxThreadsPerThreadgroup
        
        var size = MTLSize(
            width: min(maxSize.width, maxDeviceSize.width),
            height: min(maxSize.height, maxDeviceSize.height),
            depth: min(maxSize.depth, maxDeviceSize.depth)
        )
        
        while size.width * size.height * size.depth > max(maxDeviceSize.width, maxDeviceSize.height, maxDeviceSize.depth) {
            // Shrink the largest size first, begin with depth
            // If there is no largest size, shrink anyway, begin with depth
            
            if size.depth > size.width && size.depth > size.height {
                size.depth /= 2
            } else if size.height > size.width && size.height > size.depth {
                size.height /= 2
            } else if size.width > size.height && size.width > size.depth {
                size.width /= 2
            } else if size.depth >= 2 {
                size.depth /= 2
            } else if size.height >= 2 {
                size.height /= 2
            } else if size.width >= 2 {
                size.width /= 2
            } else {
                fatalError("Cannot determine dispatch threadgroup size")
            }
        }
        
//        let threadGroups = MTLSize(
//            width:  (maxSize.width  + size.width  - 1) / size.width,
//            height: (maxSize.height + size.height - 1) / size.height,
//            depth:  (maxSize.depth  + size.depth  - 1) / size.depth
//        )
        let threadgroupsPerGrid = MTLSize(width: size.width, height: size.height, depth: size.depth)
        
        self.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: size)
    }
    
}

struct GPUFunction {
    var device: MTLDevice
    var function: MTLFunction
    
    private func makePipelineState() -> MTLComputePipelineState {
        do {
            return try device.makeComputePipelineState(function: function)
        } catch let error {
            fatalError("Could not make pipeline for function (\(error))")
        }
    }
    
    private func execute(workSize: (width: Int, height: Int, depth: Int), _ args: [GPUArgument]) {
        let buffer = GPU.currentCommandBuffer
        let encoder = buffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(makePipelineState())
        
        var offset = 0
        for arg in args {
            arg.add(to: encoder, from: offset)
            offset += arg.slotCount
        }
        
        encoder.dispatch(workSize: workSize)
        encoder.endEncoding()
    }
}

extension GPUFunction {
    func execute(workSize: (width: Int, height: Int, depth: Int), _ args: GPUArgument...) {
        execute(workSize: workSize, args)
    }
    
    func callAsFunction(workSize: (width: Int, height: Int, depth: Int), _ args: GPUArgument...) {
        execute(workSize: workSize, args)
    }
}

#endif

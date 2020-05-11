//
//  CPUEngine.swift
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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

private class BundleIdentifyingClass: NSObject {}

@dynamicMemberLookup
public struct GPU: DeviceType {
    public typealias Memory = VRAMAllocator
    public typealias Engine = GPUEngine
    
    static let device: MTLDevice = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not available.")
        }
        print("Using \(device.name)")
        print("\(device.currentAllocatedSize)B of \(device.recommendedMaxWorkingSetSize)B in use.")
        return device
    }()
    
    static let library: MTLLibrary = {
        do {
            var options = MTLCompileOptions()
            options.fastMathEnabled = true
            options.languageVersion = .version2_2
            return try GPU.device.makeLibrary(source: SHADER_SOURCE, options: options)
        } catch let error {
            fatalError("Cannot get metal library (\(error))")
        }
    }()

    private static var functionCache: [String: GPUFunction] = [:]
    
    static func function(named functionName: String) -> GPUFunction {
        if let cached = functionCache[functionName] {
            return cached
        }
        
        guard let function = library.makeFunction(name: functionName) else {
            fatalError("Could not obtain function named '\(functionName)'")
        }
        let fn = GPUFunction(device: device, function: function)
        functionCache[functionName] = fn
        return fn
    }
    
    static var commandQueue: MTLCommandQueue = {
        guard let queue = device.makeCommandQueue() else {
            fatalError("Could not make command queue")
        }
        return queue
    }()
    
    static var currentCommandBuffer: MTLCommandBuffer = {
       commandQueue.makeCommandBuffer()!
    }()
    
    static func synchronize() {
        currentCommandBuffer.commit()
        currentCommandBuffer.waitUntilCompleted()
        currentCommandBuffer = commandQueue.makeCommandBuffer()!
    }
    
    static subscript (dynamicMember memberName: String) -> GPUFunction {
        get {
            return self.function(named: memberName)
        }
    }
}

#endif

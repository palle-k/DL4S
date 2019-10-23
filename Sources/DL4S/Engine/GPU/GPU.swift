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
        print("Running on \(device.name)")
        return device
    }()
    
    static let library: MTLLibrary = {
        do {
            return try GPU.device.makeDefaultLibrary(bundle: Bundle(for: BundleIdentifyingClass.self))
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

protocol GPUArgument {
    var slotCount: Int { get }
    func add(to encoder: MTLComputeCommandEncoder, from index: Int)
}

extension ShapedBuffer: GPUArgument where Device == GPU {
    var slotCount: Int {
        return 3
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        let valueBuffer = self.values.memory.buffer
        
        encoder.setBuffer(valueBuffer, offset: 0, index: index)
        encoder.setBytes([Int32(dim)], length: MemoryLayout<Int32>.size, index: index + 1)
        encoder.setBytes(shape.map(Int32.init), length: shape.count * MemoryLayout<Int32>.stride, index: index + 2)
    }
}

extension Buffer: GPUArgument where Device == GPU {
    var slotCount: Int {
        return 2
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        let valueBuffer = self.memory.buffer
        
        encoder.setBuffer(valueBuffer, offset: 0, index: index)
        encoder.setBytes([Int32(count)], length: MemoryLayout<Int32>.size, index: index + 1)
    }
}

extension Float: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Float>.size, index: index)
    }
}

extension Double: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Double>.size, index: index)
    }
}

extension Int32: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([self], length: MemoryLayout<Int32>.size, index: index)
    }
}

extension Int: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes([Int32(self)], length: MemoryLayout<Int32>.size, index: index)
    }
}

extension VRAMBuffer: GPUArgument {
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBuffer(buffer, offset: self.offset, index: index)
    }
}

struct ArrayValuesArgument<Element>: GPUArgument {
    var array: [Element]
    
    var slotCount: Int {
        return 1
    }
    
    func add(to encoder: MTLComputeCommandEncoder, from index: Int) {
        encoder.setBytes(array, length: MemoryLayout<Element>.stride * array.count, index: index)
    }
}

extension Array where Element == Int {
    var valueArg: ArrayValuesArgument<Int32> {
        ArrayValuesArgument(array: self.map(Int32.init))
    }
}

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
        
        let threadGroups = MTLSize(
            width:  (maxSize.width  + size.width  - 1) / size.width,
            height: (maxSize.height + size.height - 1) / size.height,
            depth:  (maxSize.depth  + size.depth  - 1) / size.depth
        )
        
        self.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: size)
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

public struct VRAMAllocator: MemoryOperatorsType {
    public typealias RawBuffer = VRAMBuffer
    public typealias Device = GPU
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, GPU> {
        let stride = MemoryLayout<Element>.stride
        guard let buffer = Device.device.makeBuffer(length: stride * capacity, options: .storageModeShared) else {
            fatalError("Could not allocate memory")
        }
        return Buffer<Element, GPU>(memory: VRAMBuffer(buffer: buffer, offset: 0))
    }
    
    public static func allocateBuffer<Element>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, GPU> {
        let count = shape.reduce(1, *)
        return ShapedBuffer(values: allocateBuffer(withCapacity: count, type: Element.self), shape: shape)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, GPU>) {
        // Noop, MTLBuffer is reference counted
        
    }
    
    public static func free<Element>(_ buffer: ShapedBuffer<Element, GPU>) {
        // Noop, MTLBuffer is reference counted
    }
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, GPU>, count: Int) {
        destination.memory.buffer.contents().initializeMemory(as: Element.self, from: source.pointer(capacity: count), count: count)
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: Buffer<Element, GPU>, count: Int) {
        let buffer = GPU.currentCommandBuffer
        let encoder = buffer.makeBlitCommandEncoder()!
        encoder.copy(from: source.memory.buffer, sourceOffset: source.memory.offset, to: destination.memory.buffer, destinationOffset: destination.memory.offset, size: MemoryLayout<Element>.stride * count)
        encoder.endEncoding()
    }
    
    public static func assign<Element>(from source: Buffer<Element, GPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        GPU.synchronize()
        destination.assign(
            from: UnsafeBufferPointer(
                start: source.memory.buffer.contents()
                    .advanced(by: source.memory.offset)
                    .assumingMemoryBound(to: Element.self),
                count: count
            ),
            count: count
        )
    }
    
    public static func getValue<Element>(from source: Buffer<Element, GPU>) -> Element {
        GPU.synchronize()
        return source.memory.buffer.contents()
            .advanced(by: source.memory.offset)
            .assumingMemoryBound(to: Element.self).pointee
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, GPU>) -> Int {
        return buffer.memory.buffer.length / MemoryLayout<Element>.stride
    }
    
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        // Prevent unneccessary copies when index ends with nil
        let slice = slice.reversed().drop(while: {$0 == nil}).reversed()
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = CPUMemoryOperators.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            let rawOffset = MemoryLayout<Element>.stride * offset
            let advanced = buffer.memory.advanced(by: rawOffset)
            return (Buffer<Element, GPU>(memory: advanced), false, Array(shape.dropFirst(nonNilIndices.count)))
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
            
            fatalError("TODO")
            //iterativeRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
        
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        fatalError("TODO")
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        fatalError("TODO")
    }
    
    public static func advance<Element>(buffer: Buffer<Element, GPU>, by advancement: Int) -> Buffer<Element, GPU> {
        Buffer(memory: buffer.memory.advanced(by: MemoryLayout<Element>.stride * advancement))
    }
    
    public static func setPointee<Element>(of buffer: Buffer<Element, GPU>, to newValue: Element) {
        fatalError("TODO")
    }
}


public struct VRAMBuffer: Hashable {
    public static func == (lhs: VRAMBuffer, rhs: VRAMBuffer) -> Bool {
        return lhs.buffer.hash == rhs.buffer.hash && lhs.offset == rhs.offset
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(buffer.hash)
        hasher.combine(offset)
    }
    
    var buffer: MTLBuffer
    var offset: Int
    
    func advanced(by offset: Int) -> VRAMBuffer {
        VRAMBuffer(buffer: buffer, offset: self.offset + offset)
    }
}


public struct GPUEngine: EngineType {
    public typealias Device = GPU

    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int) {
        guard let value = value as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vFill_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), value, result)
    }

    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, stride: Int, count: Int) {
        guard let value = value as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vFillStride_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), value, result, stride)
    }

    public static func transpose<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, srcRows: Int, srcCols: Int) {
        GPU.function(named: "mTrans_\(N.gpuTypeIdentifier)").execute(workSize: (srcCols, srcRows, 1), val.memory, srcCols, srcRows, result.memory)
    }

    public static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vAdd_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, result.memory)
    }

    public static func vsAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        guard let rhs = rhs as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vsAdd_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs, result.memory)
    }
    
    public static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vNeg_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), val.memory, result.memory)
    }
    
    public static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vSub_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, result.memory)
    }
    
    public static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vMul_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, result.memory)
    }
    
    public static func vMA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vMA_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, add.memory, result.memory)
    }
    
    public static func vsMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        guard let rhs = rhs as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vsMul_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs, result.memory)
    }
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vDiv_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, result.memory)
    }
    
    public static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        guard let lhs = lhs as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vsMul_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs, rhs.memory, result.memory)
    }
    
    public static func vSquare<N: NumericType>(values: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vSquare_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), values.memory, result.memory)
    }
    
    public static func matMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        matMul(
            lhs: ShapedBuffer(values: lhs, shape: [lhsRows, lhsCols]),
            rhs: ShapedBuffer(values: rhs, shape: [lhsCols, rhsCols]),
            result: ShapedBuffer(values: result, shape: [lhsRows, rhsCols])
        )
    }
    
    public static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        let function = MPSMatrixMultiplication(
            device: GPU.device,
            transposeLeft: transposeFirst,
            transposeRight: transposeSecond,
            resultRows: resultShape.0,
            resultColumns: resultShape.1,
            interiorColumns: transposeFirst ? lhsShape.0 : lhsShape.1,
            alpha: 1,
            beta: 1
        )
        
        let cmdBuffer = GPU.currentCommandBuffer
        function.encode(
            commandBuffer: cmdBuffer,
            leftMatrix: MPSMatrix(
                buffer: lhs.memory.buffer,
                offset: lhs.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: lhsShape.0, columns: lhsShape.1, rowBytes: lhsShape.1 * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            ),
            rightMatrix: MPSMatrix(
                buffer: rhs.memory.buffer,
                offset: rhs.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: rhsShape.0, columns: rhsShape.1, rowBytes: rhsShape.1 * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            ),
            resultMatrix: MPSMatrix(
                buffer: result.memory.buffer,
                offset: result.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: resultShape.0, columns: resultShape.1, rowBytes: resultShape.1 * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            )
        )
    }
    
    public static func dot<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, count: Int) -> N {
        let result = GPU.Memory.allocateBuffer(withCapacity: 1, type: N.self)
        matMul(
            lhs: ShapedBuffer(values: lhs, shape: [1, count]),
            rhs: ShapedBuffer(values: rhs, shape: [count, 1]),
            result: ShapedBuffer(values: result, shape: [1, 1])
        )
        return result.pointee
    }
    
    public static func vMulSA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: N, result: Buffer<N, Device>, count: Int) {
        guard let add = add as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vMulSA_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs.memory, add, result.memory)
    }
    
    public static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        guard let rhs = rhs as? GPUArgument else {
            fatalError("Cannot use value of type \(N.self) as an argument.")
        }
        GPU.function(named: "vsMulVAdd_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs.memory, rhs, add.memory, result.memory)
    }
    
    public static func log<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vLog_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), val.memory, result.memory)
    }
    
    public static func exp<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vExp_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), val.memory, result.memory)
    }
    
    public static func relu<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vRelu_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func isPositive<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vHeaviside_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func tanh<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vTanh_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vSqrt_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, Device>, count: Int) -> N {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func gemm<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastGemm<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N) {
        // fatalError("\(#function) not available for GPU")
        GPU.synchronize()
        return CPU.Engine.argmax(
            values: Buffer<N, CPU>(
                memory: UnsafeMutableRawBufferPointer(
                    start: values.memory.buffer.contents().advanced(by: values.memory.offset),
                    count: count * MemoryLayout<N>.stride
                )
            ),
            count: count
        )
    }
    
    public static func conv2d<N>(input: Buffer<N, GPU>, filter: Buffer<N, GPU>, result: Buffer<N, GPU>, width: Int, height: Int, batchSize: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxes<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], destination: Buffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxesAdd<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], add: Buffer<N, GPU>, destination: Buffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }

    public static func matMul<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "mMul_\(N.gpuTypeIdentifier)").execute(workSize: (result.shape[1], result.shape[0], 1), lhs, rhs, result)
    }
    
    public static func matMulAdd<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func broadcastAdd<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vAdd_Broadcast_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs, rhs, result)
    }
    
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vSub_Broadcast_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs, rhs, result)
    }
    
    public static func broadcastMul<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vMul_Broadcast_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs, rhs, result)
    }
    
    public static func broadcastDiv<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vDiv_Broadcast_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), lhs, rhs, result)
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        GPU.function(named: "vSum_Reduce_\(N.gpuTypeIdentifier)").execute(
            workSize: (result.count, 1, 1),
            values,
            CPU.Memory.strides(from: values.shape).valueArg,
            result,
            CPU.Memory.strides(from: result.shape).valueArg,
            axis
        )
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        if let context = context {
            GPU.function(named: "vMax_ReduceCtx_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                CPU.Memory.strides(from: result.shape).valueArg,
                context,
                axis
            )
        } else {
            GPU.function(named: "vMax_Reduce_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                CPU.Memory.strides(from: result.shape).valueArg,
                axis
            )
        }
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        if let context = context {
            GPU.function(named: "vMin_ReduceCtx_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                CPU.Memory.strides(from: result.shape).valueArg,
                context,
                axis
            )
        } else {
            GPU.function(named: "vMin_Reduce_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                CPU.Memory.strides(from: result.shape).valueArg,
                axis
            )
        }
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        if axes.isEmpty {
            GPU.Memory.assign(from: values.values, to: result.values, count: values.count)
            return
        } else if axes.count == 1 {
            GPU.function(named: "vSum_Reduce_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                CPU.Memory.strides(from: result.shape).valueArg,
                axes[0]
            )
            return
        }
        fatalError("Reduce not available on GPU with more than 1 axis")
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func expandContext<N: NumericType>(reduced: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) {
        GPU.function(named: "vScatter_AddInPlace_\(N.gpuTypeIdentifier)").execute(
            workSize: (reduced.count, 1, 1),
            reduced,
            CPU.Memory.strides(from: reduced.shape).valueArg,
            reduced,
            result,
            CPU.Memory.strides(from: result.shape).valueArg,
            axis
        )
    }
    
    public static func sum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func mean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func max<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func min<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func square<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vSquare_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func relu<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vRelu_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func heaviside<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N: NumericType {
        GPU.function(named: "permuteAxes_\(N.gpuTypeIdentifier)").execute(
            workSize: (values.count, 1, 1),
            values,
            result,
            arangement.valueArg,
            CPU.Memory.strides(from: values.shape).valueArg
        )
    }
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N : NumericType {
        GPU.function(named: "permuteAxesAdd_\(N.gpuTypeIdentifier)").execute(
            workSize: (values.count, 1, 1),
            values,
            add,
            result,
            arangement.valueArg,
            CPU.Memory.strides(from: values.shape).valueArg
        )
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptReadAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func subscriptWriteAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reverse<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func reverseAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, GPU>], result: ShapedBuffer<N, GPU>, axis: Int) {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func unstackAdd<N>(stacked: ShapedBuffer<N, GPU>, add: [ShapedBuffer<N, GPU>], result: [ShapedBuffer<N, GPU>], axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func unstack<N>(stacked: ShapedBuffer<N, GPU>, result: [ShapedBuffer<N, GPU>], axis: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func arange<N>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func img2col<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func col2img<N>(matrix: ShapedBuffer<N, GPU>, image: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func exp<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vExp_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func log<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vLog_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }

    public static func sqrt<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vSqrt_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func sin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vSin_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func cos<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vCos_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func tan<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vTan_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func sinh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vSinh_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func cosh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vCosh_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func tanh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        GPU.function(named: "vTanh_\(N.gpuTypeIdentifier)").execute(workSize: (result.count, 1, 1), values.values.memory, result.values.memory)
    }
}

#endif

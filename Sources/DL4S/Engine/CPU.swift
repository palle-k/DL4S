//
//  CPU.swift
//  DL4S
//
//  Created by Palle Klewitz on 11.03.19.
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

import Foundation
import Accelerate


public struct CPU: DeviceType {
    public typealias Memory = CPUMemoryOperators
    public typealias Engine = CPUEngine
}

public struct CPUMemoryOperators: MemoryOperatorsType {
    public typealias RawBuffer = UnsafeMutableRawBufferPointer
    public typealias Device = CPU
    
    static func strides(from shape: [Int]) -> [Int] {
        let dim = shape.count
        
        if dim == 0 {
            return []
        }
        
        var str = [Int](repeating: 1, count: dim)
        for i in (0 ..< dim - 1).reversed() {
            str[i] = str[i + 1] * shape[i + 1]
        }
        return str
    }
    
    static func linearIndex(from index: [Int], shape: [Int]) -> Int {
        let strides = CPUMemoryOperators.strides(from: shape)
        return zip(index, strides).map(*).reduce(0, +)
    }
    
    static func index(from linearIndex: Int, shape: [Int]) -> [Int] {
        let strides = CPUMemoryOperators.strides(from: shape)
        return zip(shape, strides).map { dim, str in (linearIndex / str) % dim}
    }
    
    public static func allocateBuffer<Element>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, CPU> where Element : NumericType {
        let stride = MemoryLayout<Element>.stride
        let alignment = max(MemoryLayout<Element>.alignment, 16)
        
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: stride * capacity, alignment: alignment)
        
        return Buffer<Element, CPU>(memory: buffer)
    }
    
    public static func free<Element>(_ buffer: Buffer<Element, CPU>) where Element : NumericType {
        buffer.memory.deallocate()
    }
    
    
    public static func assign<Element>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, CPU>, count: Int) where Element : NumericType {
        destination.memory.bindMemory(to: Element.self).assign(from: source, count: count)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: Buffer<Element, CPU>, count: Int) where Element : NumericType {
        destination.memory.bindMemory(to: Element.self).assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
    }
    
    public static func assign<Element>(from source: Buffer<Element, CPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) where Element : NumericType {
        destination.assign(from: source.memory.bindMemory(to: Element.self).immutable, count: count)
    }
    
    public static func get<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) where Element : NumericType {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let nonNilIndices = slice.compactMap {$0}
        let strides = CPUMemoryOperators.strides(from: shape)
        
        if nonNilIndices.count == slice.count {
            // Simple offset into storage
            let offset = zip(nonNilIndices, strides).map(*).reduce(0, +)
            let advanced = buffer.memory.bindMemory(to: Element.self).advanced(by: offset)
            let advancedRaw = UnsafeMutableRawBufferPointer(advanced)
            return (Buffer<Element, CPU>(memory: advancedRaw), false, Array(shape.dropFirst(nonNilIndices.count)))
        } else {
            let padded = slice + [Int?](repeating: nil, count: shape.count - slice.count)
            
            let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
                let (index, dimSize) = el
                return index == nil ? dimSize : nil
            }
            let flattenedResultShape = resultShape.compactMap {$0}
            
            let resultCount = flattenedResultShape.reduce(1, *)
            let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
            
            recursiveRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
            
            return (resultBuffer, true, flattenedResultShape)
        }
    }
    
    public static func get<Element>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, CPU>, with shape: [Int]) -> (Buffer<Element, CPU>, Bool, [Int]) where Element : NumericType {
        precondition(slice.count <= shape.count, "Index must be smaller than or equal to vector size")
        
        let strides = CPUMemoryOperators.strides(from: shape)
        
        let padded = slice + [Range<Int>?](repeating: nil, count: shape.count - slice.count)
        
        let resultShape = zip(padded, shape).enumerated().map { idx, el -> Int? in
            let (index, dimSize) = el
            return index.map {$0.count} ?? dimSize
        }
        let flattenedResultShape = resultShape.compactMap {$0}
        
        let resultCount = flattenedResultShape.reduce(1, *)
        let resultBuffer = allocateBuffer(withCapacity: resultCount, type: Element.self)
        
        recursiveRead(source: buffer.memory.bindMemory(to: Element.self).immutable, destination: resultBuffer.memory.bindMemory(to: Element.self), srcIndex: padded, srcStrides: strides, srcShape: shape)
        
        return (resultBuffer, true, flattenedResultShape)
    }
    
    public static func set<Element>(slice: [Int?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) where Element : NumericType {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Int?](repeating: nil, count: dstShape.count - slice.count)
        
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        recursiveWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func set<Element>(slice: [Range<Int>?], of buffer: Buffer<Element, CPU>, with dstShape: [Int], from source: Buffer<Element, CPU>, with sourceShape: [Int]) where Element : NumericType {
        precondition(sourceShape.count == dstShape.count - slice.filter {$0 != nil}.count, "Shape of source must be equal to source of destination minus number of knowns in slice")
        
        let padded = slice + [Range<Int>?](repeating: nil, count: dstShape.count - slice.count)
        let dstStrides = CPUMemoryOperators.strides(from: dstShape)
        
        recursiveWrite(source: source.memory.bindMemory(to: Element.self).immutable, destination: buffer.memory.bindMemory(to: Element.self), dstIndex: padded, dstStrides: dstStrides, dstShape: dstShape)
    }
    
    public static func getValue<Element>(from source: Buffer<Element, CPU>) -> Element where Element : NumericType {
        return source.memory.bindMemory(to: Element.self).pointee
    }
    
    public static func getSize<Element>(of buffer: Buffer<Element, CPU>) -> Int where Element : NumericType {
        return buffer.memory.bindMemory(to: Element.self).count
    }
    
    public static func advance<Element>(buffer: Buffer<Element, CPU>, by advancement: Int) -> Buffer<Element, CPU> where Element : NumericType {
        return Buffer<Element, CPU>(
            memory: UnsafeMutableRawBufferPointer(
                buffer.memory
                    .bindMemory(to: Element.self)
                    .advanced(by: advancement)
            )
        )
    }
}

public struct CPUEngine: EngineType {
    public typealias Device = CPU
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, stride: Int, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func transpose<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, srcRows: Int, srcCols: Int) {
        N.transpose(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), srcRows: srcRows, srcCols: srcCols)
    }
    
    public static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        N.vsAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vNeg<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vNeg(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vSub<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vSub(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vMA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vMA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, result: Buffer<N, Device>, count: Int) {
        N.vsMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vDiv(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func svDiv<N: NumericType>(lhs: N, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.svDiv(lhs: lhs, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vSquare<N: NumericType>(values: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vSquare(values: values.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func matMul<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsRows: Int, lhsCols: Int, rhsCols: Int) {
        N.matMul(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsRows: lhsRows, lhsCols: lhsCols, rhsCols: rhsCols)
    }
    
    public static func matMulAddInPlace<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, lhsShape: (Int, Int), rhsShape: (Int, Int), resultShape: (Int, Int), transposeFirst: Bool, transposeSecond: Bool) {
        N.matMulAddInPlace(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), lhsShape: lhsShape, rhsShape: rhsShape, resultShape: resultShape, transposeFirst: transposeFirst, transposeSecond: transposeSecond)
    }
    
    public static func dot<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, count: Int) -> N {
        return N.dot(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func vMulSA<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, add: N, result: Buffer<N, Device>, count: Int) {
        N.vMulSA(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, add: add, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vsMulVAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: N, add: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vsMulVAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs, add: add.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func log<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.log(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func exp<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.exp(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func relu<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.relu(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func isPositive<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        // Use non-Buffer pointer to avoid repeated bound checking, bounds are checked initially
        let src = val.memory.bindMemory(to: N.self).pointer(capacity: count)
        let dst = result.memory.bindMemory(to: N.self).pointer(capacity: count)
        
        for i in 0 ..< count {
            dst[i] = src[i] > 0 ? 1 : 0
        }
    }
    
    public static func tanh<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.tanh(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.sqrt(val: val.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, Device>, count: Int) -> N {
        return N.sum(val: val.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func copysign<N: NumericType>(values: Buffer<N, Device>, signs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.copysign(values: values.memory.bindMemory(to: N.self).immutable, signs: signs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N) {
        return N.argmax(values: values.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    public static func conv2d<N: NumericType>(input: Buffer<N, Device>, filter: Buffer<N, Device>, result: Buffer<N, Device>, width: Int, height: Int, kernelWidth: Int, kernelHeight: Int, kernelDepth: Int, kernelCount: Int) {
        N.conv2d(input: input.memory.bindMemory(to: N.self).immutable, filter: filter.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), width: width, height: height, kernelWidth: kernelWidth, kernelHeight: kernelHeight, kernelDepth: kernelDepth, kernelCount: kernelCount)
    }
    
    public static func permuteAxes<N>(input: Buffer<N, CPU>, arangement: [Int], shape: [Int], destination: Buffer<N, CPU>) where N : NumericType {
        let sourceMem = input.memory.bindMemory(to: N.self)
        let dstMem = destination.memory.bindMemory(to: N.self)
        let dim = shape.count
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[arangement[i]] = shape[i]
        }
        
        for index in iterate(shape) {
            var dstIdx = [Int](repeating: 0, count: dim)
            for i in dstIdx.indices {
                dstIdx[arangement[i]] = index[i]
            }
            
            let lsIdx = CPUMemoryOperators.linearIndex(from: index, shape: shape)
            let ldIdx = CPUMemoryOperators.linearIndex(from: dstIdx, shape: dstShape)
            
            dstMem[ldIdx] = sourceMem[lsIdx]
        }
    }
    
    public static func permuteAxesAdd<N>(input: Buffer<N, CPU>, arangement: [Int], shape: [Int], add: Buffer<N, CPU>, destination: Buffer<N, CPU>) where N : NumericType {
        let sourceMem = input.memory.bindMemory(to: N.self)
        let addMem = add.memory.bindMemory(to: N.self)
        let dstMem = destination.memory.bindMemory(to: N.self)
        let dim = shape.count
        
        var dstShape = [Int](repeating: 0, count: dim)
        
        for i in dstShape.indices {
            dstShape[arangement[i]] = shape[i]
        }
        
        for index in iterate(shape) {
            var dstIdx = [Int](repeating: 0, count: dim)
            for i in dstIdx.indices {
                dstIdx[arangement[i]] = index[i]
            }
            
            let lsIdx = CPUMemoryOperators.linearIndex(from: index, shape: shape)
            let ldIdx = CPUMemoryOperators.linearIndex(from: dstIdx, shape: dstShape)
            
            dstMem[ldIdx] = addMem[ldIdx] + sourceMem[lsIdx]
        }
    }
    
    public static func maxPool2d<N>(
        input: Buffer<N, CPU>,
        result: Buffer<N, CPU>,
        resultContext: Buffer<Int32, CPU>,
        inputSize: (batchSize: Int, depth: Int, height: Int, width: Int),
        kernelSize: (height: Int, width: Int),
        stride: (vertical: Int, horizontal: Int)
    ) where N : NumericType {
        let dstWidth = (inputSize.width - kernelSize.width) / stride.horizontal + 1
        let dstHeight = (inputSize.height - kernelSize.height) / stride.vertical + 1
        
        let srcRowStride = inputSize.width
        let srcZStride = srcRowStride * inputSize.height
        let srcBatchStride = srcZStride * inputSize.depth
        
        let dstRowStride = dstWidth
        let dstZStride = dstRowStride * dstHeight
        let dstBatchStride = dstZStride * inputSize.depth
        
        let src = input.memory.bindMemory(to: N.self)
            .pointer(capacity: srcBatchStride * inputSize.batchSize)
        let dst = result.memory.bindMemory(to: N.self)
            .pointer(capacity: dstBatchStride * inputSize.batchSize)
        let ctx = resultContext.memory.bindMemory(to: Int32.self)
            .pointer(capacity: dstBatchStride * inputSize.batchSize)
        
        
        for batch in 0 ..< inputSize.batchSize {
            let srcBatch = src.advanced(by: srcBatchStride * batch)
            let dstBatch = dst.advanced(by: dstBatchStride * batch)
            let ctxBatch = ctx.advanced(by: dstBatchStride * batch)
            
            for z in 0 ..< inputSize.depth {
                let srcMatrix = srcBatch.advanced(by: z * srcZStride)
                let dstMatrix = dstBatch.advanced(by: z * dstZStride)
                let ctxMatrix = ctxBatch.advanced(by: z * dstZStride)
                
                for dstRow in 0 ..< dstHeight {
                    let srcRow = dstRow * stride.vertical
                    
                    for dstCol in 0 ..< dstWidth {
                        let srcCol = dstCol * stride.horizontal
                        
                        var min = srcMatrix[srcRow * srcRowStride + srcCol]
                        var minI = 0
                        
                        for i in 0 ..< kernelSize.height * kernelSize.width {
                            let x = i % kernelSize.width
                            let y = i / kernelSize.width
                            
                            let candidate = srcMatrix[(srcRow + y) * srcRowStride + srcCol + x]
                            
                            if candidate < min {
                                min = candidate
                                minI = i
                            }
                        }
                        
                        dstMatrix[dstRow * dstRowStride + dstCol] = min
                        ctxMatrix[dstRow * dstRowStride + dstCol] = Int32(minI)
                    }
                }
            }
        }
    }
    
    public static func maxPool2DRevAdd<N>(pooled: Buffer<N, CPU>, poolCtx: Buffer<Int32, CPU>, add: Buffer<Int32, CPU>, target: Buffer<Int32, CPU>, inputSize: (batchSize: Int, depth: Int, height: Int, width: Int), kernelSize: (height: Int, width: Int), stride: (vertical: Int, horizontal: Int)) where N : NumericType {
        let dstWidth = (inputSize.width - kernelSize.width) / stride.horizontal + 1
        let dstHeight = (inputSize.height - kernelSize.height) / stride.vertical + 1
        
        let srcRowStride = inputSize.width
        let srcZStride = srcRowStride * inputSize.height
        let srcBatchStride = srcZStride * inputSize.depth
        
        let dstRowStride = dstWidth
        let dstZStride = dstRowStride * dstHeight
        let dstBatchStride = dstZStride * inputSize.depth
        
        let dst = target.memory.bindMemory(to: N.self)
            .pointer(capacity: srcBatchStride * inputSize.batchSize)
        let addBase = add.memory.bindMemory(to: N.self)
            .pointer(capacity: srcBatchStride * inputSize.batchSize)
        let grad = pooled.memory.bindMemory(to: N.self)
            .pointer(capacity: dstBatchStride * inputSize.batchSize)
        let ctx = poolCtx.memory.bindMemory(to: Int32.self)
            .pointer(capacity: dstBatchStride * inputSize.batchSize)
        
        
        for batch in 0 ..< inputSize.batchSize {
            let dstBatch = dst.advanced(by: srcBatchStride * batch)
            let addBatch = addBase.advanced(by: srcBatchStride * batch)
            let gradBatch = grad.advanced(by: dstBatchStride * batch)
            let ctxBatch = ctx.advanced(by: dstBatchStride * batch)
            
            for z in 0 ..< inputSize.depth {
                let dstMatrix = dstBatch.advanced(by: z * srcZStride)
                let addMatrix = addBatch.advanced(by: z * srcZStride)
                let gradMatrix = gradBatch.advanced(by: z * dstZStride)
                let ctxMatrix = ctxBatch.advanced(by: z * dstZStride)
                
                for dstRow in 0 ..< dstHeight {
                    let srcRow = dstRow * stride.vertical
                    
                    for dstCol in 0 ..< dstWidth {
                        let srcCol = dstCol * stride.horizontal
                        
                        let i = Int(ctxMatrix[dstRow * dstRowStride + dstCol])
                        let g = gradMatrix[dstRow * dstRowStride + dstCol]
                        
                        let x = i % kernelSize.width
                        let y = i / kernelSize.width
                        
                        let a = addMatrix[(srcRow + y) * srcRowStride + srcCol + x]
                        dstMatrix[(srcRow + y) * srcRowStride + srcCol + x] = a + g
                    }
                }
            }
        }
    }
}

//
//  GPUEngine.swift
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
    
    public static func gemm<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        let function = MPSMatrixMultiplication(
            device: GPU.device,
            transposeLeft: transposeFirst,
            transposeRight: transposeSecond,
            resultRows: result.shape[0],
            resultColumns: result.shape[1],
            interiorColumns: transposeFirst ? lhs.shape[0] : lhs.shape[1],
            alpha: 1,
            beta: 1
        )
        
        let cmdBuffer = GPU.currentCommandBuffer
        function.encode(
            commandBuffer: cmdBuffer,
            leftMatrix: MPSMatrix(
                buffer: lhs.values.memory.buffer,
                offset: lhs.values.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: lhs.shape[0], columns: lhs.shape[1], rowBytes: lhs.shape[1] * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            ),
            rightMatrix: MPSMatrix(
                buffer: rhs.values.memory.buffer,
                offset: rhs.values.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: rhs.shape[0], columns: rhs.shape[1], rowBytes: rhs.shape[1] * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            ),
            resultMatrix: MPSMatrix(
                buffer: result.values.memory.buffer,
                offset: result.values.memory.offset,
                descriptor: MPSMatrixDescriptor(rows: result.shape[0], columns: result.shape[1], rowBytes: result.shape[1] * MemoryLayout<N>.stride, dataType: N.mpsDataType)
            )
        )
    }
    
    public static func broadcastGemm<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        fatalError("\(#function) not available for GPU")
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
    
    public static func tanh<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vTanh_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func sqrt<N: NumericType>(val: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        GPU.function(named: "vSqrt_\(N.gpuTypeIdentifier)").execute(workSize: (count, 1, 1), val.memory, result.memory)
    }
    
    public static func sum<N: NumericType>(val: Buffer<N, Device>, count: Int) -> N {
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
    
    public static func permuteAxes<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], destination: Buffer<N, GPU>) where N : NumericType {
        fatalError("\(#function) not available for GPU")
    }
    
    public static func permuteAxesAdd<N>(input: Buffer<N, GPU>, arangement: [Int], shape: [Int], add: Buffer<N, GPU>, destination: Buffer<N, GPU>) where N : NumericType {
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
        if axes.isEmpty || axes.allSatisfy({values.shape[$0] == 1}) {
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
        } else {
            GPU.function(named: "vSum_ReduceMulti_\(N.gpuTypeIdentifier)").execute(
                workSize: (result.count, 1, 1),
                values,
                CPU.Memory.strides(from: values.shape).valueArg,
                result,
                axes.count,
                axes.valueArg
            )
        }
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        if axes.count == 1 {
            reduceMax(values: values, result: result, context: context, axis: axes[0])
        } else {
            fatalError("\(#function) not available with multiple axes")
        }
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        if axes.count == 1 {
            reduceMin(values: values, result: result, context: context, axis: axes[0])
        } else {
            fatalError("\(#function) not available with multiple axes")
        }
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
        GPU.function(named: "vHeaviside_\(N.gpuTypeIdentifier)").execute(workSize: (values.count, 1, 1), values.values.memory, result.values.memory)
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N: NumericType {
        GPU.function(named: "permuteAxes_\(N.gpuTypeIdentifier)").execute(
            workSize: (values.count, 1, 1),
            values,
            result,
            arangement.valueArg,
            CPU.Memory.strides(from: result.shape).valueArg
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

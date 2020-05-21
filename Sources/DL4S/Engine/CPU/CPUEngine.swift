//
//  CPUEngineV2.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.03.19.
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


extension ShapedBuffer where Device == CPU {
    var pointer: UnsafeMutableBufferPointer<Element> {
        return values.memory.bindMemory(to: Element.self)
    }
    
    var immutable: UnsafeBufferPointer<Element> {
        return values.memory.bindMemory(to: Element.self).immutable
    }
}

extension Buffer where Device == CPU {
    var pointer: UnsafeMutableBufferPointer<Element> {
        return memory.bindMemory(to: Element.self)
    }
    
    var immutable: UnsafeBufferPointer<Element> {
        return memory.bindMemory(to: Element.self).immutable
    }
}


private enum BroadcastMode: Int {
    case vectorVector
    case vectorScalar
    case scalarVector
}


public struct CPUEngine: EngineType {
    public typealias Device = CPU
    
    public static func fill<N: NumericType>(value: N, result: Buffer<N, Device>, count: Int) {
        N.fill(value: value, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func vAdd<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vAdd(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
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
    
    public static func vDiv<N: NumericType>(lhs: Buffer<N, Device>, rhs: Buffer<N, Device>, result: Buffer<N, Device>, count: Int) {
        N.vDiv(lhs: lhs.memory.bindMemory(to: N.self).immutable, rhs: rhs.memory.bindMemory(to: N.self).immutable, result: result.memory.bindMemory(to: N.self), count: count)
    }
    
    public static func argmax<N: NumericType>(values: Buffer<N, Device>, count: Int) -> (Int, N) {
        return N.argmax(values: values.memory.bindMemory(to: N.self).immutable, count: count)
    }
    
    @inline(__always)
    @_specialize(where N == Float)
    private static func broadcast<N>(
        lhs: ShapedBuffer<N, CPU>,
        rhs: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        operator: (UnsafeBufferPointer<N>, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, Int) -> (),
        scalarOperatorA: (N, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, Int) -> (),
        scalarOperatorB: (UnsafeBufferPointer<N>, N, UnsafeMutableBufferPointer<N>, Int) -> ()
    ) {
        let dim = Swift.max(lhs.dim, rhs.dim)
        
        // Reshape operands to their broadcasting shape by padding shape with ones from the left
        let lhs = lhs.reshaped(to: [Int](repeating: 1, count: dim - lhs.dim) + lhs.shape)
        let rhs = rhs.reshaped(to: [Int](repeating: 1, count: dim - rhs.dim) + rhs.shape)
        
        #if DEBUG
        // Result must have same dimensionality as broadcasted operands
        assert(dim == result.dim)
        // Check, whether lhs and rhs have compatible shapes.
        // Every dimension must be either equal or 1 for one of the operands
        assert(zip(lhs.shape, rhs.shape).map {$0 == $1 || $0 == 1 || $1 == 1}.allSatisfy {$0 == true})
        // Check whether shape of result buffer matches combined operands
        assert(zip(zip(lhs.shape, rhs.shape), result.shape).map {Swift.max($0.0, $0.1) == $1}.allSatisfy {$0 == true})
        #endif
        
        // Determine, whether both operands have a suffix with more than 1 element
        // If that is the case, vector-vector mode is used
        
        let vectorSuffix = zip(lhs.shape, rhs.shape).suffix(while: {$0 == $1}).map {$1}
        
        let iterShape: [Int]
        let sliceCount: Int
        let mode: BroadcastMode
        
        let sc = vectorSuffix.reduce(1, *)
        
        if sc > 1 {
            // Iteration shape contains all dimensions from zero through the first non-equal sized dimension from the back
            iterShape = result.shape.dropLast(vectorSuffix.count)
            sliceCount = sc
            mode = .vectorVector
        } else {
            // Determine at which dimension the shape difference occurs and select the mode to either scalar-vector or vector-scalar
            // depending on whether the lhs or rhs operand is bigger in that dimension
            var m: BroadcastMode? = nil
            var devIndex = 0
            for i in result.shape.indices.reversed() {
                devIndex = i
                if lhs.shape[i] > rhs.shape[i] {
                    m = .vectorScalar
                    break
                } else if rhs.shape[i] > lhs.shape[i] {
                    m = .scalarVector
                    break
                }
            }
            if let m = m {
                mode = m
                
                // Determine the maximum suffix that can be processed in a single scalar-vector / vector-scalar operation
                var prefixCount = 0
                var sc = 1
                
                loop: for i in (0 ... devIndex).reversed() {
                    switch mode {
                    case .scalarVector:
                        if lhs.shape[i] > 1 {
                            break loop
                        }
                        sc *= rhs.shape[i]
                    case .vectorScalar:
                        if rhs.shape[i] > 1 {
                            break loop
                        }
                        sc *= lhs.shape[i]
                    default:
                        break
                    }
                    prefixCount = i
                }
                iterShape = Array(result.shape.prefix(prefixCount))
                sliceCount = sc
            } else {
                // All dimensions equally 1, thereby the operation is a scalar-scalar operation, which is not implemented separately
                mode = .vectorVector
                iterShape = []
                sliceCount = 1
            }
        }
        
        let lhsStrides = CPU.Memory.strides(from: lhs.shape)
        let rhsStrides = CPU.Memory.strides(from: rhs.shape)
        let dstStrides = CPU.Memory.strides(from: result.shape)
        
        let resultIndices = flatIterate(iterShape)
        let resultDim = iterShape.count
        let indexCount = resultIndices.count / Swift.max(resultDim, 1)
        
        for k in 0 ..< Swift.max(indexCount, 1) {
            let base = resultDim * k
            var lhsIdx = 0
            var rhsIdx = 0
            var dstIdx = 0
            
            for i in 0 ..< resultDim {
                lhsIdx += lhsStrides[i] * Swift.min(lhs.shape[i] - 1, resultIndices[base + i])
                rhsIdx += rhsStrides[i] * Swift.min(rhs.shape[i] - 1, resultIndices[base + i])
                dstIdx += dstStrides[i] * resultIndices[base + i]
            }
            
            let lhsSlice = lhs.immutable.advanced(by: lhsIdx)
            let rhsSlice = rhs.immutable.advanced(by: rhsIdx)
            let resultSlice = result.pointer.advanced(by: dstIdx)
            
            switch mode {
            case .vectorVector:
                `operator`(lhsSlice, rhsSlice, resultSlice, sliceCount)
                
            case .scalarVector:
                scalarOperatorA(lhsSlice.pointee, rhsSlice, resultSlice, sliceCount)
                
            case .vectorScalar:
                scalarOperatorB(lhsSlice, rhsSlice.pointee, resultSlice, sliceCount)
            }
        }
    }
    
    @inline(__always)
    @_specialize(where N == Float)
    private static func reduce<N>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        axis: Int,
        reduceOperator: (_ buffer: UnsafeBufferPointer<N>, _  stride: Int, _ count: Int) -> N
    ) {
        #if DEBUG
        var shape = values.shape
        shape.remove(at: axis)
        precondition(shape == result.shape)
        #endif
        
        let srcStrides = CPU.Memory.strides(from: values.shape)
        let reductionStride = srcStrides[axis]
        let axisSize = values.shape[axis]
        let dstStrides = CPU.Memory.strides(from: result.shape)
        
        let indices = flatIterate(result.shape)
        let dim = result.dim
        let count = indices.count / dim
        
        for k in 0 ..< count {
            let base = k * dim
            var prefixOffset = 0
            var suffixOffset = 0
            var linearIndex = 0
            for i in 0 ..< axis {
                prefixOffset += srcStrides[i] * indices[base + i]
            }
            for i in Swift.min(axis + 1, dim) ..< dim {
                suffixOffset += srcStrides[i] * indices[base + i]
            }
            for i in 0 ..< dim {
                linearIndex += indices[base + i] * dstStrides[i]
            }
            
            // let prefixOffset = zip(srcStrides.prefix(upTo: axis), idx).map(*).reduce(0, +)
            // let suffixOffset = zip(srcStrides.suffix(from: axis+1), idx.suffix(from: axis)).map(*).reduce(0, +)
            let totalOffset = prefixOffset + suffixOffset
            
            let reduced = reduceOperator(values.immutable.advanced(by: totalOffset), reductionStride, axisSize)
            // let linearIndex = zip(idx, dstStrides).map(*).reduce(0,+)
            result.values[linearIndex] = reduced
        }
    }
    
    // @available(*, deprecated, message: "Don't use this one, it is slow")
    @inline(__always)
    @_specialize(where N == Float)
    private static func reduceMultiAxis<N>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        axes: [Int],
        reduceOperator: (UnsafeBufferPointer<N>, Int) -> N
    ) {
        #if DEBUG
        var shape = values.shape
        for axis in axes.reversed() {
            shape.remove(at: axis)
        }
        precondition(shape == result.shape)
        #endif
        
        let dstStrides = CPU.Memory.strides(from: result.shape)
        let sliceCount = axes.map {values.shape[$0]}.reduce(1, *)
        
        for idx in iterate(result.shape) {
            var srcIdx: [Int?] = idx
            
            for axis in axes {
                srcIdx.insert(nil, at: axis)
            }
            
            let (slice, isCopy, _) = CPU.Memory.get(slice: srcIdx, of: values.values, with: values.shape)
            
            let reduced = reduceOperator(slice.immutable, sliceCount)
            
            let linearIndex = zip(dstStrides, idx).map(*).reduce(0, +)
            result.values[linearIndex] = reduced
            
            if isCopy {
                CPU.Memory.free(slice)
            }
        }
    }
    
    @inline(__always)
    @_specialize(where N == Float)
    private static func reduceMultiAxis<N: ZeroableType>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        axes: [Int],
        reduceOperator: (_ buffer: UnsafeBufferPointer<N>, _  stride: Int, _ count: Int) -> N,
        reduceCombine: (N, N) -> N
    ) {
        #if DEBUG
        var shape = values.shape
        for axis in axes.reversed() {
            shape.remove(at: axis)
        }
        precondition(shape == result.shape)
        #endif
        
        let srcStrides = CPU.Memory.strides(from: values.shape)
        let dstStrides = CPU.Memory.strides(from: result.shape)
        
        var reducedShape = [Int]()
        var reducedStrides = [Int]()
        var srcStridesDstIdx = srcStrides
        
        // reducedShape.reserveCapacity(axes.count)
        // reducedStrides.reserveCapacity(axes.count)
        for a in axes {
            reducedShape.append(values.shape[a])
            reducedStrides.append(srcStrides[a])
        }
        
        let reductionStride = reducedStrides.last ?? 1
        let reductionCount = reducedShape.last ?? 1
        
        for a in axes.reversed() {
            srcStridesDstIdx.remove(at: a)
        }
        
        let srcReduceIndices = iterate(reducedShape.dropLast())
        
        let dstPtr = result.values.pointer.pointer(capacity: result.count)
        
        var dstIdx: Int
        var srcOffset: Int
        var srcAddOffset: Int
        
        for idx in iterate(result.shape) {
            dstIdx = 0
            srcOffset = 0
            for i in 0 ..< idx.count {
                dstIdx += dstStrides[i] * idx[i]
                srcOffset += srcStridesDstIdx[i] * idx[i]
            }
            
            var reduced: N = .zero
            
            for srcReduceIndex in srcReduceIndices {
                srcAddOffset = 0
                for i in 0 ..< srcReduceIndex.count {
                    srcAddOffset += reducedStrides[i] * srcReduceIndex[i]
                }
                
                reduced = reduceCombine(
                    reduced,
                    reduceOperator(
                        values.immutable.advanced(by: srcOffset + srcAddOffset),
                        reductionStride,
                        reductionCount
                    )
                )
            }
            
            dstPtr[dstIdx] = reduced
        }
    }
    
    @inline(__always)
    @_specialize(where N == Float)
    private static func reducePrefix<N: NumericType>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        reduceColumns: (UnsafeBufferPointer<N>, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, Int) -> ()
    ) {
        if result.count == 0 {
            return
        }
        
        let stride = result.count
        let count = values.count / stride
        
        N.fill(value: 0, result: result.pointer, count: result.count)
        
        for i in 0 ..< count {
            let offset = stride * i
            
            reduceColumns(values.immutable.advanced(by: offset), result.immutable, result.pointer, stride)
        }
    }
    
    @inline(__always)
    @_specialize(where N == Float, Context == Int32)
    private static func reduceWithContext<N, Context>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        context: ShapedBuffer<Context, CPU>,
        axis: Int,
        reduceOperator: (UnsafeBufferPointer<N>, Int, Int) -> (N, Context)
    ) {
        #if DEBUG
        precondition(result.shape == context.shape)
        
        var shape = values.shape
        shape.remove(at: axis)
        precondition(shape == result.shape)
        #endif
        
        let srcPtr = values.immutable
        let dstPtr = result.pointer
        let ctxPtr = context.pointer
        
        let dstStrides = CPU.Memory.strides(from: result.shape)
        var srcStrides = CPU.Memory.strides(from: values.shape)
        let reductionStride = srcStrides.remove(at: axis)
        let reductionCount = values.shape[axis]
        
        let indices = flatIterate(result.shape)
        let resultDim = result.dim
        
        for i in 0 ..< result.count {
            var srcBase = 0
            var dstIdx = 0
            
            for j in 0 ..< result.dim {
                srcBase += srcStrides[j] * indices[j + i * resultDim]
                dstIdx += dstStrides[j] * indices[j + i * resultDim]
            }
            
            let (val, ctx) = reduceOperator(srcPtr.advanced(by: srcBase), reductionStride, reductionCount)
            dstPtr[dstIdx] = val
            ctxPtr[dstIdx] = ctx
        }
    }
    
    // @available(*, deprecated, message: "Don't use this one, it is slow")
    @inline(__always)
    @_specialize(where N == Float, Context == Int32)
    private static func reduceMultiAxisWithContext<N, Context>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        context: ShapedBuffer<Context, CPU>,
        axes: [Int],
        reduceOperator: (UnsafeBufferPointer<N>, Int) -> (N, Context)
        ) {
        #if DEBUG
        precondition(result.shape == context.shape)

        var shape = values.shape
        for axis in axes.reversed() {
            shape.remove(at: axis)
        }
        precondition(shape == result.shape)
        #endif

        let dstStrides = CPU.Memory.strides(from: result.shape)
        let sliceCount = axes.map {values.shape[$0]}.reduce(1, *)

        for idx in iterate(result.shape) {
            var srcIdx: [Int?] = idx

            for axis in axes {
                srcIdx.insert(nil, at: axis)
            }

            let (slice, isCopy, _) = CPU.Memory.get(slice: srcIdx, of: values.values, with: values.shape)

            let (reduced, ctxValue) = reduceOperator(slice.immutable, sliceCount)

            let linearIndex = zip(dstStrides, idx).map(*).reduce(0, +)
            result.values[linearIndex] = reduced
            context.values[linearIndex] = ctxValue

            if isCopy {
                CPU.Memory.free(slice)
            }
        }
    }
    
    @inline(__always)
    @_specialize(where N == Float, Context == Int32)
    private static func reducePrefixWithContext<N: NumericType, Context: NumericType>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        context: ShapedBuffer<Context, CPU>,
        reduceColumns: (UnsafeBufferPointer<N>, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, UnsafeMutableBufferPointer<Context>, Int) -> ()
    ) {
        if result.count == 0 {
            return
        }
        
        let stride = result.count
        let count = values.count / stride
        
        N.fill(value: 0, result: result.pointer, count: result.count)
        Context.fill(value: 0, result: context.pointer, count: context.count)
        
        for i in 0 ..< count {
            let offset = stride * i
            
            reduceColumns(values.immutable.advanced(by: offset), result.immutable, result.pointer, context.pointer, stride)
        }
    }
    
    @_specialize(where N == Float)
    public static func gemm<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        N.gemm(
            lhs: lhs.values.memory.bindMemory(to: N.self).immutable,
            rhs: rhs.values.memory.bindMemory(to: N.self).immutable,
            result: result.values.memory.bindMemory(to: N.self),
            lhsShape: (lhs.shape[0], lhs.shape[1]),
            rhsShape: (rhs.shape[0], rhs.shape[1]),
            resultShape: (result.shape[0], result.shape[1]),
            alpha: alpha,
            beta: beta,
            transposeFirst: transposeFirst,
            transposeSecond: transposeSecond
        )
    }
    
    @_specialize(where N == Float)
    public static func broadcastAdd<N: NumericType>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        broadcast(
            lhs: lhs,
            rhs: rhs,
            result: result,
            operator: N.vAdd,
            scalarOperatorA: {N.vsAdd(lhs: $1, rhs: $0, result: $2, count: $3)},
            scalarOperatorB: N.vsAdd
        )
    }
    
    @_specialize(where N == Float)
    public static func broadcastSub<N: NumericType>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        broadcast(
            lhs: lhs,
            rhs: rhs,
            result: result,
            operator: N.vSub,
            scalarOperatorA: {
                N.vsAdd(lhs: $1, rhs: -$0, result: $2, count: $3)
                N.vNeg(val: $2.immutable, result: $2, count: $3)
            },
            scalarOperatorB: {N.vsAdd(lhs: $0, rhs: -$1, result: $2, count: $3)}
        )
    }
    
    @_specialize(where N == Float)
    public static func broadcastMul<N: NumericType>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        broadcast(
            lhs: lhs,
            rhs: rhs,
            result: result,
            operator: N.vMul,
            scalarOperatorA: {N.vsMul(lhs: $1, rhs: $0, result: $2, count: $3)},
            scalarOperatorB: N.vsMul
        )
    }
    
    @_specialize(where N == Float)
    public static func broadcastDiv<N: NumericType>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        broadcast(
            lhs: lhs,
            rhs: rhs,
            result: result,
            operator: N.vDiv,
            scalarOperatorA: N.svDiv,
            scalarOperatorB: {N.vsMul(lhs: $0, rhs: 1 / $1, result: $2, count: $3)}
        )
    }
    
    @_specialize(where N == Float)
    public static func reduceSum<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) {
        // Choose the operation that performs better
        if axis == 0 && result.shape.reduce(1, *) > 1 {
            reducePrefix(values: values, result: result, reduceColumns: N.vAdd)
        } else {
            reduce(
                values: values,
                result: result,
                axis: axis,
                reduceOperator: N.sum(val:stride:count:)
            )
        }
    }
    
    @_specialize(where N == Float)
    public static func reduceMax<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) {
        if let context = context {
            reduceWithContext(
                values: values,
                result: result,
                context: context,
                axis: axis,
                reduceOperator: { buffer, stride, count -> (N, Int32) in
                    let (arg, max) = N.argmax(values: buffer, stride: stride, count: count)
                    return (max, Int32(arg))
                }
            )
        } else if axis == 0 && result.shape.reduce(1, *) > 1 {
            reducePrefix(values: values, result: result, reduceColumns: N.max)
        } else {
            reduce(
                values: values,
                result: result,
                axis: axis
            ) { buffer, stride, count in
                N.argmax(values: buffer, stride: stride, count: stride).1
            }
        }
    }
    
    @_specialize(where N == Float)
    public static func reduceMin<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) {
        if let context = context {
            reduceWithContext(
                values: values,
                result: result,
                context: context,
                axis: axis,
                reduceOperator: { buffer, stride, count -> (N, Int32) in
                    let (arg, min) = N.argmin(values: buffer, stride: stride, count: count)
                    return (min, Int32(arg))
                }
            )
        } else if axis == 0 && result.shape.reduce(1, *) > 1 {
            reducePrefix(values: values, result: result, reduceColumns: N.min)
        } else {
            reduce(
                values: values,
                result: result,
                axis: axis
            ) { values, stride, count in
                N.argmin(values: values, stride: stride, count: count).1
            }
        }
    }
    
    @_specialize(where N == Float)
    public static func reduceMean<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) {
        reduceSum(values: values, result: result, axis: axis)
        let axisCount = values.shape[axis]
        N.vsMul(lhs: result.immutable, rhs: 1 / N(axisCount), result: result.pointer, count: result.count)
    }
    
    @_specialize(where N == Float)
    public static func reduceSum<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axes: [Int]) {
        if axes.elementsEqual(0 ..< axes.count) && result.shape.reduce(1, *) > 1 {
            reducePrefix(
                values: values,
                result: result,
                reduceColumns: N.vAdd
            )
        } else {
            reduceMultiAxis(
                values: values,
                result: result,
                axes: axes,
                reduceOperator: N.sum,
                reduceCombine: +
            )
        }
    }
    
    @_specialize(where N == Float)
    public static func reduceMean<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axes: [Int]) {
        reduceSum(values: values, result: result, axes: axes)
        let axisCount = axes.map {values.shape[$0]}.reduce(1, *)
        N.vsMul(lhs: result.immutable, rhs: 1 / N(axisCount), result: result.pointer, count: result.count)
    }
    
    @_specialize(where N == Float)
    public static func reduceMax<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axes: [Int]) {
        if let context = context {
            reduceMultiAxisWithContext(
                values: values,
                result: result,
                context: context,
                axes: axes,
                reduceOperator: { buffer, count -> (N, Int32) in
                    let (arg, max) = N.argmax(values: buffer, count: count)
                    return (max, Int32(arg))
                }
            )
        } else {
            reduceMultiAxis(values: values, result: result, axes: axes) {
                N.argmax(values: $0, count: $1).1
            }
        }
    }
    
    @_specialize(where N == Float)
    public static func reduceMin<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axes: [Int]) {
        if let context = context {
            reduceMultiAxisWithContext(
                values: values,
                result: result,
                context: context,
                axes: axes,
                reduceOperator: { buffer, count -> (N, Int32) in
                    let (arg, min) = N.argmin(values: buffer, count: count)
                    return (min, Int32(arg))
                }
            )
        } else {
            reduceMultiAxis(values: values, result: result, axes: axes) {
                N.argmin(values: $0, count: $1).1
            }
        }
    }
    
    public static func scatter<N: NumericType>(reduced: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>, result: ShapedBuffer<N, CPU>, axis: Int, ignoreIndex: Int32) {
        N.scatter(values: reduced.immutable, context: context.immutable, result: result.pointer, dst_shape: result.shape, axis: axis, ignoreIndex: ignoreIndex)
    }
    
    public static func gather<N: NumericType>(expanded: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>, result: ShapedBuffer<N, CPU>, axis: Int, ignoreIndex: Int32) {
        N.gather(values: expanded.immutable, context: context.immutable, result: result.pointer, src_shape: expanded.shape, axis: axis, ignoreIndex: ignoreIndex)
    }
    
    @_specialize(where N == Float)
    public static func sum<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        result.values.pointee = N.sum(val: values.immutable, count: values.count)
    }
    
    @_specialize(where N == Float)
    public static func mean<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        result.values.pointee = N.sum(val: values.immutable, count: values.count) / N(values.count)
    }
    
    @discardableResult
    @_specialize(where N == Float)
    public static func max<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) -> Int {
        let (arg, max) = N.argmax(values: values.immutable, count: values.count)
        result.values.pointee = max
        return arg
    }
    
    @discardableResult
    @_specialize(where N == Float)
    public static func min<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) -> Int {
        let (arg, min) = N.argmin(values: values.immutable, count: values.count)
        result.values.pointee = min
        return arg
    }
    
    public static func exp<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.exp(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func log<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.log(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func sqrt<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.sqrt(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func relu<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.relu(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func heaviside<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        if let values = values as? ShapedBuffer<Float, CPU>, let result = result as? ShapedBuffer<Float, CPU> {
            Float.heaviside(values: values.immutable, result: result.pointer, count: result.count)
            return
        }
        
        let srcPtr = values.pointer.pointer(capacity: result.count)
        let dstPtr = result.pointer.pointer(capacity: result.count)
        
        for i in 0 ..< result.count {
            dstPtr[i] = srcPtr[i] > 0 ? 1 : 0
        }
    }
    
    public static func sin<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.sin(values: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func cos<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.cos(values: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func tan<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.tan(values: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func sinh<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func cosh<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func tanh<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        N.tanh(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, CPU>, _ rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        
        N.max(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, count: result.count)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, CPU>, _ rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<N, CPU>) where N : NumericType {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        precondition(context.shape == result.shape, "Shapes of context and result must match")
        
        N.max(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, context: context.pointer, count: result.count)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, CPU>, _ rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        
        N.min(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, count: result.count)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, CPU>, _ rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<N, CPU>) where N : NumericType {
        precondition(lhs.shape == rhs.shape, "Shapes of lhs and rhs must match")
        precondition(lhs.shape == result.shape, "Shapes of inputs and result must match")
        precondition(context.shape == result.shape, "Shapes of context and result must match")
        
        N.min(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, context: context.pointer, count: result.count)
    }
    
    public static func permuteAxes<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, arangement: [Int]) {
        let dim = values.dim
        
        if dim == 2 && arangement == [1, 0] {
            // Fast path if operation is matrix transpose
            N.transpose(val: values.immutable, result: result.pointer, srcRows: values.shape[0], srcCols: values.shape[1])
            return
        }
        
        let sourceMem = values.immutable.pointer(capacity: values.count)
        let dstMem = result.pointer.pointer(capacity: result.count)
        
        let shape = values.shape
        let dstShape = result.shape
        
        let suffix = zip(shape.indices, arangement).suffix(while: {$0 == $1}).count
        // let suffix = 0
        
        let copyCount = shape.suffix(suffix).reduce(1, *)
        let iterShape = shape.dropLast(suffix) as Array

        let srcStrides = CPU.Memory.strides(from: shape)
        let dstStrides = CPU.Memory.strides(from: dstShape)
        
        let indexDim = iterShape.count
        let indices = flatIterate(iterShape)
        let indexCount = indices.count / indexDim
        for j in 0 ..< indexCount {
            let offset = indexDim * j
            var srcIdx = 0
            var dstIdx = 0
            for i in 0 ..< indexDim {
                srcIdx += indices[offset + i] * srcStrides[i]
                dstIdx += indices[offset + i] * dstStrides[arangement[i]]
            }
            
            dstMem.advanced(by: dstIdx).assign(from: sourceMem.advanced(by: srcIdx), count: copyCount)
        }
    }
    
    public static func permuteAxesAdd<N: NumericType>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, arangement: [Int]) {
        let sourceMem = values.immutable
        let addMem = add.immutable
        let dstMem = result.pointer
        
        let shape = values.shape
        let dstShape = result.shape
        
        let suffix = zip(shape.indices, arangement).suffix(while: {$0 == $1}).count
        
        let copyCount = shape.suffix(suffix).reduce(1, *)
        let iterShape = shape.dropLast(suffix) as Array
        
        let srcStrides = CPU.Memory.strides(from: shape)
        let dstStrides = CPU.Memory.strides(from: dstShape)
        
        let indexDim = iterShape.count
        let indices = flatIterate(iterShape)
        let indexCount = indices.count / indexDim
        for j in 0 ..< indexCount {
            let offset = indexDim * j
            var srcIdx = 0
            var dstIdx = 0
            for i in 0 ..< indexDim {
                srcIdx += indices[offset + i] * srcStrides[i]
                dstIdx += indices[offset + i] * dstStrides[arangement[i]]
            }
            
            N.vAdd(lhs: sourceMem.advanced(by: srcIdx), rhs: addMem.advanced(by: dstIdx), result: dstMem.advanced(by: dstIdx), count: copyCount)
        }
    }
    
    public static func arange<N: NumericType>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, CPU>) {
        N.arange(start: lowerBound, end: upperBound, result: result.pointer, count: result.count)
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) {
        let (buffer, isCopy, bufferShape) = CPU.Memory.get(slice: index, of: values.values, with: values.shape)
        
        result.pointer.assign(from: buffer.immutable, count: bufferShape.reduce(1, *))
        
        if isCopy {
            CPU.Memory.free(buffer)
        }
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func subscriptReadAdd<N: NumericType>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func subscriptWriteAdd<N: NumericType>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, CPU>], result: ShapedBuffer<N, CPU>, axis: Int) {
        var offset = 0
        
        let dstPtr = result.pointer.pointer(capacity: buffers.map {$0.count}.reduce(0, +))
        let dstStrides = CPU.Memory.strides(from: result.shape)
        
        for buffer in buffers {
            let dst = dstPtr.advanced(by: offset)
            let srcStrides = CPU.Memory.strides(from: buffer.shape)
            let copyCount = buffer.shape[axis] * srcStrides[axis]
            
            let iterShape = Array(buffer.shape.prefix(upTo: axis))
            let src = buffer.immutable.pointer(capacity: buffer.count)
            
            for idx in iterate(iterShape) {
                var srcIdx = 0
                var dstIdx = 0
                
                for i in 0 ..< idx.count {
                    srcIdx += srcStrides[i] * idx[i]
                    dstIdx += dstStrides[i] * idx[i]
                }
                
                dst.advanced(by: dstIdx).assign(from: src.advanced(by: srcIdx), count: copyCount)
            }
            
            offset += copyCount
        }
    }
    
    public static func unstackAdd<N: NumericType>(stacked: ShapedBuffer<N, CPU>, add: [ShapedBuffer<N, CPU>], result: [ShapedBuffer<N, CPU>], axis: Int) {
        var offset = 0
        
        let srcPtr = stacked.immutable
        let srcStrides = CPU.Memory.strides(from: stacked.shape)
        
        for (buffer, addBuffer) in zip(result, add) {
            let src = srcPtr.advanced(by: offset)
            let dstStrides = CPU.Memory.strides(from: buffer.shape)
            let copyCount = buffer.shape[axis] * dstStrides[axis]
            
            let iterShape = Array(buffer.shape.prefix(upTo: axis))
            let dst = buffer.pointer
            let a = addBuffer.immutable
            
            for idx in iterate(iterShape) {
                var srcIdx = 0
                var dstIdx = 0
                
                for i in 0 ..< idx.count {
                    srcIdx += srcStrides[i] * idx[i]
                    dstIdx += dstStrides[i] * idx[i]
                }
                
                N.vAdd(lhs: src.advanced(by: srcIdx), rhs: a.advanced(by: dstIdx), result: dst.advanced(by: dstIdx), count: copyCount)
            }
            
            offset += copyCount
        }
    }
    
    public static func unstack<N>(stacked: ShapedBuffer<N, CPU>, result: [ShapedBuffer<N, CPU>], axis: Int) where N : NumericType {
        var offset = 0
        
        let srcPtr = stacked.immutable
        let srcStrides = CPU.Memory.strides(from: stacked.shape)
        
        for buffer in result {
            let src = srcPtr.advanced(by: offset)
            let dstStrides = CPU.Memory.strides(from: buffer.shape)
            let copyCount = buffer.shape[axis] * dstStrides[axis]
            
            let iterShape = Array(buffer.shape.prefix(upTo: axis))
            let dst = buffer.pointer

            for idx in iterate(iterShape) {
                var srcIdx = 0
                var dstIdx = 0
                
                for i in 0 ..< idx.count {
                    srcIdx += srcStrides[i] * idx[i]
                    dstIdx += dstStrides[i] * idx[i]
                }
                
                dst.advanced(by: dstIdx).assign(from: src.advanced(by: srcIdx), count: copyCount)
            }
            
            offset += copyCount
        }
    }
    
    public static func reverse<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        precondition(values.shape == result.shape)
        
        let stride = CPU.Memory.strides(from: values.shape)[0]
        let count = values.shape[0]
        
        let srcPtr = values.immutable.pointer(capacity: stride * count)
        let dstPtr = values.pointer.pointer(capacity: stride * count)
        
        for srcIdx in 0 ..< count {
            let dstIdx = count - srcIdx - 1
            
            dstPtr.advanced(by: dstIdx * stride).assign(from: srcPtr.advanced(by: srcIdx * stride), count: stride)
        }
    }
    
    public static func reverseAdd<N: NumericType>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) {
        precondition(values.shape == result.shape && values.shape == add.shape)
        
        let stride = CPU.Memory.strides(from: values.shape)[0]
        let count = values.shape[0]
        
        let srcPtr = values.immutable
        let addPtr = values.immutable
        let dstPtr = values.pointer
        
        for srcIdx in 0 ..< count {
            let dstIdx = count - srcIdx - 1
            N.vAdd(
                lhs: srcPtr.advanced(by: srcIdx * stride),
                rhs: addPtr.advanced(by: dstIdx * stride),
                result: dstPtr.advanced(by: dstIdx * stride),
                count: stride
            )
        }
    }
    
    @_specialize(where N == Float)
    public static func img2col<N: NumericType>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(values.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        N.img2col(values: values.immutable, result: result.pointer, batchSize: values.shape[0], channels: values.shape[1], height: values.shape[2], width: values.shape[3], kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride);
    }
    
    @_specialize(where N == Float)
    public static func col2img<N: NumericType>(matrix: ShapedBuffer<N, CPU>, image: ShapedBuffer<N, CPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(image.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        N.col2img(values: matrix.immutable, result: image.pointer, batchSize: image.shape[0], channels: image.shape[1], height: image.shape[2], width: image.shape[3], kernelHeight: kernelHeight, kernelWidth: kernelWidth, padding: padding, stride: stride);
    }
    
    public static func band<N>(buffer: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, belowDiagonal: Int?, aboveDiagonal: Int?) where N : NumericType {
        precondition(buffer.shape == result.shape, "Shape of result must be equal to shape of buffer.")
        precondition(buffer.dim == 2, "Band can only be computed on tensor of dimensionality 2.")
        
        let rows = buffer.shape[0]
        let cols = buffer.shape[1]
        
        let belowDiagonal = belowDiagonal ?? Swift.max(rows, cols)
        let aboveDiagonal = aboveDiagonal ?? Swift.max(rows, cols)
        
        let src = buffer.values.memory.bindMemory(to: N.self).immutable.pointer(capacity: rows * cols)
        let dst = result.values.memory.bindMemory(to: N.self).pointer(capacity: rows * cols)
        
        for i in 0 ..< rows {
            let start = Swift.min(Swift.max(0, i - belowDiagonal), cols)
            let end = Swift.max(Swift.min(cols, i + aboveDiagonal + 1), start)
            
            memcpy(
                UnsafeMutableRawPointer(dst.advanced(by: i * cols + start)),
                UnsafeRawPointer(src.advanced(by: i * cols + start)),
                MemoryLayout<N>.stride * (end - start)
            )
        }
    }
}

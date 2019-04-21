//
//  CPUEngineV2.swift
//  DL4S
//
//  Created by Palle Klewitz on 16.03.19.
//

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

extension CPUEngine: EngineTypeV2 {
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
        
        for resultIdx in iterate(iterShape) {
            var lhsIdx = 0
            var rhsIdx = 0
            var dstIdx = 0
            
            for i in 0 ..< resultIdx.count {
                lhsIdx += lhsStrides[i] * Swift.min(lhs.shape[i] - 1, resultIdx[i])
                rhsIdx += rhsStrides[i] * Swift.min(rhs.shape[i] - 1, resultIdx[i])
                dstIdx += dstStrides[i] * resultIdx[i]
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
    
    @available(*, deprecated, message: "Don't use this one, it is slow")
    @inline(__always)
    @_specialize(where N == Float)
    private static func reduce<N>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        axis: Int,
        reduceOperator: (UnsafeBufferPointer<N>, Int) -> N
    ) {
        #if DEBUG
        var shape = values.shape
        shape.remove(at: axis)
        precondition(shape == result.shape)
        #endif
        
        let dstStrides = CPU.Memory.strides(from: result.shape)
        let sliceCount = values.shape[axis]
        
        for idx in iterate(result.shape) {
            var srcIdx: [Int?] = idx
            srcIdx.insert(nil, at: axis)
            
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
        
        for idx in iterate(result.shape) {
            let prefixOffset = zip(srcStrides.prefix(upTo: axis), idx).map(*).reduce(0, +)
            let suffixOffset = zip(srcStrides.suffix(from: axis+1), idx.suffix(from: axis)).map(*).reduce(0, +)
            let totalOffset = prefixOffset + suffixOffset
            
            let reduced = reduceOperator(values.immutable.advanced(by: totalOffset), reductionStride, axisSize)
            let linearIndex = zip(idx, dstStrides).map(*).reduce(0,+)
            result.values[linearIndex] = reduced
        }
    }
    
    @available(*, deprecated, message: "Don't use this one, it is slow")
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
    private static func reduceMultiAxis<N>(
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
            
            var reduced: N = 0
            
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
    private static func reducePrefix<N>(
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
    
    @available(*, deprecated, message: "Don't use this one, it is slow")
    @inline(__always)
    @_specialize(where N == Float, Context == Int32)
    private static func reduceWithContext<N, Context>(
        values: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        context: ShapedBuffer<Context, CPU>,
        axis: Int,
        reduceOperator: (UnsafeBufferPointer<N>, Int) -> (N, Context)
    ) {
        #if DEBUG
        precondition(result.shape == context.shape)
        
        var shape = values.shape
        shape.remove(at: axis)
        precondition(shape == result.shape)
        #endif
        
        let dstStrides = CPU.Memory.strides(from: result.shape)
        let sliceCount = values.shape[axis]
        
        for idx in iterate(result.shape) {
            var srcIdx: [Int?] = idx
            srcIdx.insert(nil, at: axis)
            
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
    
    @available(*, deprecated, message: "Don't use this one, it is slow")
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
    private static func reducePrefixWithContext<N, Context>(
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
    public static func matMul<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.matMul(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, lhsRows: lhs.shape[0], lhsCols: lhs.shape[1], rhsCols: rhs.shape[1])
    }
    
    @_specialize(where N == Float)
    public static func matMulAdd<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        matMul(lhs: lhs, rhs: rhs, result: result)
        broadcastAdd(lhs: result, rhs: add, result: result)
    }
    
    @_specialize(where N == Float)
    public static func broadcastAdd<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        broadcast(
            lhs:
            lhs,
            rhs: rhs,
            result: result,
            operator: N.vAdd,
            scalarOperatorA: {N.vsAdd(lhs: $1, rhs: $0, result: $2, count: $3)},
            scalarOperatorB: N.vsAdd
        )
    }
    
    @_specialize(where N == Float)
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
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
    public static func broadcastMul<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
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
    public static func broadcastDiv<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
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
    public static func reduceSum<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
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
    public static func reduceMax<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) where N : NumericType {
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
    public static func reduceMin<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) where N : NumericType {
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
    public static func reduceMean<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
        reduceSum(values: values, result: result, axis: axis)
        let axisCount = values.shape[axis]
        N.vsMul(lhs: result.immutable, rhs: 1 / N(axisCount), result: result.pointer, count: result.count)
    }
    
    @_specialize(where N == Float)
    public static func reduceSum<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axes: [Int]) where N : NumericType {
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
    public static func reduceMean<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axes: [Int]) where N : NumericType {
        reduceSum(values: values, result: result, axes: axes)
        let axisCount = axes.map {values.shape[$0]}.reduce(1, *)
        N.vsMul(lhs: result.immutable, rhs: 1 / N(axisCount), result: result.pointer, count: result.count)
    }
    
    @_specialize(where N == Float)
    public static func reduceMax<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axes: [Int]) where N : NumericType {
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
    public static func reduceMin<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axes: [Int]) where N : NumericType {
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
    
    public static func expandContext<N>(reduced: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
        let srcStrides = CPUMemoryOperators.strides(from: reduced.shape)
        var dstStrides = CPUMemoryOperators.strides(from: result.shape)
        let axisStride = dstStrides.remove(at: axis)
        
        let srcPtr = reduced.immutable.pointer(capacity: reduced.count)
        let ctxPtr = context.immutable.pointer(capacity: reduced.count)
        let dstPtr = result.pointer.pointer(capacity: result.count)
        
        let count = reduced.count
        let dim = reduced.dim
        let indices = flatIterate(reduced.shape)
        
        for i in 0 ..< count {
            var srcIdx = 0
            var dstIdx = 0
            for j in 0 ..< dim {
                srcIdx += indices[i * dim + j] * srcStrides[j]
                dstIdx += indices[i * dim + j] * dstStrides[j]
            }
            
            dstIdx += axisStride * Int(ctxPtr[srcIdx])
            dstPtr[dstIdx] = srcPtr[srcIdx]
        }
    }
    
    @_specialize(where N == Float)
    public static func sum<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        result.values.pointee = N.sum(val: values.immutable, count: values.count)
    }
    
    @_specialize(where N == Float)
    public static func mean<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        result.values.pointee = N.sum(val: values.immutable, count: values.count) / N(values.count)
    }
    
    @discardableResult
    @_specialize(where N == Float)
    public static func max<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) -> Int where N : NumericType {
        let (arg, max) = N.argmax(values: values.immutable, count: values.count)
        result.values.pointee = max
        return arg
    }
    
    @discardableResult
    @_specialize(where N == Float)
    public static func min<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) -> Int where N : NumericType {
        let (arg, min) = N.argmin(values: values.immutable, count: values.count)
        result.values.pointee = min
        return arg
    }
    
    public static func exp<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.exp(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func log<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.log(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func sqrt<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.sqrt(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func square<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.vSquare(values: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func relu<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.relu(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func heaviside<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        let srcPtr = values.pointer.pointer(capacity: result.count)
        let dstPtr = result.pointer.pointer(capacity: result.count)
        
        for i in 0 ..< result.count {
            dstPtr[i] = srcPtr[i] > 0 ? 1 : 0
        }
    }
    
    public static func sin<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func cos<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func tan<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func sinh<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func cosh<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func tanh<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.tanh(val: values.immutable, result: result.pointer, count: result.count)
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, arangement: [Int]) where N : NumericType {
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

        for index in iterate(iterShape) {
            var srcIdx = 0
            var dstIdx = 0
            for i in 0 ..< index.count {
                srcIdx += index[i] * srcStrides[i]
                dstIdx += index[i] * dstStrides[arangement[i]]
            }
            
            dstMem.advanced(by: dstIdx).assign(from: sourceMem.advanced(by: srcIdx), count: copyCount)
        }
    }
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, arangement: [Int]) where N : NumericType {
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
        
        for index in iterate(iterShape) {
            var srcIdx = 0
            var dstIdx = 0
            for i in 0 ..< index.count {
                srcIdx += index[i] * srcStrides[i]
                dstIdx += index[i] * dstStrides[arangement[i]]
            }
            
            N.vAdd(lhs: sourceMem.advanced(by: srcIdx), rhs: addMem.advanced(by: dstIdx), result: dstMem.advanced(by: dstIdx), count: copyCount)
        }
    }
    
    public static func arange<N>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.arange(start: lowerBound, end: upperBound, result: result.pointer, count: result.count)
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) where N : NumericType {
        let (buffer, isCopy, bufferShape) = CPU.Memory.get(slice: index, of: values.values, with: values.shape)
        
        result.pointer.assign(from: buffer.immutable, count: bufferShape.reduce(1, *))
        
        if isCopy {
            CPU.Memory.free(buffer)
        }
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func subscriptReadAdd<N>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func subscriptWriteAdd<N>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, CPU>], result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    @_specialize(where N == Float)
    public static func img2col<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(values.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        
        let batchSize = values.shape[0]
        let channels = values.shape[1]
        let height = values.shape[2]
        let width = values.shape[3]
        
        let verticalStride = width
        let depthStride = width * height
        let featureMapStride = depthStride * channels
        
        let rows = result.shape[0]
        let cols = result.shape[1]
        
        let outputHeight = (height + 2 * padding - kernelHeight) / stride + 1
        let outputWidth = (width + 2 * padding - kernelWidth) / stride + 1
        
        #if DEBUG
        precondition(rows == kernelWidth * kernelHeight * channels)
        precondition(cols == outputWidth * outputHeight * batchSize)
        #endif
        
        let srcPtr = values.immutable.pointer(capacity: featureMapStride * batchSize)
        let dstPtr = result.pointer.pointer(capacity: rows * cols)
        
        let patchesPerKernel = outputWidth * outputHeight
        // let patchesPerFeatureMap = patchesPerKernel * kernelCount
        
        var rowBuffer = [Int](repeating: 0, count: rows)
        var colBuffer = [Int](repeating: 0, count: rows)
        var chanDepthStrideBuffer = [Int](repeating: 0, count: rows)
        var rColsBuffer = [Int](repeating: 0, count: rows)
        
        for r in 0 ..< rows {
            rowBuffer[r] = r / kernelWidth
            colBuffer[r] = r % kernelWidth
            chanDepthStrideBuffer[r] = (r / (kernelWidth * kernelHeight)) * depthStride
            rColsBuffer[r] = r * cols
        }
        
        for c in 0 ..< cols {
            let patchIdx = c % patchesPerKernel
            let featureMapIdx = c / patchesPerKernel
            
            let baseDstCol = patchIdx % outputWidth
            let baseDstRow = (patchIdx / outputWidth) % outputHeight
            
            let baseSrcCol = baseDstCol * stride - padding
            let baseSrcRow = baseDstRow * stride - padding
            
            let featureMap = srcPtr.advanced(by: featureMapStride * featureMapIdx)
            
            // Using a while-loop gives a ~5% performance gain
            var r = 0
            while r < rows {
                let row = baseSrcRow + rowBuffer[r]
                let col = baseSrcCol + colBuffer[r]
                
                if row < 0 || row >= height || col < 0 || col >= width {
                    dstPtr[rColsBuffer[r] + c] = 0
                } else {
                    dstPtr[rColsBuffer[r] + c] = featureMap[chanDepthStrideBuffer[r] + row * verticalStride + col]
                }
                r += 1
            }
        }
    }
    
    @_specialize(where N == Float)
    public static func col2img<N>(matrix: ShapedBuffer<N, CPU>, image: ShapedBuffer<N, CPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) {
        precondition(image.dim == 4, "im2col input must be 4D tensor (batchSize x channels x height x width)")
        
        let batchSize = image.shape[0]
        let channels = image.shape[1]
        let height = image.shape[2]
        let width = image.shape[3]
        
        let verticalStride = width
        let depthStride = width * height
        let featureMapStride = depthStride * channels
        
        let rows = matrix.shape[0]
        let cols = matrix.shape[1]
        
        let outputHeight = (height + 2 * padding - kernelHeight) / stride + 1
        let outputWidth = (width + 2 * padding - kernelWidth) / stride + 1
        
        #if DEBUG
        precondition(rows == kernelWidth * kernelHeight * channels)
        precondition(cols == outputWidth * outputHeight * batchSize)
        #endif
        
        let dstPtr = image.pointer.pointer(capacity: featureMapStride * batchSize)
        let srcPtr = matrix.immutable.pointer(capacity: rows * cols)
        
        let patchesPerKernel = outputWidth * outputHeight
        // let patchesPerFeatureMap = patchesPerKernel * kernelCount
        
        // Zero fill image
        N.fill(value: 0, result: image.pointer, count: image.count)
        
        var rowBuffer = [Int](repeating: 0, count: rows)
        var colBuffer = [Int](repeating: 0, count: rows)
        var chanDepthStrideBuffer = [Int](repeating: 0, count: rows)
        var rColsBuffer = [Int](repeating: 0, count: rows)
        
        for r in 0 ..< rows {
            rowBuffer[r] = r / kernelWidth
            colBuffer[r] = r % kernelWidth
            chanDepthStrideBuffer[r] = (r / (kernelWidth * kernelHeight)) * depthStride
            rColsBuffer[r] = r * cols
        }
        
        for c in 0 ..< cols {
            let patchIdx = c % patchesPerKernel
            let featureMapIdx = c / patchesPerKernel
            
            let baseSrcCol = patchIdx % outputWidth
            let baseSrcRow = (patchIdx / outputWidth) % outputHeight
            
            let baseDstCol = baseSrcCol * stride - padding
            let baseDstRow = baseSrcRow * stride - padding
            
            let featureMap = dstPtr.advanced(by: featureMapStride * featureMapIdx)
            
            var r = 0
            while r < rows {
                let row = baseDstRow + rowBuffer[r]
                let col = baseDstCol + colBuffer[r]
                
                if row >= 0 && row < height && col >= 0 && col < width {
                    featureMap[chanDepthStrideBuffer[r] + row * verticalStride + col] += srcPtr[rColsBuffer[r] + c]
                }
                r += 1
            }
        }
    }
    
    
    public static func conv2d<N>(values: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, padding: Int, stride: Int) where N : NumericType  {
        let tmp = CPU.Memory.allocateBuffer(
            withShape: [
                filters.shape[1] * filters.shape[2] * filters.shape[3],
                result.count / filters.shape[0]
            ],
            type: N.self
        )
        let tmpResult = CPU.Memory.allocateBuffer(
            withShape: [filters.shape[0], result.count / filters.shape[0]],
            type: N.self
        )
        defer {
            CPU.Memory.free(tmp)
            CPU.Memory.free(tmpResult)
        }
        
        img2col(values: values, result: tmp, kernelWidth: filters.shape[3], kernelHeight: filters.shape[2], padding: padding, stride: stride)
        matMul(
            lhs: filters.reshaped(to: [filters.shape[0], filters.shape[1] * filters.shape[2] * filters.shape[3]]),
            rhs: tmp,
            result: tmpResult
        )
        // result layout: [num-filters, num-patches]
        // target: [batch-size, num-filters, height, width]
        permuteAxes(values: tmpResult.reshaped(to: [result.shape[1], result.shape[0], result.shape[2], result.shape[3]]), result: result, arangement: [1, 0, 2, 3])
    }
    
    public static func revConv2d<N>(values: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func kernelGradConv2d<N>(values: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func maxPool2D<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func avgPool2D<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func revMaxPool2D<N>(values: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
    
    public static func revAvgPool2D<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int), kernelSize: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
    }
}

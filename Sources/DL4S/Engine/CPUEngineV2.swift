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
    private static func broadcast<N>(
        lhs: ShapedBuffer<N, CPU>,
        rhs: ShapedBuffer<N, CPU>,
        result: ShapedBuffer<N, CPU>,
        operator: (UnsafeBufferPointer<N>, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, Int) -> (),
        scalarOperatorA: (N, UnsafeBufferPointer<N>, UnsafeMutableBufferPointer<N>, Int) -> (),
        scalarOperatorB: (UnsafeBufferPointer<N>, N, UnsafeMutableBufferPointer<N>, Int) -> ()
    ) {
        let dim = Swift.max(lhs.dim, rhs.dim)
        
        let lhs = lhs.reshaped(to: [Int](repeating: 1, count: dim - lhs.dim) + lhs.shape)
        let rhs = rhs.reshaped(to: [Int](repeating: 1, count: dim - rhs.dim) + rhs.shape)
        
        assert(zip(lhs.shape, rhs.shape).map {$0 == $1 || $0 == 1 || $1 == 1}.allSatisfy {$0 == true})
        
        let matchingShapeSuffix = zip(lhs.shape, rhs.shape).reversed().prefix(while: {$0 == $1}).map {$1}.reversed()
        let resultShapePrefix = result.shape.prefix(result.dim - matchingShapeSuffix.count)
        
        let iterShape: [Int]
        
        let sliceCount: Int
        
        let mode: BroadcastMode
        
        if resultShapePrefix.count == 0 {
            mode = .vectorVector
            sliceCount = matchingShapeSuffix.reduce(1, *)
            iterShape = Array(resultShapePrefix)
            
        } else if lhs.shape[resultShapePrefix.count - 1] == 1 {
            mode = .scalarVector
            sliceCount = matchingShapeSuffix.reduce(1, *) * rhs.shape[resultShapePrefix.count - 1]
            iterShape = Array(resultShapePrefix.dropLast())
            
        } else {
            mode = .vectorScalar
            sliceCount = matchingShapeSuffix.reduce(1, *) * lhs.shape[resultShapePrefix.count - 1]
            iterShape = Array(resultShapePrefix.dropLast())
        }
        
        for resultIdx in iterate(iterShape) {
            let lhsIdx = zip(resultIdx, lhs.shape).map {Swift.min($0, $1 - 1)}
            let rhsIdx = zip(resultIdx, rhs.shape).map {Swift.min($0, $1 - 1)}
            
            let (lhsSlice, lhsIsCopy, _) = CPU.Memory.get(slice: lhsIdx, of: lhs.values, with: lhs.shape)
            let (rhsSlice, rhsIsCopy, _) = CPU.Memory.get(slice: rhsIdx, of: rhs.values, with: rhs.shape)
            
            let (resultSlice, resultIsCopy, _) = CPU.Memory.get(slice: resultIdx, of: result.values, with: result.shape)
            
            assert(!lhsIsCopy && !rhsIsCopy && !resultIsCopy, "[internal]: Tensor slicing violation")
            
            switch mode {
            case .vectorVector:
                `operator`(lhsSlice.immutable, rhsSlice.immutable, resultSlice.memory.bindMemory(to: N.self), sliceCount)
                
            case .scalarVector:
                scalarOperatorA(lhsSlice.pointee, rhsSlice.immutable, resultSlice.pointer, sliceCount)
                
            case .vectorScalar:
                scalarOperatorB(lhsSlice.immutable, rhsSlice.pointee, resultSlice.pointer, sliceCount)
            }
        }
    }
    
    @inline(__always)
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
    
    public static func matMul<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        N.matMul(lhs: lhs.immutable, rhs: rhs.immutable, result: result.pointer, lhsRows: lhs.shape[0], lhsCols: lhs.shape[1], rhsCols: rhs.shape[1])
    }
    
    public static func matMulAdd<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        matMul(lhs: lhs, rhs: rhs, result: result)
        broadcastAdd(lhs: result, rhs: add, result: result)
    }
    
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
    
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, CPU>, rhs: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        broadcast(
            lhs: lhs,
            rhs: rhs,
            result: result,
            operator: N.vSub,
            scalarOperatorA: {
                N.vsAdd(lhs: $1, rhs: $0, result: $2, count: $3)
                N.vNeg(val: $2.immutable, result: $2, count: $3)
            },
            scalarOperatorB: {N.vsAdd(lhs: $0, rhs: -$1, result: $2, count: $3)}
        )
    }
    
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
    
    public static func reduceSum<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
        reduce(
            values: values,
            result: result,
            axis: axis,
            reduceOperator: N.sum
        )
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) where N : NumericType {
        if let context = context {
            reduceWithContext(
                values: values,
                result: result,
                context: context,
                axis: axis,
                reduceOperator: { buffer, count -> (N, Int32) in
                    let (arg, max) = N.argmax(values: buffer, count: count)
                    return (max, Int32(arg))
                }
            )
        } else {
            reduce(values: values, result: result, axis: axis) {
                N.argmax(values: $0, count: $1).1
            }
        }
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, context: ShapedBuffer<Int32, CPU>?, axis: Int) where N : NumericType {
        if let context = context {
            reduceWithContext(
                values: values,
                result: result,
                context: context,
                axis: axis,
                reduceOperator: { buffer, count -> (N, Int32) in
                    let (arg, min) = N.argmin(values: buffer, count: count)
                    return (min, Int32(arg))
            }
            )
        } else {
            reduce(values: values, result: result, axis: axis) {
                N.argmin(values: $0, count: $1).1
            }
        }
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, axis: Int) where N : NumericType {
        reduceSum(values: values, result: result, axis: axis)
        let axisCount = values.shape[axis]
        N.vsMul(lhs: result.immutable, rhs: 1 / N(axisCount), result: result.pointer, count: result.count)
    }
    
    public static func sum<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        result.values.pointee = N.sum(val: values.immutable, count: values.count)
    }
    
    public static func mean<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) where N : NumericType {
        result.values.pointee = N.sum(val: values.immutable, count: values.count) / N(values.count)
    }
    
    @discardableResult
    public static func max<N>(values: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>) -> Int where N : NumericType {
        let (arg, max) = N.argmax(values: values.immutable, count: values.count)
        result.values.pointee = max
        return arg
    }
    
    @discardableResult
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
        let sourceMem = values.immutable
        let dstMem = result.pointer
        let dim = values.dim
        
        let shape = values.shape
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
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, CPU>, add: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, arangement: [Int]) where N : NumericType {
        let sourceMem = values.immutable
        let addMem = add.immutable
        let dstMem = result.pointer
        let dim = values.dim
        
        let shape = values.shape
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
    
    public static func conv2d<N>(values: ShapedBuffer<N, CPU>, filters: ShapedBuffer<N, CPU>, result: ShapedBuffer<N, CPU>, strides: (vertical: Int, horizontal: Int)) where N : NumericType {
        fatalError("\(#function) is not implemented for type \(self)")
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

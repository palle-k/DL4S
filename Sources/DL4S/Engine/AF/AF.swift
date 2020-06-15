//
//  File.swift
//  
//
//  Created by Palle Klewitz on 20.05.20.
//

import Foundation
#if AF_ENABLE
import AF

public protocol AFNumeric {
    static var GPUType: af_dtype { get }
}
extension Float: AFNumeric {
    public static var GPUType: af_dtype {
        return f32
    }
}
extension Double: AFNumeric {
    public static var GPUType: af_dtype {
        return f64
    }
}
extension Int32: AFNumeric {
    public static var GPUType: af_dtype {
        return s32
    }
}

public struct GPU: DeviceType {
    public typealias Memory = AFMemoryOps
    public typealias Engine = AFEngine
    
    public static func setOpenCL() {
        af_set_backend(AF_BACKEND_OPENCL)
    }
    
    public static func setCUDA() {
        af_set_backend(AF_BACKEND_CUDA)
    }
    
    public static func printInfo() {
        af_info()
    }
    
    public static func printMemInfo() {
        af_print_mem_info("MemInfo:", 0)
    }
}

public struct AFMemoryOps: MemoryOperatorsType {
    public typealias RawBuffer = d4af_array
    public typealias Device = GPU
}

public extension AFMemoryOps {
    static func allocateBuffer<Element: NumericType>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, GPU> {
        return Buffer(memory: d4af_allocate(dim_t(capacity), type.GPUType))
    }
    
    static func allocateBuffer<Element: NumericType>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, GPU> {
        return ShapedBuffer(values: allocateBuffer(withCapacity: shape.reduce(1, *), type: Element.self), shape: shape)
    }
    
    static func free<Element: NumericType>(_ buffer: Buffer<Element, GPU>) {
        d4af_free(buffer.memory)
    }
    
    static func free<Element: NumericType>(_ buffer: ShapedBuffer<Element, GPU>) {
        d4af_free(buffer.values.memory)
    }
    
    static func assign<Element: NumericType>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, GPU>, count: Int) {
        d4af_assign_h2d(destination.memory, UnsafeRawPointer(source.pointer(capacity: count)), count * MemoryLayout<Element>.stride)
    }
    
    static func assign<Element: NumericType>(from source: Buffer<Element, GPU>, to destination: Buffer<Element, GPU>, count: Int) {
        d4af_assign_d2d(destination.memory, source.memory)
    }
    
    static func assign<Element: NumericType>(from source: Buffer<Element, GPU>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        d4af_assign_d2h(UnsafeMutableRawPointer(destination.pointer(capacity: count)), source.memory)
    }
    
    static func getValue<Element: NumericType>(from source: Buffer<Element, GPU>) -> Element {
        if Element.self == Float.self {
            return Element(d4af_get_pointee_32f(source.memory))
        } else if Element.self == Int32.self {
            return Element(d4af_get_pointee_32s(source.memory))
        } else if Element.self == Double.self {
            return Element(d4af_get_pointee_64f(source.memory))
        } else {
            fatalError("Unsupported element type")
        }
    }
    
    static func getSize<Element: NumericType>(of buffer: Buffer<Element, GPU>) -> Int {
        return d4af_get_size(buffer.memory)
    }
    
    static func get<Element: NumericType>(slice: [Int?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        let slice = slice + Array(repeating: nil, count: shape.count - slice.count)
        var resultShape = shape
        for (offset, x) in slice.enumerated().reversed() where x != nil {
            resultShape.remove(at: offset)
        }
        let result = allocateBuffer(withShape: resultShape, type: Element.self).values
        
        let paddedShape = Array(repeating: 1, count: 4 - shape.count) + shape
        let paddedIndex = Array(repeating: nil, count: 4 - slice.count) + slice
        
        d4af_subscript(result.memory, buffer.memory, paddedShape.map(Int32.init).reversed(), paddedIndex.map {Int32($0 ?? -1)}.reversed())
        return (result, true, resultShape)
    }
    
    static func get<Element: NumericType>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, GPU>, with shape: [Int]) -> (Buffer<Element, GPU>, Bool, [Int]) {
        let slice = slice + Array(repeating: nil, count: shape.count - slice.count)
        let resultShape = zip(shape, slice).map {$1?.count ?? $0}
        let result = allocateBuffer(withShape: resultShape, type: Element.self).values
        
        let paddedSlice = Array(repeating: nil, count: 4 - slice.count) + slice
        let paddedShape = Array(repeating: 1, count: 4 - shape.count) + shape
        
        d4af_subscript_range(result.memory, buffer.memory, paddedShape.map(Int32.init).reversed(), paddedSlice.map {Int32($0?.lowerBound ?? -1)}.reversed(), paddedSlice.map {Int32($0?.upperBound ?? -1)}.reversed())
        return (result, true, resultShape)
    }
    
    static func set<Element: NumericType>(slice: [Int?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        let paddedShape = Array(repeating: 1, count: 4 - dstShape.count) + dstShape
        let paddedIndex = Array(repeating: nil, count: 4 - dstShape.count) + slice + Array(repeating: nil, count: dstShape.count - slice.count)
        
        d4af_subscript_write(
            buffer.memory,
            source.memory,
            paddedShape.reversed().map(dim_t.init),
            paddedIndex.reversed().map {Int32($0 ?? -1)}
        )
    }
    
    static func set<Element: NumericType>(slice: [Range<Int>?], of buffer: Buffer<Element, GPU>, with dstShape: [Int], from source: Buffer<Element, GPU>, with sourceShape: [Int]) {
        let paddedShape = Array(repeating: 1, count: 4 - dstShape.count) + dstShape
        let paddedIndex = Array(repeating: nil, count: 4 - dstShape.count) + slice + Array(repeating: nil, count: dstShape.count - slice.count)
        
        d4af_subscript_write_range(
            buffer.memory,
            source.memory,
            paddedShape.reversed().map(dim_t.init),
            paddedIndex.reversed().map {Int32($0?.lowerBound ?? -1)},
            paddedIndex.reversed().map {Int32($0?.upperBound ?? -1)}
        )
    }
    
    static func setPointee<Element: NumericType>(of buffer: Buffer<Element, GPU>, to newValue: Element) {
        if Element.self == Float.self {
            d4af_set_pointee_32f(buffer.memory, newValue.floatValue)
        } else if Element.self == Int32.self {
            d4af_set_pointee_64f(buffer.memory, newValue.doubleValue)
        } else if Element.self == Double.self {
            d4af_set_pointee_32s(buffer.memory, newValue.intValue)
        } else {
            fatalError("Unsupported element type")
        }
    }
    
    static func advance<Element: NumericType>(buffer: Buffer<Element, GPU>, by advancement: Int) -> Buffer<Element, GPU> {
        return get(slice: [advancement ..< buffer.count], of: buffer, with: [buffer.count]).0
    }
}

public struct AFEngine: EngineType {
    public typealias Device = GPU
    
    public static func fill<N>(value: N, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        if N.self == Float.self {
            d4af_fill_32f(result.memory, Float(element: value))
        } else if N.self == Double.self {
            d4af_fill_64f(result.memory, Double(element: value))
        } else if N.self == Int32.self {
            d4af_fill_32s(result.memory, Int32(element: value))
        } else {
            fatalError("Unsupported type \(N.self)")
        }
    }
    
    public static func fillRandomNormal<N>(result: Buffer<N, GPU>, mean: N, stdev: N, count: Int) where N : RandomizableType {
        if N.self == Float.self {
            d4af_randn_32f(result.memory, mean.floatValue, stdev.floatValue, dim_t(count))
        } else if N.self == Double.self {
            d4af_randn_64f(result.memory, mean.doubleValue, stdev.doubleValue, dim_t(count))
        } else if N.self == Int32.self {
            d4af_randn_32s(result.memory, mean.intValue, stdev.intValue, dim_t(count))
        } else {
            fatalError("Unsupported type \(N.self)")
        }
    }
    
    public static func fillRandomUniform<N>(result: Buffer<N, GPU>, lowerBound: N, upperBound: N, count: Int) where N : RandomizableType {
        if N.self == Float.self {
            d4af_randu_32f(result.memory, lowerBound.floatValue, upperBound.floatValue, dim_t(count))
        } else if N.self == Double.self {
            d4af_randu_64f(result.memory, lowerBound.doubleValue, upperBound.doubleValue, dim_t(count))
        } else if N.self == Int32.self {
            d4af_randu_32s(result.memory, lowerBound.intValue, upperBound.intValue, dim_t(count))
        } else {
           fatalError("Unsupported type \(N.self)")
        }
    }
    
    public static func fillRandomBernoulli<N>(result: Buffer<N, GPU>, prob: Float, count: Int) where N : NumericType {
        d4af_randb(result.memory, prob, N.GPUType, dim_t(count))
    }
    
    public static func vAdd<N>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        d4af_add(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vNeg<N>(val: Buffer<N, GPU>, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        d4af_neg(result.memory, val.memory)
    }
    
    public static func vSub<N>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        d4af_sub(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vMul<N>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        d4af_mul(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vDiv<N>(lhs: Buffer<N, GPU>, rhs: Buffer<N, GPU>, result: Buffer<N, GPU>, count: Int) where N : NumericType {
        d4af_div(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func gemm<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
        d4af_gemm(
            result.values.memory,
            lhs.values.memory,
            rhs.values.memory,
            transposeFirst,
            transposeSecond,
            Float(element: alpha),
            Float(element: beta),
            dim_t(lhs.shape[0]),
            dim_t(lhs.shape[1]),
            dim_t(rhs.shape[0]),
            dim_t(rhs.shape[1]),
            dim_t(result.shape[0]),
            dim_t(result.shape[1])
        )
    }
    
    public static func band<N>(buffer: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, belowDiagonal: Int?, aboveDiagonal: Int?) where N : NumericType {
        precondition(buffer.dim == 2, "Band input must be a matrix")
        d4af_band(result.values.memory, buffer.values.memory, Int32(buffer.shape[0]), Int32(buffer.shape[1]), belowDiagonal.map(Int32.init) ?? Int32.max, aboveDiagonal.map(Int32.init) ?? Int32.max)
    }
    
    public static func broadcastAdd<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        let paddedLhsShape = Array(repeating: 1, count: result.dim - lhs.dim) + lhs.shape
        let paddedRhsShape = Array(repeating: 1, count: result.dim - rhs.dim) + rhs.shape
        d4af_broadcast_add(
            result.values.memory,
            lhs.values.memory,
            rhs.values.memory,
            dim_t(result.dim),
            paddedLhsShape.reversed().map(dim_t.init),
            paddedRhsShape.reversed().map(dim_t.init)
        )
    }
    
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        let paddedLhsShape = Array(repeating: 1, count: result.dim - lhs.dim) + lhs.shape
        let paddedRhsShape = Array(repeating: 1, count: result.dim - rhs.dim) + rhs.shape
        d4af_broadcast_sub(
            result.values.memory,
            lhs.values.memory,
            rhs.values.memory,
            dim_t(result.dim),
            paddedLhsShape.reversed().map(dim_t.init),
            paddedRhsShape.reversed().map(dim_t.init)
        )
    }
    
    public static func broadcastMul<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        let paddedLhsShape = Array(repeating: 1, count: result.dim - lhs.dim) + lhs.shape
        let paddedRhsShape = Array(repeating: 1, count: result.dim - rhs.dim) + rhs.shape
        d4af_broadcast_mul(
            result.values.memory,
            lhs.values.memory,
            rhs.values.memory,
            dim_t(result.dim),
            paddedLhsShape.reversed().map(dim_t.init),
            paddedRhsShape.reversed().map(dim_t.init)
        )
    }
    
    public static func broadcastDiv<N>(lhs: ShapedBuffer<N, GPU>, rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        let paddedLhsShape = Array(repeating: 1, count: result.dim - lhs.dim) + lhs.shape
        let paddedRhsShape = Array(repeating: 1, count: result.dim - rhs.dim) + rhs.shape
        d4af_broadcast_div(
            result.values.memory,
            lhs.values.memory,
            rhs.values.memory,
            dim_t(result.dim),
            paddedLhsShape.reversed().map(dim_t.init),
            paddedRhsShape.reversed().map(dim_t.init)
        )
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        d4af_reduce_sum(
            result.values.memory,
            values.values.memory,
            dim_t(values.dim),
            values.shape.reversed().map(dim_t.init),
            dim_t(values.dim - axis - 1)
        )
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        if let context = context {
            d4af_reduce_max_ctx(
                result.values.memory,
                context.values.memory,
                values.values.memory,
                dim_t(values.dim),
                values.shape.reversed().map(dim_t.init),
                dim_t(values.dim - axis - 1)
            )
        } else {
            d4af_reduce_max(
                result.values.memory,
                values.values.memory,
                dim_t(values.dim),
                values.shape.reversed().map(dim_t.init),
                dim_t(values.dim - axis - 1)
            )
        }
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axis: Int) where N : NumericType {
        if let context = context {
            d4af_reduce_min_ctx(
                result.values.memory,
                context.values.memory,
                values.values.memory,
                dim_t(values.dim),
                values.shape.reversed().map(dim_t.init),
                dim_t(values.dim - axis - 1)
            )
        } else {
            d4af_reduce_min(
                result.values.memory,
                values.values.memory,
                dim_t(values.dim),
                values.shape.reversed().map(dim_t.init),
                dim_t(values.dim - axis - 1)
            )
        }
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        d4af_reduce_mean(
            result.values.memory,
            values.values.memory,
            dim_t(values.dim),
            values.shape.reversed().map(dim_t.init),
            dim_t(values.dim - axis - 1)
        )
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        d4af_reduce_sum_multi(
            result.values.memory,
            values.values.memory,
            dim_t(values.dim),
            values.shape.reversed().map(dim_t.init),
            dim_t(axes.count),
            axes.map {dim_t(values.dim - $0 - 1)}.reversed()
        )
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        if axes.count == 1 {
            reduceMax(values: values, result: result, context: context, axis: axes[0])
        } else {
            fatalError("\(#function) is unavailable for multiple axes")
        }
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>?, axes: [Int]) where N : NumericType {
        if axes.count == 1 {
            reduceMin(values: values, result: result, context: context, axis: axes[0])
        } else {
            fatalError("\(#function) is unavailable for multiple axes")
        }
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, axes: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func sum<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_sum_all(result.values.memory, values.values.memory)
    }
    
    public static func mean<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_mean_all(result.values.memory, values.values.memory)
    }
    
    public static func max<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        return Int(d4af_argmax(result.values.memory, values.values.memory))
    }
    
    public static func min<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) -> Int where N : NumericType {
        return Int(d4af_argmin(result.values.memory, values.values.memory))
    }
    
    public static func argmax<N>(values: Buffer<N, GPU>, count: Int) -> (Int, N) where N : NumericType {
        let tmp = GPU.Memory.allocateBuffer(withCapacity: 1, type: N.self)
        let result = Int(d4af_argmax(tmp.memory, values.memory))
        let value = GPU.Memory.getValue(from: tmp)
        GPU.Memory.free(tmp)
        return (result, value)
    }
    
    public static func exp<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_exp(result.values.memory, values.values.memory)
    }
    
    public static func log<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_log(result.values.memory, values.values.memory)
    }
    
    public static func sqrt<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_sqrt(result.values.memory, values.values.memory)
    }
    
    public static func relu<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_relu(result.values.memory, values.values.memory)
    }
    
    public static func heaviside<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_heaviside(result.values.memory, values.values.memory)
    }
    
    public static func sin<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_sin(result.values.memory, values.values.memory)
    }
    
    public static func cos<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_cos(result.values.memory, values.values.memory)
    }
    
    public static func tan<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_tan(result.values.memory, values.values.memory)
    }
    
    public static func sinh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_sinh(result.values.memory, values.values.memory)
    }
    
    public static func cosh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_cosh(result.values.memory, values.values.memory)
    }
    
    public static func tanh<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_tanh(result.values.memory, values.values.memory)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_max(result.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_max_ctx(result.values.memory, context.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_min(result.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, GPU>, _ rhs: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, context: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_min_ctx(result.values.memory, context.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func scatter<N>(reduced: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: ShapedBuffer<N, GPU>, axis: Int, ignoreIndex: Int32) where N : NumericType {
        d4af_scatter(
            result.values.memory,
            reduced.values.memory,
            context.values.memory,
            (Array(repeating: 1, count: 4 - result.dim) + result.shape).map(dim_t.init).reversed(),
            Int32(result.dim - axis - 1),
            ignoreIndex
        )
    }
    
    public static func gather<N>(expanded: ShapedBuffer<N, GPU>, context: ShapedBuffer<Int32, GPU>, result: ShapedBuffer<N, GPU>, axis: Int, ignoreIndex: Int32) where N : NumericType {
        d4af_gather(
            result.values.memory,
            expanded.values.memory,
            context.values.memory,
            (Array(repeating: 1, count: 4 - expanded.dim) + expanded.shape).map(dim_t.init).reversed(),
            Int32(expanded.dim - axis - 1),
            ignoreIndex
        )
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N : NumericType {
        let paddedShape = Array(repeating: 1, count: 4 - values.shape.count) + values.shape
        let paddedArangement = (0 ..< (4 - values.shape.count)) + arangement.map {$0 + (4 - values.shape.count)}
        let invertedArangement = paddedArangement.reversed().map {
            3 - $0
        }
        
        d4af_permute(result.values.memory, values.values.memory, 4, paddedShape.reversed().map(dim_t.init), invertedArangement.map(dim_t.init))
    }
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, arangement: [Int]) where N : NumericType {
        let paddedShape = Array(repeating: 1, count: values.shape.count) + values.shape
        let paddedArangement = arangement + Array(0 ..< 4).dropFirst(arangement.count)
        let invertedArangement = paddedArangement.reversed().map {
            3 - $0
        }
        
        d4af_permute_add(result.values.memory, values.values.memory, 4, paddedShape.reversed().map(dim_t.init), invertedArangement.map(dim_t.init), add.values.memory)
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        d4af_subscript(
            result.values.memory,
            values.values.memory,
            (Array(repeating: 1, count: 4 - values.dim) + values.shape).map(Int32.init).reversed(),
            (Array(repeating: nil, count: 4 - index.count) + index).map {Int32($0 ?? -1)}.reversed()
        )
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        d4af_subscript_write(
            result.values.memory,
            values.values.memory,
            (Array(repeating: 1, count: 4 - result.dim) + result.shape).map(dim_t.init).reversed(),
            (index + Array(repeating: nil, count: 4 - index.count)).map {Int32($0 ?? -1)}.reversed()
        )
    }
    
    public static func subscriptReadAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func subscriptWriteAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func reverse<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_reverse(result.values.memory, values.values.memory)
    }
    
    public static func reverseAdd<N>(values: ShapedBuffer<N, GPU>, add: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>) where N : NumericType {
        d4af_reverse_add(result.values.memory, values.values.memory, add.values.memory)
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, GPU>], result: ShapedBuffer<N, GPU>, axis: Int) where N : NumericType {
        d4af_stack(
            result.values.memory,
            buffers.map {$0.values.memory},
            UInt32(buffers.count),
            buffers.flatMap { buffer in
                (Array(repeating: 1, count: 4 - buffer.dim) + buffer.shape.map(dim_t.init)).reversed()
            },
            Int32(result.dim - axis - 1)
        )
    }
    
    public static func unstackAdd<N>(stacked: ShapedBuffer<N, GPU>, add: [ShapedBuffer<N, GPU>], result: [ShapedBuffer<N, GPU>], axis: Int) where N : NumericType {
        d4af_unstack_add(
            result.map {$0.values.memory},
            add.map {$0.values.memory},
            result.count,
            result.map {dim_t($0.shape[axis])},
            stacked.values.memory,
            Int64(stacked.dim),
            stacked.shape.reversed().map(dim_t.init),
            Int32(stacked.dim - axis - 1)
        )
    }
    
    public static func unstack<N>(stacked: ShapedBuffer<N, GPU>, result: [ShapedBuffer<N, GPU>], axis: Int) where N : NumericType {
        d4af_unstack(
            result.map {$0.values.memory},
            result.count,
            result.map {dim_t($0.shape[axis])},
            stacked.values.memory,
            Int64(stacked.dim),
            stacked.shape.reversed().map(dim_t.init),
            Int32(stacked.dim - axis - 1)
        )
    }
    
    public static func arange<N>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, GPU>) where N : NumericType {
        if N.self == Float.self {
            d4af_arange_32f(result.values.memory, lowerBound as! Float, upperBound as! Float, dim_t(result.count))
        } else if N.self == Double.self {
            d4af_arange_64f(result.values.memory, lowerBound as! Double, upperBound as! Double, dim_t(result.count))
        } else if N.self == Int32.self {
            d4af_arange_32s(result.values.memory, lowerBound as! Int32, upperBound as! Int32, dim_t(result.count))
        } else {
            fatalError("\(#function) not available for type \(N.self)")
        }
    }
    
    public static func img2col<N>(values: ShapedBuffer<N, GPU>, result: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        d4af_im2col(
            result.values.memory,
            values.values.memory,
            dim_t(values.shape[0]),
            dim_t(values.shape[1]),
            dim_t(values.shape[2]),
            dim_t(values.shape[3]),
            dim_t(kernelWidth),
            dim_t(kernelHeight),
            dim_t(stride),
            dim_t(padding)
        )
    }
    
    public static func col2img<N>(matrix: ShapedBuffer<N, GPU>, image: ShapedBuffer<N, GPU>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
        d4af_col2im(
            image.values.memory,
            matrix.values.memory,
            dim_t(image.shape[0]),
            dim_t(image.shape[1]),
            dim_t(image.shape[2]),
            dim_t(image.shape[3]),
            dim_t(kernelWidth),
            dim_t(kernelHeight),
            dim_t(stride),
            dim_t(padding)
        )
    }
    
}

#endif

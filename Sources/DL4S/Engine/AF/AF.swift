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
    static var arrayfireType: af_dtype { get }
}
extension Float: AFNumeric {
    public static var arrayfireType: af_dtype {
        return f32
    }
}
extension Double: AFNumeric {
    public static var arrayfireType: af_dtype {
        return f64
    }
}
extension Int32: AFNumeric {
    public static var arrayfireType: af_dtype {
        return s32
    }
}

public struct ArrayFire: DeviceType {
    public typealias Memory = AFMemoryOps
    public typealias Engine = AFEngine
    
    public static func setOpenCL() {
        af_set_backend(AF_BACKEND_OPENCL)
    }
    
    public static func setCUDA() {
        af_set_backend(AF_BACKEND_CUDA)
    }
    
    public static func setCPU() {
        af_set_backend(AF_BACKEND_CPU)
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
    public typealias Device = ArrayFire
}

public extension AFMemoryOps {
    static func allocateBuffer<Element: NumericType>(withCapacity capacity: Int, type: Element.Type) -> Buffer<Element, ArrayFire> {
        return Buffer(memory: d4af_allocate(dim_t(capacity), type.arrayfireType))
    }
    
    static func allocateBuffer<Element: NumericType>(withShape shape: [Int], type: Element.Type) -> ShapedBuffer<Element, ArrayFire> {
        return ShapedBuffer(values: allocateBuffer(withCapacity: shape.reduce(1, *), type: Element.self), shape: shape)
    }
    
    static func free<Element: NumericType>(_ buffer: Buffer<Element, ArrayFire>) {
        d4af_free(buffer.memory)
    }
    
    static func free<Element: NumericType>(_ buffer: ShapedBuffer<Element, ArrayFire>) {
        d4af_free(buffer.values.memory)
    }
    
    static func assign<Element: NumericType>(from source: UnsafeBufferPointer<Element>, to destination: Buffer<Element, ArrayFire>, count: Int) {
        d4af_assign_h2d(destination.memory, UnsafeRawPointer(source.pointer(capacity: count)), count * MemoryLayout<Element>.stride)
    }
    
    static func assign<Element: NumericType>(from source: Buffer<Element, ArrayFire>, to destination: Buffer<Element, ArrayFire>, count: Int) {
        d4af_assign_d2d(destination.memory, source.memory)
    }
    
    static func assign<Element: NumericType>(from source: Buffer<Element, ArrayFire>, to destination: UnsafeMutableBufferPointer<Element>, count: Int) {
        d4af_assign_d2h(UnsafeMutableRawPointer(destination.pointer(capacity: count)), source.memory)
    }
    
    static func getValue<Element: NumericType>(from source: Buffer<Element, ArrayFire>) -> Element {
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
    
    static func getSize<Element: NumericType>(of buffer: Buffer<Element, ArrayFire>) -> Int {
        return d4af_get_size(buffer.memory)
    }
    
    static func get<Element: NumericType>(slice: [Int?], of buffer: Buffer<Element, ArrayFire>, with shape: [Int]) -> (Buffer<Element, ArrayFire>, Bool, [Int]) {
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
    
    static func get<Element: NumericType>(slice: [(CountableRange<Int>)?], of buffer: Buffer<Element, ArrayFire>, with shape: [Int]) -> (Buffer<Element, ArrayFire>, Bool, [Int]) {
        let slice = slice + Array(repeating: nil, count: shape.count - slice.count)
        let resultShape = zip(shape, slice).map {$1?.count ?? $0}
        let result = allocateBuffer(withShape: resultShape, type: Element.self).values
        
        let paddedSlice = Array(repeating: nil, count: 4 - slice.count) + slice
        let paddedShape = Array(repeating: 1, count: 4 - shape.count) + shape
        
        d4af_subscript_range(result.memory, buffer.memory, paddedShape.map(Int32.init).reversed(), paddedSlice.map {Int32($0?.lowerBound ?? -1)}.reversed(), paddedSlice.map {Int32($0?.upperBound ?? -1)}.reversed())
        return (result, true, resultShape)
    }
    
    static func set<Element: NumericType>(slice: [Int?], of buffer: Buffer<Element, ArrayFire>, with dstShape: [Int], from source: Buffer<Element, ArrayFire>, with sourceShape: [Int]) {
        fatalError("\(#function) unavailable")
    }
    
    static func set<Element: NumericType>(slice: [Range<Int>?], of buffer: Buffer<Element, ArrayFire>, with dstShape: [Int], from source: Buffer<Element, ArrayFire>, with sourceShape: [Int]) {
        fatalError("\(#function) unavailable")
    }
    
    static func setPointee<Element: NumericType>(of buffer: Buffer<Element, ArrayFire>, to newValue: Element) {
        // let status = af_write_array(buffer.memory, [newValue], MemoryLayout<Element>.stride, afHost)
        fatalError("\(#function) unavailable")
    }
    
    static func advance<Element: NumericType>(buffer: Buffer<Element, ArrayFire>, by advancement: Int) -> Buffer<Element, ArrayFire> {
        return get(slice: [advancement ..< buffer.count], of: buffer, with: [buffer.count]).0
    }
}

public struct AFEngine: EngineType {
    public typealias Device = ArrayFire
    
    public static func fill<N>(value: N, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        if N.self == Float.self {
            d4af_fill_32f(result.memory, Float(element: value))
        } else if N.self == Double.self {
            d4af_fill_64f(result.memory, Double(element: value))
        } else if N.self == Int32.self {
            d4af_fill_32s(result.memory, Int32(element: value))
        }
    }
    
    public static func vAdd<N>(lhs: Buffer<N, ArrayFire>, rhs: Buffer<N, ArrayFire>, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        d4af_add(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vNeg<N>(val: Buffer<N, ArrayFire>, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        d4af_neg(result.memory, val.memory)
    }
    
    public static func vSub<N>(lhs: Buffer<N, ArrayFire>, rhs: Buffer<N, ArrayFire>, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        d4af_sub(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vMul<N>(lhs: Buffer<N, ArrayFire>, rhs: Buffer<N, ArrayFire>, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        d4af_mul(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func vDiv<N>(lhs: Buffer<N, ArrayFire>, rhs: Buffer<N, ArrayFire>, result: Buffer<N, ArrayFire>, count: Int) where N : NumericType {
        d4af_div(result.memory, lhs.memory, rhs.memory)
    }
    
    public static func gemm<N>(lhs: ShapedBuffer<N, ArrayFire>, rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, alpha: N, beta: N, transposeFirst: Bool, transposeSecond: Bool) where N : NumericType {
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
    
    public static func band<N>(buffer: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, belowDiagonal: Int?, aboveDiagonal: Int?) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func broadcastAdd<N>(lhs: ShapedBuffer<N, ArrayFire>, rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
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
    
    public static func broadcastSub<N>(lhs: ShapedBuffer<N, ArrayFire>, rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
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
    
    public static func broadcastMul<N>(lhs: ShapedBuffer<N, ArrayFire>, rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
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
    
    public static func broadcastDiv<N>(lhs: ShapedBuffer<N, ArrayFire>, rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
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
    
    public static func reduceSum<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axis: Int) where N : NumericType {
        d4af_reduce_sum(
            result.values.memory,
            values.values.memory,
            dim_t(values.dim),
            values.shape.reversed().map(dim_t.init),
            dim_t(values.dim - axis - 1)
        )
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>?, axis: Int) where N : NumericType {
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
    
    public static func reduceMin<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>?, axis: Int) where N : NumericType {
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
    
    public static func reduceMean<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axis: Int) where N : NumericType {
        d4af_reduce_mean(
            result.values.memory,
            values.values.memory,
            dim_t(values.dim),
            values.shape.reversed().map(dim_t.init),
            dim_t(values.dim - axis - 1)
        )
    }
    
    public static func reduceSum<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axes: [Int]) where N : NumericType {
        if let firstAxis = axes.first, axes.count == 1 {
            reduceSum(values: values, result: result, axis: firstAxis)
        } else {
            fatalError("\(#function) unavailable")
        }
    }
    
    public static func reduceMax<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func reduceMin<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>?, axes: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func reduceMean<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axes: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func sum<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_sum_all(result.values.memory, values.values.memory)
    }
    
    public static func mean<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_mean_all(result.values.memory, values.values.memory)
    }
    
    public static func max<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) -> Int where N : NumericType {
        return Int(d4af_argmax(result.values.memory, values.values.memory))
    }
    
    public static func min<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) -> Int where N : NumericType {
        return Int(d4af_argmin(result.values.memory, values.values.memory))
    }
    
    public static func argmax<N>(values: Buffer<N, ArrayFire>, count: Int) -> (Int, N) where N : NumericType {
        let tmp = ArrayFire.Memory.allocateBuffer(withCapacity: 1, type: N.self)
        let result = Int(d4af_argmax(tmp.memory, values.memory))
        let value = ArrayFire.Memory.getValue(from: tmp)
        ArrayFire.Memory.free(tmp)
        return (result, value)
    }
    
    public static func exp<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_exp(result.values.memory, values.values.memory)
    }
    
    public static func log<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_log(result.values.memory, values.values.memory)
    }
    
    public static func sqrt<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_sqrt(result.values.memory, values.values.memory)
    }
    
    public static func relu<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_relu(result.values.memory, values.values.memory)
    }
    
    public static func heaviside<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_heaviside(result.values.memory, values.values.memory)
    }
    
    public static func sin<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_sin(result.values.memory, values.values.memory)
    }
    
    public static func cos<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_cos(result.values.memory, values.values.memory)
    }
    
    public static func tan<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_tan(result.values.memory, values.values.memory)
    }
    
    public static func sinh<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_sinh(result.values.memory, values.values.memory)
    }
    
    public static func cosh<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_cosh(result.values.memory, values.values.memory)
    }
    
    public static func tanh<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_tanh(result.values.memory, values.values.memory)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, ArrayFire>, _ rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_max(result.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func max<N>(_ lhs: ShapedBuffer<N, ArrayFire>, _ rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_max_ctx(result.values.memory, context.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, ArrayFire>, _ rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_min(result.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func min<N>(_ lhs: ShapedBuffer<N, ArrayFire>, _ rhs: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_min_ctx(result.values.memory, context.values.memory, lhs.values.memory, rhs.values.memory)
    }
    
    public static func scatter<N>(reduced: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axis: Int, ignoreIndex: Int32) where N : NumericType {
        d4af_scatter(
            result.values.memory,
            reduced.values.memory,
            context.values.memory,
            (Array(repeating: 1, count: 4 - result.dim) + result.shape).map(dim_t.init).reversed(),
            Int32(result.dim - axis - 1)
        )
    }
    
    public static func gather<N>(expanded: ShapedBuffer<N, ArrayFire>, context: ShapedBuffer<Int32, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, axis: Int, ignoreIndex: Int32) where N : NumericType {
        d4af_gather(
            result.values.memory,
            expanded.values.memory,
            context.values.memory,
            (Array(repeating: 1, count: 4 - expanded.dim) + expanded.shape).map(dim_t.init).reversed(),
            Int32(expanded.dim - axis - 1)
        )
    }
    
    public static func permuteAxes<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, arangement: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func permuteAxesAdd<N>(values: ShapedBuffer<N, ArrayFire>, add: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, arangement: [Int]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func subscriptRead<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, index: [Int?]) where N : NumericType {
        d4af_subscript(
            result.values.memory,
            values.values.memory,
            (Array(repeating: 1, count: 4 - values.dim) + values.shape).map(Int32.init).reversed(),
            (Array(repeating: nil, count: 4 - index.count) + index).map {Int32($0 ?? -1)}.reversed()
        )
    }
    
    public static func subscriptWrite<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, index: [Int?]) where N : NumericType {
        d4af_subscript_write(
            result.values.memory,
            values.values.memory,
            (Array(repeating: 1, count: 4 - result.dim) + result.shape).map(Int32.init).reversed(),
            (Array(repeating: nil, count: 4 - index.count) + index).map {Int32($0 ?? -1)}.reversed()
        )
    }
    
    public static func subscriptReadAdd<N>(values: ShapedBuffer<N, ArrayFire>, add: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func subscriptWriteAdd<N>(values: ShapedBuffer<N, ArrayFire>, add: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, index: [Int?]) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func reverse<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_reverse(result.values.memory, values.values.memory)
    }
    
    public static func reverseAdd<N>(values: ShapedBuffer<N, ArrayFire>, add: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
        d4af_reverse_add(result.values.memory, values.values.memory, add.values.memory)
    }
    
    public static func stack<N>(buffers: [ShapedBuffer<N, ArrayFire>], result: ShapedBuffer<N, ArrayFire>, axis: Int) where N : NumericType {
        d4af_stack(
            result.values.memory,
            buffers.map {$0.values.memory},
            UInt32(buffers.count),
            buffers.flatMap { buffer in
                (Array(repeating: 1, count: 4 - buffer.dim) + buffer.shape.map(dim_t.init)).reversed()
            },
            Int32(axis)
        )
    }
    
    public static func unstackAdd<N>(stacked: ShapedBuffer<N, ArrayFire>, add: [ShapedBuffer<N, ArrayFire>], result: [ShapedBuffer<N, ArrayFire>], axis: Int) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func unstack<N>(stacked: ShapedBuffer<N, ArrayFire>, result: [ShapedBuffer<N, ArrayFire>], axis: Int) where N : NumericType {
        fatalError("\(#function) unavailable")
    }
    
    public static func arange<N>(lowerBound: N, upperBound: N, result: ShapedBuffer<N, ArrayFire>) where N : NumericType {
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
    
    public static func img2col<N>(values: ShapedBuffer<N, ArrayFire>, result: ShapedBuffer<N, ArrayFire>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
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
    
    public static func col2img<N>(matrix: ShapedBuffer<N, ArrayFire>, image: ShapedBuffer<N, ArrayFire>, kernelWidth: Int, kernelHeight: Int, padding: Int, stride: Int) where N : NumericType {
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

public struct AFRawBuffer {
    fileprivate var array: d4af_array
    
    init(array: d4af_array) {
        self.array = array
    }
}


#endif

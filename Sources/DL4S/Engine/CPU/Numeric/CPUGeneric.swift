//
//  CPUGeneric.swift
//  DL4S
//
//  Created by Palle Klewitz on 31.10.19.
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

#if MKL_ENABLE
import MKL
#elseif canImport(Accelerate)
import Accelerate
#endif

struct D4Img2ColSetup {
    let batch_size: Int
    let channels: Int
    let height: Int
    let width: Int
    let kernel_height: Int
    let kernel_width: Int
    let padding: Int
    let stride: Int
}

public extension CPUNumeric {
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func img2col(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        let src = values.baseAddress!
        let dst = result.baseAddress!
        
        let setup = D4Img2ColSetup(
            batch_size: batchSize,
            channels: channels,
            height: height,
            width: width,
            kernel_height: kernelHeight,
            kernel_width: kernelWidth,
            padding: padding,
            stride: stride
        )
        
        let depth_stride = setup.width * setup.height;
        let featuremap_stride = depth_stride * setup.channels;
        
        let output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
        let output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
        let dst_batch_stride = output_width * output_height;
        let dst_full_stride = dst_batch_stride * setup.batch_size;
        
        for k in 0 ..< setup.kernel_width &* setup.kernel_height &* setup.channels {
            let kx = k % setup.kernel_width
            let kyz = k / setup.kernel_width
            let ky = kyz % setup.kernel_height
            let kz = kyz / setup.kernel_height
            for b in 0 ..< setup.batch_size {
                
                for y in 0 ..< output_height {
                    let in_y = y &* setup.stride &- setup.padding &+ ky
                    
                    if in_y >= 0 && in_y < setup.height {
                        for x in 0 ..< output_width {
                            let in_x = x &* setup.stride &- setup.padding &+ kx
                            let input: Self
                            if (in_x >= 0 && in_x < setup.width) {
                                input = src[in_x &+ in_y &* setup.width &+ kz * depth_stride &+ b &* featuremap_stride]
                            } else {
                                input = Self.zero
                            }
                            dst[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width &+ x] = input
                        }
                    } else {
                        if Self.self == Float.self {
                            let dst_float = (dst as! UnsafeMutablePointer<Float>)
                            #if MKL_ENABLE
                            ippsSet_32f(0, &dst_float[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width], Int32(output_width))
                            #elseif canImport(Accelerate)
                            vDSP_vfill([0], &dst_float[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width], 1, UInt(output_width))
                            #else
                            let dst_offset = dst_float.advanced(by: dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width)
                            for i in 0 ..< output_width {
                                dst_offset[i] = 0;
                            }
                            #endif
                        } else {
                            let ptr = dst.advanced(by: dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width)
                            let bufferPtr = UnsafeMutableBufferPointer<Self>(start: ptr, count: output_width)
                            Self.fill(value: 0, result: bufferPtr, count: output_width)
                        }
                    }
                }
            }
        }
    }
    
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func col2img(values: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) {
        let src = values.baseAddress!
        let dst = result.baseAddress!
        
        let setup = D4Img2ColSetup(
            batch_size: batchSize,
            channels: channels,
            height: height,
            width: width,
            kernel_height: kernelHeight,
            kernel_width: kernelWidth,
            padding: padding,
            stride: stride
        )
        
        let depth_stride = setup.width * setup.height
        let featuremap_stride = depth_stride * setup.channels
        
        let input_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1
        let input_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1
        let src_batch_stride = input_width * input_height
        let src_full_stride = src_batch_stride * setup.batch_size
        
        if Self.self == Float.self {
            #if MKL_ENABLE
            ippsSet_32f(0, (dst as! UnsafeMutablePointer<Float>), Int32(setup.width * setup.height * setup.channels * setup.batch_size))
            #elseif canImport(Accelerate)
            vDSP_vfill([0], (dst as! UnsafeMutablePointer<Float>), 1, UInt(setup.width * setup.height * setup.channels * setup.batch_size))
            #else
            for i in 0 ..< setup.width * setup.height * setup.channels * setup.batch_size {
                dst[i] = 0;
            }
            #endif
        } else {
            Self.fill(value: 0, result: UnsafeMutableBufferPointer<Self>(start: dst, count: setup.width * setup.height * setup.channels * setup.batch_size), count: setup.width * setup.height * setup.channels * setup.batch_size)
        }
        
        for k in 0 ..< setup.kernel_width * setup.kernel_height * setup.channels {
            let kx = k % setup.kernel_width;
            let kyz = k / setup.kernel_width;
            let ky = kyz % setup.kernel_height;
            let kz = kyz / setup.kernel_height;
            
            for b in 0 ..< setup.batch_size {
                
                for y in 0 ..< input_height {
                    let in_y = y &* setup.stride &- setup.padding &+ ky;
                    
                    for x in 0 ..< input_width {
                        let in_x = x &* setup.stride &- setup.padding &+ kx;
                        
                        if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                            let input = src[src_full_stride &* k &+ b &* src_batch_stride &+ y &* input_width &+ x];
                            dst[in_x &+ in_y &* setup.width &+ kz &* depth_stride &+ b &* featuremap_stride] += input;
                        }
                    }
                }
            }
        }
    }
    
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    internal static func gemm_generic(_ transA: Bool, _ transB: Bool, _ __M: Int, _ __N: Int, _ __K: Int, _ alpha: Self, _ __A: UnsafePointer<Self>, _ lda: Int, _ __B: UnsafePointer<Self>, _ ldb: Int, _ beta: Self, _ __C: UnsafeMutablePointer<Self>, _ ldc: Int) {
        if (__M == 0 || __N == 0 || ((alpha == 0 || __K == 0) && beta == 1)) {
            return
        }
        
        if (alpha == 0) {
            if (beta == 0) {
                for i in 0 ..< __M * __N {
                    __C[i] = 0
                }
            } else {
                for i in 0 ..< __M * __N {
                    __C[i] *= beta
                }
            }
        }
        
        if (beta == 0) {
            for i in 0 ..< __M * __N {
                __C[i] = 0
            }
        } else {
            for i in 0 ..< __M * __N {
                __C[i] *= beta
            }
        }
        
        if (transA) {
            if (transB) {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[r &+ l &* __M] * __B[l &+ c &* __K]
                        }
                        __C[r &* __N &+ c] = alpha * tmp
                    }
                }
            } else {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[r &+ l &* __M] * __B[l &* __N &+ c]
                        }
                        __C[r &* __N &+ c] = alpha * tmp
                    }
                }
            }
        } else {
            if (transB) {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[l &+ r &* __K] * __B[l &+ c &* __K]
                        }
                        __C[r &* __N &+ c] = alpha * tmp
                    }
                }
            } else {
                for r in 0 ..< __M {
                    for c in 0 ..< __N {
                        var tmp: Self = 0
                        for l in 0 ..< __K {
                            tmp += __A[l &+ r &* __K] * __B[l &* __N &+ c]
                        }
                        __C[r &* __N &+ c] = alpha * tmp
                    }
                }
            }
        }
    }
    
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func scatter(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, dst_shape: [Int], axis: Int, ignoreIndex: Int32) {
        let src = values.baseAddress!
        let target = result.baseAddress!
        let context = context.baseAddress!
        
        let dst_dim = dst_shape.count
        
        let src_strides = malloc(MemoryLayout<Int>.stride * dst_dim - 1).assumingMemoryBound(to: Int.self)
        let src_shape = malloc(MemoryLayout<Int>.stride * dst_dim - 1).assumingMemoryBound(to: Int.self)
        let dst_strides = malloc(MemoryLayout<Int>.stride * dst_dim).assumingMemoryBound(to: Int.self)
        
        defer {
            free(src_strides)
            free(src_shape)
            free(dst_strides)
        }
        
        dst_strides[dst_dim - 1] = 1
        src_strides[dst_dim - 2] = 1
        
        for i in (0 ... (dst_dim - 2)).reversed() {
            dst_strides[i] = dst_shape[i &+ 1] &* dst_strides[i &+ 1]
        }
        for i in (0 ... (dst_dim - 2)).reversed() {
            src_shape[i] = dst_shape[i >= axis ? i &+ 1 : i]
            if (i < dst_dim - 2) {
                src_strides[i] = src_shape[i &+ 1] &* src_strides[i &+ 1]
            } else {
                src_strides[i] = 1
            }
        }
        let count = src_shape[0] * src_strides[0]
        
        let dst_count = dst_strides[0] * dst_shape[0]
        if Self.self == Float.self {
            #if MKL_ENABLE
            ippsSet_32f(0, (target as! UnsafeMutablePointer<Float>), Int32(dst_count))
            #elseif canImport(Accelerate)
            vDSP_vfill([0], (target as! UnsafeMutablePointer<Float>), 1, UInt(dst_count))
            #else
            for i in 0 ..< count {
                target[i] = 0
            }
            #endif
        } else {
            for i in 0 ..< count {
                target[i] = 0
            }
        }
        
        for i in 0 ..< count {
            let src_idx = i
            let c = context[i]
            if c == ignoreIndex {
                continue
            }
            var dst_idx = Int(c) &* dst_strides[axis]
            
            for a in 0 ..< dst_dim - 1 {
                let src_dim_idx = (i / src_strides[a]) % src_shape[a]
                dst_idx = dst_idx &+ src_dim_idx &* dst_strides[a >= axis ? a &+ 1 : a]
            }
            target[dst_idx] = src[src_idx]
        }
    }
    
    @_specialize(where Self == Float)
    @_specialize(where Self == Int32)
    @_specialize(where Self == Double)
    static func gather(values: UnsafeBufferPointer<Self>, context: UnsafeBufferPointer<Int32>, result: UnsafeMutableBufferPointer<Self>, src_shape: [Int], axis: Int, ignoreIndex: Int32) {
        let src_dim = src_shape.count
        
        let src = values.baseAddress!
        let target = result.baseAddress!
        let context = context.baseAddress!
        
        let dst_strides = malloc(MemoryLayout<Int>.stride &* src_dim &- 1).assumingMemoryBound(to: Int.self)
        let dst_shape = malloc(MemoryLayout<Int>.stride &* src_dim &- 1).assumingMemoryBound(to: Int.self)
        let src_strides = malloc(MemoryLayout<Int>.stride &* src_dim).assumingMemoryBound(to: Int.self)
        
        defer {
            free(dst_strides)
            free(dst_shape)
            free(src_strides)
        }
        
        src_strides[src_dim - 1] = 1
        dst_strides[src_dim - 2] = 1
        
        for i  in (0 ... (src_dim - 2)).reversed() {
            src_strides[i] = src_shape[i &+ 1] * src_strides[i &+ 1]
        }
        for i in (0 ... (src_dim - 2)).reversed() {
            dst_shape[i] = src_shape[i >= axis ? i &+ 1 : i]
            if (i < src_dim &- 2) {
                dst_strides[i] = dst_shape[i &+ 1] &* dst_strides[i &+ 1]
            } else {
                dst_strides[i] = 1
            }
        }
        
        let count = dst_shape[0] &* dst_strides[0]
        
        for i in 0 ..< count {
            let dst_idx = i
            let c = context[i]
            if c == ignoreIndex {
                target[dst_idx] = 0
                continue
            }
            
            var src_idx = Int(c) &* src_strides[axis]
            
            for a in 0 ..< src_dim - 1 {
                let dst_dim_idx = (i / dst_strides[a]) % dst_shape[a]
                src_idx = src_idx &+ dst_dim_idx &* src_strides[a >= axis ? a &+ 1 : a]
            }
            target[dst_idx] = src[src_idx]
        }
    }
    
    @_specialize(where Self == Int32)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    static func max(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhsPtr = lhs.baseAddress!
        let rhsPtr = rhs.baseAddress!
        let resultPtr = result.baseAddress!
        let contextPtr = context.baseAddress!
        
        var i = 0
        while i < count {
            let l = lhsPtr[i]
            let r = rhsPtr[i]
            if l >= r {
                resultPtr[i] = l
                contextPtr[i] = 0
            } else {
                resultPtr[i] = r
                contextPtr[i] = 1
            }
            
            i &+= 1
        }
    }
    
    @_specialize(where Self == Int32)
    @_specialize(where Self == Float)
    @_specialize(where Self == Double)
    static func min(lhs: UnsafeBufferPointer<Self>, rhs: UnsafeBufferPointer<Self>, result: UnsafeMutableBufferPointer<Self>, context: UnsafeMutableBufferPointer<Self>, count: Int) {
        let lhsPtr = lhs.baseAddress!
        let rhsPtr = rhs.baseAddress!
        let resultPtr = result.baseAddress!
        let contextPtr = context.baseAddress!
        
        var i = 0
        while i < count {
            let l = lhsPtr[i]
            let r = rhsPtr[i]
            if l <= r {
                resultPtr[i] = l
                contextPtr[i] = 0
            } else {
                resultPtr[i] = r
                contextPtr[i] = 1
            }
            
            i &+= 1
        }
    }
}

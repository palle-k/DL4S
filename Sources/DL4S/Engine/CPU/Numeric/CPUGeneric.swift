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
import DL4SLib

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
                            d4lib_sfill([0], &dst_float[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width], 1, d4lib_length(output_width))
                        } else {
                            Self.fill(value: 0, result: UnsafeMutableBufferPointer<Self>(start: &dst[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width], count: output_width), count: output_width)
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
            d4lib_sfill([0], (dst as! UnsafeMutablePointer<Float>), 1, d4lib_length(setup.width * setup.height * setup.channels * setup.batch_size))
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
}

//
//  dl4slib.c
//  DL4SLib
//
//  Created by Palle Klewitz on 20.10.19.
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

#include "d4lib.h"
#include <immintrin.h>
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <stdbool.h>

void avxcpy(void* __restrict dst, const void* __restrict src, size_t count) {
    memcpy(dst, src, count);
    return;
#ifdef __AVX2__
#warning "AVX2 enabled"
    const __m256i *pSrc = src;
    __m256i *pDest = dst;
    size_t nVects = (count + sizeof(*pSrc) - 1) / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        const __m256i loaded = _mm256_stream_load_si256(pSrc);
        _mm256_stream_si256(pDest, loaded);
    }
    _mm_sfence();
#elif defined __AVX__
#warning "AVX enabled"
    const __m128 *pSrc = src;
    __m128 *pDest = dst;
    size_t nVects = (count + sizeof(*pSrc) - 1) / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
      __m128 buffer = _mm_load_ps(pSrc);
      _mm_store_ps(pDest, buffer);
    }
    _mm_sfence();
#else
#warning "NO AVX enabled"
    const long long *pSrc = src;
    long long *pDest = dst;
    size_t nVects = (count + sizeof(*pSrc) - 1) / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        *pDest = *pSrc;
    }
#endif
}


#if !defined(__APPLE__) && !defined(__INTEL_MKL__)

#endif

void d4lib_igemm(D4LIB_ORDER order, D4LIB_TRANSPOSE __transA, D4LIB_TRANSPOSE __transB, int __M, int __N, int __K, int alpha, const int* __A, int lda, const int* __B, int ldb, int beta, int* __C, int ldc) {
    if (order == D4LIB_ColMajor) {
        // fatalError("CblasColMajor is unsupported. This parameter only exists for compatibility purposes")
        fprintf(stderr, "ColMajor layout is unsupported for d4lib_igemm.\n");
        raise(SIGINT);
    }
    
    bool transA = (__transA == D4LIB_Trans);
    bool transB = (__transB == D4LIB_Trans);
    
    if (__M == 0 || __N == 0 || ((alpha == 0 || __K == 0) && beta == 1)) {
        return;
    }
    
    if (alpha == 0) {
        if (beta == 0) {
            for (int i = 0; i < __M * __N; i++) {
                __C[i] = 0;
            }
        } else {
            for (int i = 0; i < __M * __N; i++) {
                __C[i] *= beta;
            }
        }
    }
    
    if (beta == 0) {
        for (int i = 0; i < __M * __N; i++) {
            __C[i] = 0;
        }
    } else {
        for (int i = 0; i < __M * __N; i++) {
            __C[i] *= beta;
        }
    }
    
    if (transA) {
        if (transB) {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    int tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[r + l * __M] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    int tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[r + l * __M] * __B[l * __N + c];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        }
    } else {
        if (transB) {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    int tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    int tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l * __N + c];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        }
    }
}

void d4lib_simg2col(const float* __restrict src, float* __restrict dst, const D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int dst_batch_stride = output_width * output_height;
    const int dst_full_stride = dst_batch_stride * setup.batch_size;
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * output_width * output_height;
            
            for (int y = 0; y < output_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                if (in_y >= 0 && in_y < setup.height) {
                    for (int x = 0; x < output_width; x++) {
                        int in_x = x * setup.stride - setup.padding + kx;
                        
                        float in;
                        if (in_x >= 0 && in_x < setup.width) {
                            in = src[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride];
                        } else {
                            in = 0;
                        }
                        dst[dst_full_stride * k + b * dst_batch_stride + y * output_width + x] = in;
                    }
                } else {
                    const float zero = 0;
                    d4lib_sfill(&zero, &dst[dst_full_stride * k + b * dst_batch_stride + y * output_width], 1, output_width);
                }
            }
        }
    }
}

void d4lib_scol2img(const float* __restrict src, float* __restrict dst, D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int input_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int input_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int src_batch_stride = input_width * input_height;
    const int src_full_stride = src_batch_stride * setup.batch_size;
    
    const float zero = 0;
    d4lib_sfill(&zero, dst, 1, setup.width * setup.height * setup.channels * setup.batch_size);
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * input_width * input_height;
            
            for (int y = 0; y < input_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                for (int x = 0; x < input_width; x++) {
                    int in_x = x * setup.stride - setup.padding + kx;
                    
                    if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                        float in = src[src_full_stride * k + b * src_batch_stride + y * input_width + x];
                        dst[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride] += in;
                    }
                }
            }
        }
    }
}

void d4lib_dimg2col(const double* __restrict src, double* __restrict dst, D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int dst_batch_stride = output_width * output_height;
    const int dst_full_stride = dst_batch_stride * setup.batch_size;
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * output_width * output_height;
            
            for (int y = 0; y < output_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                for (int x = 0; x < output_width; x++) {
                    int in_x = x * setup.stride - setup.padding + kx;
                    
                    double in;
                    if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                        in = src[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride];
                    } else {
                        in = 0;
                    }
                    dst[dst_full_stride * k + b * dst_batch_stride + y * output_width + x] = in;
                }
            }
        }
    }
}

void d4lib_dcol2img(const double* __restrict src, double* __restrict dst, D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int input_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int input_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int src_batch_stride = input_width * input_height;
    const int src_full_stride = src_batch_stride * setup.batch_size;
    
    const double zero = 0;
    d4lib_dfill(&zero, dst, 1, setup.width * setup.height * setup.channels * setup.batch_size);
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * input_width * input_height;
            
            for (int y = 0; y < input_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                for (int x = 0; x < input_width; x++) {
                    int in_x = x * setup.stride - setup.padding + kx;
                    
                    if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                        double in = src[src_full_stride * k + b * src_batch_stride + y * input_width + x];
                        dst[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride] += in;
                    }
                }
            }
        }
    }
}


void d4lib_iimg2col(const int* __restrict src, int* __restrict dst, D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int dst_batch_stride = output_width * output_height;
    const int dst_full_stride = dst_batch_stride * setup.batch_size;
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * output_width * output_height;
            
            for (int y = 0; y < output_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                for (int x = 0; x < output_width; x++) {
                    int in_x = x * setup.stride - setup.padding + kx;
                    
                    int in;
                    if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                        in = src[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride];
                    } else {
                        in = 0;
                    }
                    dst[dst_full_stride * k + b * dst_batch_stride + y * output_width + x] = in;
                }
            }
        }
    }
}

void d4lib_icol2img(const int* __restrict src, int* __restrict dst, D4LIB_Img2ColSetup setup) {
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    
    const int input_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int input_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int src_batch_stride = input_width * input_height;
    const int src_full_stride = src_batch_stride * setup.batch_size;
    
    const int zero = 0;
    d4lib_ifill(&zero, dst, 1, setup.width * setup.height * setup.channels * setup.batch_size);
    
    for (int k = 0; k < setup.kernel_width * setup.kernel_height * setup.channels; k++) {
        int kx = k % setup.kernel_width;
        int kyz = k / setup.kernel_width;
        int ky = kyz % setup.kernel_height;
        int kz = kyz / setup.kernel_height;
        int ko = ky * setup.width;
        
        for (int b = 0; b < setup.batch_size; b++) {
            int bs = b * input_width * input_height;
            
            for (int y = 0; y < input_height; y++) {
                int in_y = y * setup.stride - setup.padding + ky;
                
                for (int x = 0; x < input_width; x++) {
                    int in_x = x * setup.stride - setup.padding + kx;
                    
                    if (in_x >= 0 && in_x < setup.width && in_y >= 0 && in_y < setup.height) {
                        int in = src[src_full_stride * k + b * src_batch_stride + y * input_width + x];
                        dst[in_x + in_y * setup.width + kz * depth_stride + b * featuremap_stride] += in;
                    }
                }
            }
        }
    }
}


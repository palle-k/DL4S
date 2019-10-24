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
#include <string.h>
#include <signal.h>
#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

void avxcpy(void* __restrict dst, const void* __restrict src, size_t count) {
#ifdef __AVX2__
#pragma message "AVX2 support enabled"
    const __m256i *pSrc = src;
    __m256i *pDest = dst;
    size_t nVects = count / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        const __m256i loaded = _mm256_stream_load_si256(pSrc);
        _mm256_stream_si256(pDest, loaded);
    }
    _mm_sfence();
    
    for (int i = count & ~(sizeof(__m256i) - 1); i < count; i++) {
        ((char*) dst)[i] = ((const char*) src)[i];
    }
#elif defined __AVX__
#pragma message "AVX support enabled"
    const __m128 *pSrc = src;
    __m128i *pDest = dst;
    size_t nVects = count / sizeof(__m128i);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        __m128i buffer = _mm_load_ps((float*) pSrc);
        _mm_store_ps((float*) pDest, buffer);
    }
    _mm_sfence();
    for (size_t i = count & ~(sizeof(__m128i) - 1); i < count; i++) {
        ((char*) dst)[i] = ((const char*) src)[i];
    }
#else
#warning NO AVX enabled. Compile with -Xcc -mavx or -Xcc -mavx2
    const long long *pSrc = src;
    long long *pDest = dst;
    size_t nVects = (count + sizeof(*pSrc) - 1) / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        *pDest = *pSrc;
    }
    for (size_t i = count & ~(sizeof(long long) - 1); i < count; i++) {
        ((char*) dst)[i] = ((const char*) src)[i];
    }
#endif
}


#if !defined(__APPLE__) && !defined(MKL_ENABLE)
#warning Compiling DL4S without any accelerator library.
#define MAX(x, y) (x >= y ? x : y)

// Vector Fill
void d4lib_sfill(const float* src, float* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = *src;
    }
}
void d4lib_dfill(const double* src, double* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = *src;
    }
}
void d4lib_ifill(const int* src, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = *src;
    }
}

// Vector square
void d4lib_ssquare(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] * src[i];
    }
}
void d4lib_dsquare(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] * src[i];
    }
}
void d4lib_isquare(const int* src, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] * src[i];
    }
}

// Vector threshold
void d4lib_sthreshold(const float* src, const float* thresh, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(*thresh, src[i]);
    }
}
void d4lib_dthreshold(const double* src, const double* thresh, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(*thresh, src[i]);
    }
}
void d4lib_ithreshold(const int* src, const int* thresh, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(*thresh, src[i]);
    }
}

// Vector negate
void d4lib_sneg(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = -src[i];
    }
}
void d4lib_dneg(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = -src[i];
    }
}
void d4lib_ineg(const int* src, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = -src[i];
    }
}

// Vector add
void d4lib_saddv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + rhs[i];
    }
}
void d4lib_daddv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + rhs[i];
    }
}
void d4lib_iaddv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + rhs[i];
    }
}

// Vector scalar add
void d4lib_saddvs(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + *rhs;
    }
}
void d4lib_daddvs(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + *rhs;
    }
}
void d4lib_iaddvs(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + *rhs;
    }
}

// Vector subtract
void d4lib_ssubv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] - rhs[i];
    }
}
void d4lib_dsubv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] - rhs[i];
    }
}
void d4lib_isubv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] - rhs[i];
    }
}

// Scalar vector subtract
void d4lib_ssubsv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs - rhs[i];
    }
}
void d4lib_dsubsv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs - rhs[i];
    }
}
void d4lib_isubsv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs - rhs[i];
    }
}

// Vector multiply
void d4lib_smulv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * rhs[i];
    }
}
void d4lib_dmulv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * rhs[i];
    }
}
void d4lib_imulv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * rhs[i];
    }
}

// Vector scalar multiply
void d4lib_smulvs(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * *rhs;
    }
}
void d4lib_dmulvs(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * *rhs;
    }
}
void d4lib_imulvs(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * *rhs;
    }
}

// Vector divide
void d4lib_sdivv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] / rhs[i];
    }
}
void d4lib_ddivv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] / rhs[i];
    }
}
void d4lib_idivv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] / rhs[i];
    }
}

// Scalar vector divide
void d4lib_sdivsv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs / rhs[i];
    }
}
void d4lib_ddivsv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs / rhs[i];
    }
}
void d4lib_idivsv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *lhs / rhs[i];
    }
}

// Vector Sum
void d4lib_ssum(const float* src, d4lib_stride src_stride, float* dst, d4lib_length length) {
    float sum = 0;
    for (int i = 0; i < length; i++) {
        sum += src[i];
    }
    *dst = sum;
}
void d4lib_dsum(const double* src, d4lib_stride src_stride, double* dst, d4lib_length length) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += src[i];
    }
    *dst = sum;
}
void d4lib_isum(const int* src, d4lib_stride src_stride, int* dst, d4lib_length length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += src[i];
    }
    *dst = sum;
}

// Vector maximum value and index
void d4lib_smaxi(const float* src, d4lib_stride src_stride, float* dst, d4lib_length* dst_idx, d4lib_length length) {
    int max_i = -1;
    float max_v = -INFINITY;
    for (int i = 0; i < length; i++) {
        if (src[i] > max_v) {
            max_v = src[i * src_stride];
            max_i = i;
        }
    }
    *dst_idx = (d4lib_length) max_i;
    *dst = max_v;
}
void d4lib_dmaxi(const double* src, d4lib_stride src_stride, double* dst, d4lib_length* dst_idx, d4lib_length length) {
    int max_i = -1;
    double max_v = -INFINITY;
    for (int i = 0; i < length; i++) {
        if (src[i] > max_v) {
            max_v = src[i * src_stride];
            max_i = i;
        }
    }
    *dst_idx = (d4lib_length) max_i;
    *dst = max_v;
}
void d4lib_imaxi(const int* src, d4lib_stride src_stride, int* dst, d4lib_length* dst_idx, d4lib_length length) {
    int max_i = -1;
    int max_v = INT_MIN;
    for (int i = 0; i < length; i++) {
        if (src[i] > max_v) {
            max_v = src[i * src_stride];
            max_i = i;
        }
    }
    *dst_idx = (d4lib_length) max_i;
    *dst = max_v;
}

// Vector vector max
void d4lib_smax(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(lhs[i], rhs[i]);
    }
}
void d4lib_dmax(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(lhs[i], rhs[i]);
    }
}
void d4lib_imax(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(lhs[i], rhs[i]);
    }
}

// Vector maximum value and index
void d4lib_smini(const float* src, d4lib_stride src_stride, float* dst, d4lib_length* dst_idx, d4lib_length length) {
    int min_i = -1;
    float min_v = INFINITY;
    for (int i = 0; i < length; i++) {
        if (src[i] < min_v) {
            min_v = src[i * src_stride];
            min_i = i;
        }
    }
    *dst_idx = (d4lib_length) min_i;
    *dst = min_v;
}
void d4lib_dmini(const double* src, d4lib_stride src_stride, double* dst, d4lib_length* dst_idx, d4lib_length length) {
    int min_i = -1;
    double min_v = INFINITY;
    for (int i = 0; i < length; i++) {
        if (src[i] < min_v) {
            min_v = src[i * src_stride];
            min_i = i;
        }
    }
    *dst_idx = (d4lib_length) min_i;
    *dst = min_v;
}
void d4lib_imini(const int* src, d4lib_stride src_stride, int* dst, d4lib_length* dst_idx, d4lib_length length) {
    int min_i = -1;
    int min_v = INT_MAX;
    for (int i = 0; i < length; i++) {
        if (src[i] < min_v) {
            min_v = src[i * src_stride];
            min_i = i;
        }
    }
    *dst_idx = (d4lib_length) min_i;
    *dst = min_v;
}

// Vector ramp
void d4lib_sramp(const float* start, const float* increment, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *start + i * *increment;
    }
}
void d4lib_dramp(const double* start, const double* increment, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *start + i * *increment;
    }
}
void d4lib_iramp(const int* start, const int* increment, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = *start + i * *increment;
    }
}

// single vector math functions
void d4lib_stanh(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = tanhf(src[i]);
    }
}
void d4lib_dtanh(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = tanh(src[i]);
    }
}
void d4lib_sexp(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = expf(src[i]);
    }
}
void d4lib_dexp(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = exp(src[i]);
    }
}
void d4lib_slog(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = logf(src[i]);
    }
}
void d4lib_dlog(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = log(src[i]);
    }
}
void d4lib_ssqrt(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = sqrtf(src[i]);
    }
}
void d4lib_dsqrt(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = sqrt(src[i]);
    }
}
void d4lib_ssin(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = sinf(src[i]);
    }
}
void d4lib_dsin(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = sin(src[i]);
    }
}
void d4lib_scos(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = cosf(src[i]);
    }
}
void d4lib_dcos(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = cos(src[i]);
    }
}
void d4lib_stan(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = tanf(src[i]);
    }
}
void d4lib_dtan(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = tan(src[i]);
    }
}
void d4lib_scopysign(const float* mag, const float* sig, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = copysignf(mag[i], sig[i]);
    }
}
void d4lib_dcopysign(const double* mag, const double* sig, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = copysignf(mag[i], sig[i]);
    }
}
void d4lib_sheaviside(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] > 0 ? 1 : 0;
    }
}
void d4lib_dheaviside(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] > 0 ? 1 : 0;
    }
}

void d4lib_scopy_strided(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}
void d4lib_dcopy_strided(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}
void d4lib_icopy_strided(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}

// MARK: Matrix Functions
// Comparable to BLAS Level 3

void d4lib_stranspose(const float* src, float* dst, d4lib_length src_cols, d4lib_length src_rows)  {
    for (int x = 0; x < src_cols; x++) {
        for (int y = 0; y < src_rows; y++) {
            dst[y + x * src_rows] = src[y * src_cols + x];
        }
    }
}

void d4lib_dtranspose(const double* src, double* dst, d4lib_length src_cols, d4lib_length src_rows) {
    for (int x = 0; x < src_cols; x++) {
        for (int y = 0; y < src_rows; y++) {
            dst[y + x * src_rows] = src[y * src_cols + x];
        }
    }
}

void d4lib_itranspose(const int* src, int* dst, d4lib_length src_cols, d4lib_length src_rows) {
    for (int x = 0; x < src_cols; x++) {
        for (int y = 0; y < src_rows; y++) {
            dst[y + x * src_rows] = src[y * src_cols + x];
        }
    }
}

void d4lib_sgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE __transA, D4LIB_TRANSPOSE __transB, int __M, int __N, int __K, float alpha, const float* __A, int lda, const float* __B, int ldb, float beta, float* __C, int ldc) {
    if (order == D4LIB_ColMajor) {
        fprintf(stderr, "ColMajor layout is unsupported for d4lib_sgemm.\n");
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
                    float tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[r + l * __M] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    float tmp = 0;
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
                    float tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    float tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l * __N + c];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        }
    }
}

void d4lib_dgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE __transA, D4LIB_TRANSPOSE __transB, int __M, int __N, int __K, double alpha, const double* __A, int lda, const double* __B, int ldb, double beta, double* __C, int ldc) {
    if (order == D4LIB_ColMajor) {
        fprintf(stderr, "ColMajor layout is unsupported for d4lib_dgemm.\n");
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
                    double tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[r + l * __M] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    double tmp = 0;
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
                    double tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l + c * __K];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        } else {
            for (int r = 0; r < __M; r++) {
                for (int c = 0; c < __N; c++) {
                    double tmp = 0;
                    for (int l = 0; l < __K; l++) {
                        tmp += __A[l + r * __K] * __B[l * __N + c];
                    }
                    __C[r * __N + c] = alpha * tmp;
                }
            }
        }
    }
}

#endif

void d4lib_igemm(D4LIB_ORDER order, D4LIB_TRANSPOSE __transA, D4LIB_TRANSPOSE __transB, int __M, int __N, int __K, int alpha, const int* __A, int lda, const int* __B, int ldb, int beta, int* __C, int ldc) {
    if (order == D4LIB_ColMajor) {
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


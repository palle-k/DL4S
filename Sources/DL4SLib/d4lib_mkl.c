//
//  d4lib_mkl.c
//  DL4SLib
//
//  Created by Palle Klewitz on 22.10.19.
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

#if defined(MKL_ENABLE)
#pragma message "Using MKL"
#include "mkl.h"
// #include "ipp.h"
#include <math.h>
#include <limits.h>
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
    vsSqr(length, src, dst);
}
void d4lib_dsquare(const double* src, double* dst, d4lib_length length) {
    vdSqr(length, src, dst);
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
    avxcpy(dst, src, sizeof(float) * length);
    cblas_sscal(length, -1, dst, 1);
}
void d4lib_dneg(const double* src, double* dst, d4lib_length length) {
    avxcpy(dst, src, sizeof(double) * length);
    cblas_dscal(length, -1, dst, 1);
}
void d4lib_ineg(const int* src, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = -src[i];
    }
}

// Vector add
void d4lib_saddv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vsAdd(length, lhs, rhs, dst);
}
void d4lib_daddv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vdAdd(length, lhs, rhs, dst);
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
    vsSub(length, lhs, rhs, dst);
}
void d4lib_dsubv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vdSub(length, lhs, rhs, dst);
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
    vsMul(length, lhs, rhs, dst);
}
void d4lib_dmulv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vdMul(length, lhs, rhs, dst);
}
void d4lib_imulv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * rhs[i];
    }
}

// Vector scalar multiply
void d4lib_smulvs(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    avxcpy(dst, lhs, sizeof(float) * length);
    cblas_sscal(length, *rhs, dst, 1);
}
void d4lib_dmulvs(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    avxcpy(dst, lhs, sizeof(double) * length);
    cblas_dscal(length, *rhs, dst, 1);
}
void d4lib_imulvs(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * *rhs;
    }
}

// Vector divide
void d4lib_sdivv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vsDiv(length, lhs, rhs, dst);
}
void d4lib_ddivv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vdDiv(length, lhs, rhs, dst);
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
    vsTanh(length, src, dst);
}
void d4lib_dtanh(const double* src, double* dst, d4lib_length length) {
    vdTanh(length, src, dst);
}
void d4lib_sexp(const float* src, float* dst, d4lib_length length) {
    vsExp(length, src, dst);
}
void d4lib_dexp(const double* src, double* dst, d4lib_length length) {
    vdExp(length, src, dst);
}
void d4lib_slog(const float* src, float* dst, d4lib_length length) {
    vsLn(length, src, dst);
}
void d4lib_dlog(const double* src, double* dst, d4lib_length length) {
    vdLn(length, src, dst);
}
void d4lib_ssqrt(const float* src, float* dst, d4lib_length length) {
    vsSqrt(length, src, dst);
}
void d4lib_dsqrt(const double* src, double* dst, d4lib_length length) {
    vdSqrt(length, src, dst);
}
void d4lib_ssin(const float* src, float* dst, d4lib_length length) {
    vsSin(length, src, dst);
}
void d4lib_dsin(const double* src, double* dst, d4lib_length length) {
    vdSin(length, src, dst);
}
void d4lib_scos(const float* src, float* dst, d4lib_length length) {
    vsCos(length, src, dst);
}
void d4lib_dcos(const double* src, double* dst, d4lib_length length) {
    vdCos(length, src, dst);
}
void d4lib_stan(const float* src, float* dst, d4lib_length length) {
    vsTan(length, src, dst);
}
void d4lib_dtan(const double* src, double* dst, d4lib_length length) {
    vdTan(length, src, dst);
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
    cblas_scopy(length, src, src_stride, dst, dst_stride);
}
void d4lib_dcopy_strided(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length) {
    cblas_dcopy(length, src, src_stride, dst, dst_stride);
}
void d4lib_icopy_strided(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = src[i * src_stride];
    }
}

// MARK: Matrix Functions
// Comparable to BLAS Level 3

void d4lib_stranspose(const float* src, float* dst, d4lib_length src_cols, d4lib_length src_rows) {
    mkl_somatcopy('R', 'T', src_cols, src_rows, 1, src, src_cols, dst, src_rows);
}
void d4lib_dtranspose(const double* src, double* dst, d4lib_length src_cols, d4lib_length src_rows) {
    mkl_domatcopy('R', 'T', src_cols, src_rows, 1, src, src_cols, dst, src_rows);
}
void d4lib_itranspose(const int* src, int* dst, d4lib_length src_cols, d4lib_length src_rows) {
    for (int x = 0; x < src_cols; x++) {
        for (int y = 0; y < src_rows; y++) {
            dst[y + x * src_rows] = src[y * src_cols + x];
        }
    }
}

void d4lib_sgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
    cblas_sgemm(
        CblasRowMajor,
        transA == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        transB == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb, 
        beta,
        c, ldc
    );
}
void d4lib_dgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    cblas_dgemm(
        CblasRowMajor,
        transA == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        transB == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb, 
        beta,
        c, ldc
    );
}

#endif

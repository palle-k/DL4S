//
//  d4lib_accelerate.c
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

#include <stdio.h>

#ifdef __APPLE__
#warning Using Accelerate Framework
#include "d4lib.h"
#include <Accelerate/Accelerate.h>

void d4lib_sfill(const float* src, float* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vfill(src, dst, dst_stride, length); }
void d4lib_dfill(const double* src, double* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vfillD(src, dst, dst_stride, length); }
void d4lib_ifill(const int* src, int* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vfilli(src, dst, dst_stride, length); }

// Vector square
void d4lib_ssquare(const float* src, float* dst, d4lib_length length) { vDSP_vsq(src, 1, dst, 1, length); }
void d4lib_dsquare(const double* src, double* dst, d4lib_length length)  { vDSP_vsqD(src, 1, dst, 1, length); }
void d4lib_isquare(const int* src, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = src[i] * src[i];
    }
}

// Vector threshold
void d4lib_sthreshold(const float* src, const float* thresh, float* dst, d4lib_length length) {
    vDSP_vthres(src, 1, thresh, dst, 1, length);
}
void d4lib_dthreshold(const double* src, const double* thresh, double* dst, d4lib_length length) {
    vDSP_vthresD(src, 1, thresh, dst, 1, length);
}
void d4lib_ithreshold(const int* src, const int* thresh, int* dst, d4lib_length length) {
    float t = *thresh;
    for (int i = 0; i < length; i++) {
        int s = src[i];
        dst[i] = s > t ? s : t;
    }
}

// Vector negate
void d4lib_sneg(const float* src, float* dst, d4lib_length length) { vDSP_vneg(src, 1, dst, 1, length); }
void d4lib_dneg(const double* src, double* dst, d4lib_length length)  { vDSP_vnegD(src, 1, dst, 1, length); }
void d4lib_ineg(const int* src, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = -src[i];
    }
}

// Vector add
void d4lib_saddv(const float* lhs, const float* rhs, float* dst, d4lib_length length) { vDSP_vadd(lhs, 1, rhs, 1, dst, 1, length); }
void d4lib_daddv(const double* lhs, const double* rhs, double* dst, d4lib_length length) { vDSP_vaddD(lhs, 1, rhs, 1, dst, 1, length); }
void d4lib_iaddv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] + rhs[i];
    }
}

// Vector scalar add
void d4lib_saddvs(const float* lhs, const float* rhs , float* dst, d4lib_length length) {
    vDSP_vsadd(lhs, 1, rhs, dst, 1, length);
}
void d4lib_daddvs(const double* lhs, const double* rhs , double* dst, d4lib_length length){
    vDSP_vsaddD(lhs, 1, rhs, dst, 1, length);
}
void d4lib_iaddvs(const int* lhs, const int* rhs , int* dst, d4lib_length length) {
    vDSP_vsaddi(lhs, 1, rhs, dst, 1, length);
}

// Vector subtract
void d4lib_ssubv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vDSP_vsub(rhs, 1, lhs, 1, dst, 1, length);
}
void d4lib_dsubv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vDSP_vsubD(rhs, 1, lhs, 1, dst, 1, length);
}
void d4lib_isubv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] - rhs[i];
    }
}

// Scalar vector subtract
void d4lib_ssubsv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    float l = *lhs;
    for (int i = 0; i < length; i++) {
        dst[i] = l - rhs[i];
    }
}
void d4lib_dsubsv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    double l = *lhs;
    for (int i = 0; i < length; i++) {
        dst[i] = l - rhs[i];
    }
}
void d4lib_isubsv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    int l = *lhs;
    for (int i = 0; i < length; i++) {
        dst[i] = l - rhs[i];
    }
}

// Vector multiply
void d4lib_smulv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vDSP_vmul(lhs, 1, rhs, 1, dst, 1, length);
}
void d4lib_dmulv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vDSP_vmulD(lhs, 1, rhs, 1, dst, 1, length);
}
void d4lib_imulv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * rhs[i];
    }
}

// Vector scalar multiply
void d4lib_smulvs(const float* lhs, const float* rhs , float* dst, d4lib_length length) {
    vDSP_vsmul(lhs, 1, rhs, dst, 1, length);
}
void d4lib_dmulvs(const double* lhs, const double* rhs , double* dst, d4lib_length length) {
    vDSP_vsmulD(lhs, 1, rhs, dst, 1, length);
}
void d4lib_imulvs(const int* lhs, const int* rhs , int* dst, d4lib_length length) {
    int r = *rhs;
    for (int i = 0; i < length; i++) {
        dst[i] = lhs[i] * r;
    }
}

// Vector divide
void d4lib_sdivv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vDSP_vdiv(rhs, 1, lhs, 1, dst, 1, length);
}
void d4lib_ddivv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vDSP_vdivD(rhs, 1, lhs, 1, dst, 1, length);
}
void d4lib_idivv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    vDSP_vdivi(rhs, 1, lhs, 1, dst, 1, length);
}

// Scalar vector divide
void d4lib_sdivsv(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vDSP_svdiv(lhs, rhs, 1, dst, 1, length);
}
void d4lib_ddivsv(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vDSP_svdivD(lhs, rhs, 1, dst, 1, length);
}
void d4lib_idivsv(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    int l = *lhs;
    for (int i = 0; i < length; i++) {
        dst[i] = l / rhs[i];
    }
}

// Vector Sum
void d4lib_ssum(const float* src, d4lib_stride src_stride, float* dst, d4lib_length length) {
    vDSP_sve(src, src_stride, dst, length);
}
void d4lib_dsum(const double* src, d4lib_stride src_stride, double* dst, d4lib_length length) {
    vDSP_sveD(src, src_stride, dst, length);
}
void d4lib_isum(const int* src, d4lib_stride src_stride, int* dst, d4lib_length length)  {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += src[i * src_stride];
    }
    *dst = sum;
}

// Vector dot product
void d4lib_sdot(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length length) {
    vDSP_dotpr(lhs, lhs_stride, rhs, rhs_stride, dst, length);
}
void d4lib_ddot(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length length) {
    vDSP_dotprD(lhs, lhs_stride, rhs, rhs_stride, dst, length);
}
void d4lib_idot(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += lhs[i * lhs_stride] * rhs[i * rhs_stride];
    }
    *dst = sum;
}

// Vector maximum value and index
void d4lib_smaxi(const float* src, d4lib_stride src_stride, float* dst, d4lib_length* dst_idx, d4lib_length length) {
    vDSP_maxvi(src, src_stride, dst, dst_idx, length);
}
void d4lib_dmaxi(const double* src, d4lib_stride src_stride, double* dst, d4lib_length* dst_idx, d4lib_length length) {
    vDSP_maxviD(src, src_stride, dst, dst_idx, length);
}
void d4lib_imaxi(const int* src, d4lib_stride src_stride, int* dst, d4lib_length* dst_idx, d4lib_length length) {
    d4lib_length maxi = -1;
    int maxv = INT_MIN;
    for (int i = 0; i < length; i++) {
        float v = src[i * src_stride];
        if (v > maxv) {
            maxv = v;
            maxi = i;
        }
    }
    *dst = maxv;
    *dst_idx = maxi;
}

// Vector minimum value and index
void d4lib_smini(const float* src, d4lib_stride src_stride, float* dst, d4lib_length* dst_idx, d4lib_length length) {
    vDSP_minvi(src, src_stride, dst, dst_idx, length);
}
void d4lib_dmini(const double* src, d4lib_stride src_stride, double* dst, d4lib_length* dst_idx, d4lib_length length) {
    vDSP_minviD(src, src_stride, dst, dst_idx, length);
}
void d4lib_imini(const int* src, d4lib_stride src_stride, int* dst, d4lib_length* dst_idx, d4lib_length length) {
    d4lib_length mini = -1;
    int minv = INT_MIN;
    for (int i = 0; i < length; i++) {
        float v = src[i * src_stride];
        if (v > minv) {
            minv = v;
            mini = i;
        }
    }
    *dst = minv;
    *dst_idx = mini;
}

// Vector vector max
void d4lib_smax(const float* lhs, const float* rhs, float* dst, d4lib_length length) {
    vDSP_vmax(lhs, 1, rhs, 1, dst, 1, length);
}
void d4lib_dmax(const double* lhs, const double* rhs, double* dst, d4lib_length length) {
    vDSP_vmaxD(lhs, 1, rhs, 1, dst, 1, length);
}
void d4lib_imax(const int* lhs, const int* rhs, int* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i] = MAX(lhs[i], rhs[i]);
    }
}

// Vector ramp
void d4lib_sramp(const float* start, const float* increment, float* dst, d4lib_length length) {
    vDSP_vramp(start, increment, dst, 1, length);
}
void d4lib_dramp(const double* start, const double* increment, double* dst, d4lib_length length) {
    vDSP_vrampD(start, increment, dst, 1, length);
}
void d4lib_iramp(const int* start, const int* increment, int* dst, d4lib_length length) {
    int s = *start;
    int k = *increment;
    for (int i = 0; i < length; i++) {
        dst[i] = s + i * k;
    }
}

// single vector math functions
void d4lib_stanh(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvtanhf(dst, src, &l);
}
void d4lib_dtanh(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvtanh(dst, src, &l);
}

void d4lib_sexp(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvexpf(dst, src, &l);
}
void d4lib_dexp(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvexp(dst, src, &l);
}

void d4lib_slog(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvlogf(dst, src, &l);
}
void d4lib_dlog(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvlog(dst, src, &l);
}

void d4lib_ssqrt(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvsqrtf(dst, src, &l);
}
void d4lib_dsqrt(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvsqrt(dst, src, &l);
}

void d4lib_ssin(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvsinf(dst, src, &l);
}
void d4lib_dsin(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvsin(dst, src, &l);
}

void d4lib_scos(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvcosf(dst, src, &l);
}
void d4lib_dcos(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvcos(dst, src, &l);
}

void d4lib_stan(const float* src, float* dst, d4lib_length length) {
    int l = length;
    vvtanf(dst, src, &l);
}
void d4lib_dtan(const double* src, double* dst, d4lib_length length) {
    int l = length;
    vvtan(dst, src, &l);
}

void d4lib_scopysign(const float* mag, const float* sig, float* dst, d4lib_length length) {
    int l = length;
    vvcopysignf(dst, mag, sig, &l);
}
void d4lib_dcopysign(const double* mag, const double* sig, double* dst, d4lib_length length) {
    int l = length;
    vvcopysign(dst, mag, sig, &l);
}

void d4lib_sheaviside(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        float s = src[i];
        dst[i] = s > 0 ? 1 : 0;
    }
}
void d4lib_dheaviside(const double* src, double* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        float s = src[i];
        dst[i] = s > 0 ? 1 : 0;
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

void d4lib_stranspose(const float* src, float* dst, d4lib_length src_cols, d4lib_length src_rows) {
    vDSP_mtrans(src, 1, dst, 1, src_cols, src_rows);
}
void d4lib_dtranspose(const double* src, double* dst, d4lib_length src_cols, d4lib_length src_rows) {
    vDSP_mtransD(src, 1, dst, 1, src_cols, src_rows);
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
        order == D4LIB_RowMajor ? CblasRowMajor : CblasColMajor,
        transA == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        transB == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c,
        ldc
    );
}
void d4lib_dgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    cblas_dgemm(
        order == D4LIB_RowMajor ? CblasRowMajor : CblasColMajor,
        transA == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        transB == D4LIB_Trans ? CblasTrans : CblasNoTrans,
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c,
        ldc
    );
}

#endif

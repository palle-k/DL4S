//
//  dl4s.lib
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

#include <stdlib.h>
void avxcpy(void* dst, const void* src, size_t count);

typedef unsigned long d4lib_length;
#if defined __arm64__ && !defined __LP64__
typedef long long d4lib_stride;
#else
typedef long d4lib_stride;
#endif

// name convention: d4lib_TOD
// T: type; either s (float), d (double), i (int32)
// O: operation
// D: dimensionality (optional); either m (matrix), v (vector), s (scalar)

// Vector Fill
void d4lib_sfill(const float* src, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dfill(const double* src, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ifill(const int* src, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector square
void d4lib_ssquare(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dsquare(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_isquare(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector threshold
void d4lib_sthreshold(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dthreshold(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ithreshold(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector negate
void d4lib_sneg(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dneg(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ineg(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector add
void d4lib_saddv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_daddv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_iaddv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector scalar add
void d4lib_saddvs(const float* lhs, d4lib_stride lhs_stride, const float* rhs , float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_daddvs(const double* lhs, d4lib_stride lhs_stride, const double* rhs , double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_iaddvs(const int* lhs, d4lib_stride lhs_stride, const int* rhs , int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector subtract
void d4lib_ssubv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dsubv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_isubv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Scalar vector subtract
void d4lib_ssubsv(const float* lhs, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dsubsv(const double* lhs, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_isubsv(const int* lhs, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector multiply
void d4lib_smulv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dmulv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_imulv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector scalar multiply
void d4lib_smulvs(const float* lhs, d4lib_stride lhs_stride, const float* rhs , float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dmulvs(const double* lhs, d4lib_stride lhs_stride, const double* rhs , double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_imulvs(const int* lhs, d4lib_stride lhs_stride, const int* rhs , int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector divide
void d4lib_sdivv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ddivv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_idivv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Scalar vector divide
void d4lib_sdivsv(const float* lhs, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ddivsv(const double* lhs, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_idivsv(const int* lhs, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector Sum
void d4lib_ssum(const float* src, d4lib_stride src_stride, float* dst, d4lib_length length);
void d4lib_dsum(const double* src, d4lib_stride src_stride, double* dst, d4lib_length length);
void d4lib_isum(const int* src, d4lib_stride src_stride, int* dst, d4lib_length length);

// Vector dot product
void d4lib_sdot(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length length);
void d4lib_ddot(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length length);
void d4lib_idot(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length length);

// Vector maximum value and index
void d4lib_smaxi(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_dmaxi(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_imaxi(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length* dst_idx, d4lib_length length);

// Vector minimum value and index
void d4lib_smini(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_dmini(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_imini(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length* dst_idx, d4lib_length length);

// Vector ramp
void d4lib_sramp(const float* start, const float* increment, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dramp(const double* start, const double* increment, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_iramp(const int* start, const int* increment, int* dst, d4lib_stride dst_stride, d4lib_length length);

// single vector math functions
void d4lib_stanh(const float* start, float* dst, d4lib_length length);
void d4lib_dtanh(const double* start, double* dst, d4lib_length length);

void d4lib_sexp(const float* start, float* dst, d4lib_length length);
void d4lib_dexp(const double* start, double* dst, d4lib_length length);

void d4lib_slog(const float* start, float* dst, d4lib_length length);
void d4lib_dlog(const double* start, double* dst, d4lib_length length);

void d4lib_ssqrt(const float* start, float* dst, d4lib_length length);
void d4lib_dsqrt(const double* start, double* dst, d4lib_length length);

void d4lib_ssin(const float* start, float* dst, d4lib_length length);
void d4lib_dsin(const double* start, double* dst, d4lib_length length);

void d4lib_scos(const float* start, float* dst, d4lib_length length);
void d4lib_dcos(const double* start, double* dst, d4lib_length length);

void d4lib_stan(const float* start, float* dst, d4lib_length length);
void d4lib_dtan(const double* start, double* dst, d4lib_length length);

void d4lib_scopysign(const float* start, float* dst, d4lib_length length);
void d4lib_dcopysign(const double* start, double* dst, d4lib_length length);

void d4lib_sheaviside(const float* start, float* dst, d4lib_length length);
void d4lib_dheaviside(const double* start, double* dst, d4lib_length length);

// matrix multiplication
typedef enum {
    D4LIB_RowMajor = 101,
    D4LIB_ColMajor = 102
} D4LIB_ORDER;


typedef enum {
    D4LIB_Trans = 111,
    D4LIB_NoTrans = 112
} D4LIB_TRANSPOSE;

void d4lib_sgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc);
void d4lib_dgemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc);
void d4lib_igemm(D4LIB_ORDER order, D4LIB_TRANSPOSE transA, D4LIB_TRANSPOSE transB, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc);

// high level data reordering operations
typedef struct {
    int batch_size;
    int channels;
    int height;
    int width;
    int kernel_height;
    int kernel_width;
    int padding;
    int stride;
} D4LIB_Img2ColSetup;

void d4lib_img2col(const float* src, float* dst, D4LIB_Img2ColSetup setup);
void d4lib_col2img(const float* src, float* dst, D4LIB_Img2ColSetup setup);

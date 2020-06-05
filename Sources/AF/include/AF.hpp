//
//  ArrayFire.h
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

#ifndef d4af_h
#define d4af_h

#include "arrayfire.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _d4af_array;
typedef struct _d4af_array *d4af_array;
typedef long long dim_t;

d4af_array d4af_allocate(dim_t count, af_dtype type);
void d4af_free(d4af_array array);

void d4af_assign_d2h(void* target, const d4af_array source);
void d4af_assign_d2d(d4af_array target, d4af_array source);
void d4af_assign_h2d(d4af_array target, const void* source, const size_t byte_count);

float d4af_get_pointee_32f(const d4af_array source);
double d4af_get_pointee_64f(const d4af_array source);
int d4af_get_pointee_32s(const d4af_array source);

void d4af_set_pointee_32f(const d4af_array dst, float value);
void d4af_set_pointee_64f(const d4af_array dst, double value);
void d4af_set_pointee_32s(const d4af_array dst, int value);

size_t d4af_get_size(const d4af_array source);

void d4af_subscript(d4af_array dst, const d4af_array src, const int* shape, const int* indices);
void d4af_subscript_range(d4af_array dst, const d4af_array src, const int* shape, const int* lower_bounds, const int* upper_bounds);

void d4af_subscript_write(d4af_array dst, const d4af_array src, const int* shape, const int* indices);
void d4af_subscript_write_range(d4af_array dst, const d4af_array src, const int* shape, const int* lower_bounds, const int* upper_bounds);

void d4af_fill_32f(d4af_array dst, float value);
void d4af_fill_64f(d4af_array dst, double value);
void d4af_fill_32s(d4af_array dst, int value);

void d4af_randu_32f(d4af_array dst, float min, float max, dim_t count);
void d4af_randu_64f(d4af_array dst, double min, double max, dim_t count);
void d4af_randu_32s(d4af_array dst, int min, int max, dim_t count);

void d4af_randn_32f(d4af_array dst, float mean, float stdev, dim_t count);
void d4af_randn_64f(d4af_array dst, double mean, double stdev, dim_t count);
void d4af_randn_32s(d4af_array dst, int mean, int stdev, dim_t count);

void d4af_randb(d4af_array dst, float prob, af_dtype type, dim_t count);

void d4af_neg(d4af_array dst, const d4af_array src);
void d4af_add(d4af_array dst, const d4af_array lhs, const d4af_array rhs);
void d4af_sub(d4af_array dst, const d4af_array lhs, const d4af_array rhs);
void d4af_mul(d4af_array dst, const d4af_array lhs, const d4af_array rhs);
void d4af_div(d4af_array dst, const d4af_array lhs, const d4af_array rhs);

void d4af_broadcast_add(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape);
void d4af_broadcast_sub(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape);
void d4af_broadcast_mul(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape);
void d4af_broadcast_div(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape);

void d4af_gemm(d4af_array dst, const d4af_array lhs, const d4af_array rhs, bool transposeLhs, bool transposeRhs, float alpha, float beta, dim_t lhs_rows, dim_t lhs_cols, dim_t rhs_rows, dim_t rhs_cols, dim_t dst_rows, dim_t dst_cols);

void d4af_reduce_sum(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);
void d4af_sum_all(d4af_array dst, const d4af_array src);
void d4af_reduce_sum_multi(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t num_reduce, const dim_t* reduce_dims);
void d4af_reduce_mean(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);
void d4af_mean_all(d4af_array dst, const d4af_array src);
void d4af_reduce_max(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);
void d4af_reduce_max_ctx(d4af_array dst, d4af_array ctx, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);
void d4af_reduce_min(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);
void d4af_reduce_min_ctx(d4af_array dst, d4af_array ctx, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim);

int d4af_argmax(d4af_array dst, const d4af_array src);
int d4af_argmin(d4af_array dst, const d4af_array src);

void d4af_max(d4af_array dst, const d4af_array lhs, const d4af_array rhs);
void d4af_min(d4af_array dst, const d4af_array lhs, const d4af_array rhs);

void d4af_max_ctx(d4af_array dst, d4af_array ctx, const d4af_array lhs, const d4af_array rhs);
void d4af_min_ctx(d4af_array dst, d4af_array ctx, const d4af_array lhs, const d4af_array rhs);

void d4af_exp(d4af_array dst, const d4af_array src);
void d4af_log(d4af_array dst, const d4af_array src);
void d4af_sqrt(d4af_array dst, const d4af_array src);
void d4af_relu(d4af_array dst, const d4af_array src);
void d4af_heaviside(d4af_array dst, const d4af_array src);
void d4af_sin(d4af_array dst, const d4af_array src);
void d4af_cos(d4af_array dst, const d4af_array src);
void d4af_tan(d4af_array dst, const d4af_array src);
void d4af_sinh(d4af_array dst, const d4af_array src);
void d4af_cosh(d4af_array dst, const d4af_array src);
void d4af_tanh(d4af_array dst, const d4af_array src);

void d4af_permute(d4af_array dst, const d4af_array src, const dim_t dims, const dim_t* shape, const dim_t* arangement);
void d4af_permute_add(d4af_array dst, const d4af_array src, const dim_t dims, const dim_t* shape, const dim_t* arangement, const d4af_array add);

void d4af_reverse(d4af_array dst, const d4af_array src);
void d4af_reverse_add(d4af_array dst, const d4af_array src, const d4af_array add);

void d4af_stack(d4af_array dst, const d4af_array* srcs, const unsigned int numel, const dim_t* shapes, const int dim);
void d4af_unstack(const d4af_array* dsts, const size_t num_dst, const dim_t* unstack_dim_lengths, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const int dim);
void d4af_unstack_add(const d4af_array* dsts, const d4af_array* add, const size_t num_dst, const dim_t* unstack_dim_lengths, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const int dim);

void d4af_gather(d4af_array dst, const d4af_array src, const d4af_array ctx, const dim_t* src_shape, int dim);
void d4af_scatter(d4af_array dst, const d4af_array src, const d4af_array ctx, const dim_t* dst_shape, int dim);

void d4af_im2col(d4af_array dst, const d4af_array src, dim_t batch_size, dim_t channels, dim_t rows, dim_t columns, dim_t window_width, dim_t window_height, dim_t stride, dim_t pad);
void d4af_col2im(d4af_array dst, const d4af_array src, dim_t batch_size, dim_t channels, dim_t rows, dim_t columns, dim_t window_width, dim_t window_height, dim_t stride, dim_t pad);

void d4af_arange_32f(d4af_array dst, float lower_bound, float upper_bound, dim_t count);
void d4af_arange_64f(d4af_array dst, double lower_bound, double upper_bound, dim_t count);
void d4af_arange_32s(d4af_array dst, int lower_bound, int upper_bound, dim_t count);

void d4af_band(d4af_array dst, const d4af_array src, int rows, int columns, int below_diag, int above_diag);

#ifdef __cplusplus
}
#endif

#endif /* d4af_h */

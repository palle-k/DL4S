// needs to stay
#include "AF.hpp"
#include "arrayfire.h"
#include "af/data.h"
#include "af/array.h"
#include "af/gfor.h"
#include "af/internal.h"
#include <csignal>
#include <iostream>


extern "C" {
    struct _d4af_array {
        af::array array;
    };
}

d4af_array d4af_allocate(dim_t count, af_dtype type) {
    af::array array = af::array(count, type);
    d4af_array result = new struct _d4af_array;
    result->array = array;
    return result;
}

void d4af_free(d4af_array array) {
    array->array.~array();
    free(array);
}

void d4af_assign_h2d(d4af_array target, const void* source, const size_t byte_count) {
    target->array.write(source, byte_count);
}

void d4af_assign_d2d(const d4af_array target, const d4af_array source) {
    target->array.operator=(source->array);
}

void d4af_assign_d2h(void* target, const d4af_array source) {
    source->array.eval();
    source->array.host(target);
}

float d4af_get_pointee_32f(const d4af_array source) {
    source->array.eval();
    return source->array.scalar<float>();
}

double d4af_get_pointee_64f(const d4af_array source) {
    source->array.eval();
    return source->array.scalar<double>();
}

int d4af_get_pointee_32s(const d4af_array source) {
    source->array.eval();
    return source->array.scalar<int>();
}

size_t d4af_get_size(const d4af_array source) {
    source->array.eval();
    return source->array.elements();
}

void d4af_fill_32f(d4af_array dst, float value) {
    dst->array = value;
}

void d4af_fill_64f(d4af_array dst, double value) {
    dst->array = value;
}

void d4af_fill_32s(d4af_array dst, int value) {
    dst->array = value;
}

void d4af_randu_32f(d4af_array dst, float min, float max, dim_t count) {
    dst->array = af::randu(count, f32) * (max - min) + min;
}

void d4af_randu_64f(d4af_array dst, double min, double max, dim_t count) {
    dst->array = af::randu(count, f64) * (max - min) + min;
}

void d4af_randu_32s(d4af_array dst, int min, int max, dim_t count) {
    dst->array = af::randu(count, s32) * (max - min) + min;
}

void d4af_randn_32f(d4af_array dst, float mean, float stdev, dim_t count) {
    dst->array = af::randn(count, f32) * stdev + mean;
}

void d4af_randn_64f(d4af_array dst, double mean, double stdev, dim_t count) {
    dst->array = af::randn(count, f64) * stdev + mean;
}

void d4af_randn_32s(d4af_array dst, int mean, int stdev, dim_t count) {
    dst->array = af::randn(count, s32) * stdev + mean;
}

void d4af_randb(d4af_array dst, float prob, af_dtype type, dim_t count) {
    dst->array = (af::randu(count, f32) <= prob).as(type);
}

void d4af_subscript(d4af_array dst, const d4af_array src, const int* shape, const int* indices) {
    af::index* index = (af::index*) alloca(sizeof(af::index) * 4);
    for (int i = 0; i < 4; i++) {
        dim_t ii = indices[i];
        if (ii == -1) {
            index[i] = af::index(af::span);
        } else {
            index[i] = af::index(ii);
        }
    }
    dst->array = af::moddims(src->array, (dim_t) shape[0], (dim_t) shape[1], (dim_t) shape[2], (dim_t) shape[3])(index[0], index[1], index[2], index[3]);
    dst->array = af::flat(dst->array);
}

void d4af_subscript_range(d4af_array dst, const d4af_array src, const int* shape, const int* lower_bounds, const int* upper_bounds) {
    af::index* index = (af::index*) alloca(sizeof(af::index) * 4);
    for (int i = 0; i < 4; i++) {
        if (lower_bounds[i] == -1) {
            index[i] = af::index(af::span);
        } else {
            index[i] = af::index(af::seq((double) lower_bounds[i], (double) upper_bounds[i] - 1));
        }
    }
    dst->array = af::moddims(src->array, (dim_t) shape[0], (dim_t) shape[1], (dim_t) shape[2], (dim_t) shape[3])(index[0], index[1], index[2], index[3]);
    dst->array = af::flat(dst->array);
}

void d4af_subscript_write(d4af_array dst, const d4af_array src, const int* shape, const int* indices) {
#warning "Implementation for subscript_write is currently incorrect"
    af::index* index = (af::index*) alloca(sizeof(af::index) * 4);
    for (int i = 0; i < 4; i++) {
        dim_t ii = indices[i];
        if (ii == -1) {
            index[i] = af::index(af::span);
        } else {
            index[i] = af::index(ii);
        }
    }
    
    dst->array = af::moddims(src->array, (dim_t) shape[0], (dim_t) shape[1], (dim_t) shape[2], (dim_t) shape[3])(index[0], index[1], index[2], index[3]);
    dst->array = af::flat(dst->array);
}

void d4af_subscript_range_write(d4af_array dst, const d4af_array src, const int* shape, const int* lower_bounds, const int* upper_bounds) {
#warning "Implementation for subscript_range_write is currently incorrect"
    af::index* index = (af::index*) alloca(sizeof(af::index) * 4);
    for (int i = 0; i < 4; i++) {
        if (lower_bounds[i] == -1) {
            index[i] = af::index(af::span);
        } else {
            index[i] = af::index(af::seq((double) lower_bounds[i], (double) upper_bounds[i] - 1));
        }
    }
    dst->array = af::moddims(src->array, (dim_t) shape[0], (dim_t) shape[1], (dim_t) shape[2], (dim_t) shape[3])(index[0], index[1], index[2], index[3]);
    dst->array = af::flat(dst->array);
}

void d4af_neg(d4af_array dst, const d4af_array src) {
    dst->array = -af::flat(src->array);
}

void d4af_add(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::flat(lhs->array) + af::flat(rhs->array);
}

void d4af_sub(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::flat(lhs->array) - af::flat(rhs->array);
}

void d4af_mul(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::flat(lhs->array) * af::flat(rhs->array);
}

void d4af_div(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::flat(lhs->array) / af::flat(rhs->array);
}

void d4af_band(d4af_array dst, const d4af_array src, int belowDiag, int aboveDiag) {
    // TODO: Implement
    // Probably best way is to add diagonals from af::diag
    std::raise(SIGINT);
}

void d4af_broadcast_add(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape) {
    // https://gist.github.com/pavanky/a6904653333c5c196d82
    af::gforSet(true);
    dst->array = af::moddims(lhs->array, dims, lhs_shape) + af::moddims(rhs->array, dims, rhs_shape);
    af::gforSet(false);
}

void d4af_broadcast_mul(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape) {
    // https://gist.github.com/pavanky/a6904653333c5c196d82
    af::gforSet(true);
    dst->array = af::moddims(lhs->array, dims, lhs_shape) * af::moddims(rhs->array, dims, rhs_shape);
    af::gforSet(false);
}

void d4af_broadcast_sub(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape) {
    // https://gist.github.com/pavanky/a6904653333c5c196d82
    af::gforSet(true);
    dst->array = af::moddims(lhs->array, dims, lhs_shape) - af::moddims(rhs->array, dims, rhs_shape);
    af::gforSet(false);
}

void d4af_broadcast_div(d4af_array dst, const d4af_array lhs, const d4af_array rhs, const dim_t dims, const dim_t* lhs_shape, const dim_t* rhs_shape) {
    // https://gist.github.com/pavanky/a6904653333c5c196d82
    af::gforSet(true);
    dst->array = af::moddims(lhs->array, dims, lhs_shape) / af::moddims(rhs->array, dims, rhs_shape);
    af::gforSet(false);
}

void d4af_gemm(
               d4af_array dst,
               const d4af_array lhs,
               const d4af_array rhs,
               bool transposeLhs,
               bool transposeRhs,
               float alpha,
               float beta,
               dim_t lhs_rows,
               dim_t lhs_cols,
               dim_t rhs_rows,
               dim_t rhs_cols,
               dim_t dst_rows,
               dim_t dst_cols
) {
    af::array lhs_view = af::moddims(lhs->array, lhs_cols, lhs_rows);
    af::array rhs_view = af::moddims(rhs->array, rhs_cols, rhs_rows);

    // Apparently, ArrayFire works the other way around
    af::array result = af::transpose(af::matmul(lhs_view, rhs_view, transposeLhs ? AF_MAT_NONE : AF_MAT_TRANS, transposeRhs ? AF_MAT_NONE : AF_MAT_TRANS));
    
    if (beta == 0) {
        dst->array = alpha * result;
    } else {
        dst->array *= beta;
        dst->array += alpha * af::flat(result);
    }
}

void d4af_reduce_sum(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    dst->array = af::flat(af::sum(src_view, reduce_dim));
}

void d4af_sum_all(d4af_array dst, const d4af_array src) {
    dst->array = af::sum(src->array);
}

void d4af_reduce_mean(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    dst->array = af::flat(af::mean(src_view, reduce_dim));
}

void d4af_mean_all(d4af_array dst, const d4af_array src) {
    dst->array = af::mean(src->array);
}

void d4af_reduce_max(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    dst->array = af::flat(af::max(src_view, (int) reduce_dim));
}

void d4af_reduce_max_ctx(d4af_array dst, d4af_array ctx, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    af::max(dst->array, ctx->array, src_view);
}

void d4af_reduce_min(d4af_array dst, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    dst->array = af::flat(af::min(src_view, (int) reduce_dim));
}

void d4af_reduce_min_ctx(d4af_array dst, d4af_array ctx, const d4af_array src, const dim_t src_dim, const dim_t* src_shape, const dim_t reduce_dim) {
    af::array src_view = af::moddims(src->array, src_dim, src_shape);
    af::min(dst->array, ctx->array, src_view);
}

int d4af_argmax(d4af_array dst, const d4af_array src) {
    af::array maxIdx = af::array(1, u32);
    af::max(dst->array, maxIdx, (const af::array) src->array);
    maxIdx.eval();
    return (int) maxIdx.scalar<unsigned>();
}

int d4af_argmin(d4af_array dst, const d4af_array src) {
    auto minIdx = af::array(1, u32);
    af::min(dst->array, minIdx, (const af::array) src->array);
    minIdx.eval();
    return (int) minIdx.scalar<unsigned>();
}

void d4af_exp(d4af_array dst, const d4af_array src) {
    dst->array = af::exp(src->array);
}

void d4af_log(d4af_array dst, const d4af_array src) {
    dst->array = af::log(src->array);
}

void d4af_sqrt(d4af_array dst, const d4af_array src) {
    dst->array = af::sqrt(src->array);
}

void d4af_relu(d4af_array dst, const d4af_array src) {
    dst->array = (src->array > 0) * src->array;
}

void d4af_heaviside(d4af_array dst, const d4af_array src) {
    dst->array = (src->array > 0) * 1.0f;
}

void d4af_sin(d4af_array dst, const d4af_array src) {
    dst->array = af::sin(src->array);
}

void d4af_cos(d4af_array dst, const d4af_array src) {
    dst->array = af::cos(src->array);
}

void d4af_tan(d4af_array dst, const d4af_array src) {
    dst->array = af::tan(src->array);
}

void d4af_sinh(d4af_array dst, const d4af_array src) {
    dst->array = af::sinh(src->array);
}

void d4af_cosh(d4af_array dst, const d4af_array src) {
    dst->array = af::cosh(src->array);
}

void d4af_tanh(d4af_array dst, const d4af_array src) {
    dst->array = af::tanh(src->array);
}

void d4af_max(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::max(lhs->array, rhs->array);
}

void d4af_min(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::min(lhs->array, rhs->array);
}

void d4af_max_ctx(d4af_array dst, d4af_array ctx, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::max(af::flat(lhs->array), af::flat(rhs->array));
    ctx->array = (lhs->array < rhs->array).as(s32);
}

void d4af_min_ctx(d4af_array dst, d4af_array ctx, const d4af_array lhs, const d4af_array rhs) {
    dst->array = af::min(af::flat(lhs->array), af::flat(rhs->array));
    ctx->array = (lhs->array > rhs->array).as(s32);
}

void d4af_permute(d4af_array dst, const d4af_array src, const dim_t dims, const dim_t* shape, const dim_t* arangement) {
    // arangement must be 1-padded permutation of length 4.
    auto src_view = af::moddims(src->array, dims, shape);
    dst->array = af::reorder(src_view, arangement[0], arangement[1], arangement[2], arangement[3]);
}

void d4af_permute_add(d4af_array dst, const d4af_array src, const dim_t dims, const dim_t* shape, const dim_t* arangement, const d4af_array add) {
    // arangement must be 1-padded permutation of length 4.
    auto src_view = af::moddims(src->array, dims, shape);
    dst->array = af::flat(add->array) + af::flat(af::reorder(src_view, arangement[0], arangement[1], arangement[2], arangement[3]));
}

void d4af_reverse(d4af_array dst, const d4af_array src) {
    dst->array = af::flip(af::flat(src->array), 0);
}

void d4af_reverse_add(d4af_array dst, const d4af_array src, const d4af_array add) {
    dst->array = add->array + af::flip(af::flat(src->array), 0);
}

void d4af_stack(d4af_array dst, const d4af_array* srcs, const unsigned int numel, const dim_t* shapes, const int dim) {
    if (numel == 0) {
        dst->array = af::array(1);
        return;
    }
    af::array second;
    af::array third;
    af::array fourth;
    
    dst->array = af::moddims(srcs[0]->array, shapes[0], shapes[1], shapes[2], shapes[3]);
    
    for (unsigned int i = 1; i < numel; i += 3) {
        switch ((numel - i) > 3 ? 3 : (numel - i)) {
            case 0:
                dst->array = af::array();
                break;
            case 1:
                second = af::moddims(srcs[i]->array, shapes[i*4], shapes[i*4+1], shapes[i*4+2], shapes[i*4+3]);
                dst->array = af::join(dim, dst->array, second);
                break;
            case 2:
                second = af::moddims(srcs[i]->array, shapes[i*4], shapes[i*4+1], shapes[i*4+2], shapes[i*4+3]);
                third = af::moddims(srcs[i+1]->array, shapes[i*4+4], shapes[i*4+5], shapes[i*4+6], shapes[i*4+7]);
                dst->array = af::join(dim, dst->array, second, third);
                break;
            case 3:
                second = af::moddims(srcs[i]->array, shapes[i*4], shapes[i*4+1], shapes[i*4+2], shapes[i*4+3]);
                third = af::moddims(srcs[i+1]->array, shapes[i*4+4], shapes[i*4+5], shapes[i*4+6], shapes[i*4+7]);
                fourth = af::moddims(srcs[i+2]->array, shapes[i*4+8], shapes[i*4+9], shapes[i*4+10], shapes[i*4+11]);
                dst->array = af::join(dim, dst->array, second, third, fourth);
                break;
            default:
                break;
        }
    }
}

void d4af_gather(d4af_array dst, const d4af_array src, const d4af_array ctx, const dim_t* src_shape, int dim) {
    auto index = af::index(ctx->array);
    auto src_view = af::moddims(src->array, src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
    
    switch (dim) {
        case 0:
            dst->array = src_view(index, af::span);
            break;
        case 1:
            dst->array = src_view(af::span, index);
            break;
        case 2:
            dst->array = src_view(af::span, af::span, index);
            break;
        case 3:
            dst->array = src_view(af::span, af::span, af::span, index);
            break;
        default:
            break;
    }
    // TODO: There may be a better way to do this.
    dst->array = af::diag(dst->array);
}

void d4af_scatter(d4af_array dst, const d4af_array src, const d4af_array ctx, const dim_t* dst_shape, int dim) {
    af::array src_view = af::flat(src->array);
    af::array indices = af::flat(ctx->array);
    
    af::array result = af::constant(0.0f, dst->array.elements());
    
    dim_t n_cols = dst_shape[0];
    
    af::array linear_idx;
    
    if (dim == 1) {
        af::array col_idx = af::iota((float) ctx->array.elements());
        af::array row_idx = indices;
        
        linear_idx = col_idx + row_idx * (float) n_cols;
    } else if (dim == 0) {
        af::array row_idx = af::iota((float) ctx->array.elements());
        af::array col_idx = indices;
        
        linear_idx = col_idx + row_idx * (float) n_cols;
    } else {
        std::cerr << "Invalid scatter dimension, can only scatter along dim 0 or 1." << std::endl;
        raise(SIGINT);
        return;
    }
    result(linear_idx) = af::flat(src->array);
    dst->array = result;
}

void d4af_arange_32f(d4af_array dst, float lower_bound, float upper_bound, dim_t count) {
    dst->array = af::range(af::dim4(count), f32) * (upper_bound - lower_bound) + lower_bound;
}

void d4af_arange_64f(d4af_array dst, double lower_bound, double upper_bound, dim_t count) {
    dst->array = af::range(af::dim4(count), f64) * (upper_bound - lower_bound) + lower_bound;
}

void d4af_arange_32s(d4af_array dst, int lower_bound, int upper_bound, dim_t count) {
    dst->array = af::range(af::dim4(count), s32) * (upper_bound - lower_bound) + lower_bound;
}

af::array af_ext_pad(const af::array src, dim_t lx, dim_t rx, dim_t ly, dim_t ry, dim_t lz, dim_t rz, dim_t lw, dim_t rw) {
    // af::pad/af_pad not available in API_VERSION 36 and lower, API_VERSION 37 not available for macOS.
    auto dims = src.dims();
    af::array dst = af::array(dims.dims[0] + lx + rx, dims.dims[1] + ly + ry, dims.dims[2] + lz + rz, dims.dims[3] + lw + rw);
    dst = 0;
    dst(af::seq(lx, lx + dims.dims[0] - 1), af::seq(ly, ly + dims.dims[1] - 1), af::seq(lz, lz + dims.dims[2] - 1), af::seq(lw, lw + dims.dims[3] - 1)) = src;
    return dst;
}

void d4af_im2col(d4af_array dst, const d4af_array src, dim_t batch_size, dim_t channels, dim_t rows, dim_t columns, dim_t window_width, dim_t window_height, dim_t stride, dim_t pad) {
    auto src_view = af_ext_pad(af::moddims(src->array, columns, rows, channels, batch_size), pad, pad, pad, pad, 0, 0, 0, 0);
    dst->array = af::unwrap(src_view, window_width, window_height, stride, stride);
}

void d4af_col2im(d4af_array dst, const d4af_array src, dim_t batch_size, dim_t channels, dim_t rows, dim_t columns, dim_t window_width, dim_t window_height, dim_t stride, dim_t pad) {
    auto dst_view = af::wrap(src->array, columns + 2 * pad, rows + 2 * pad, window_width, window_height, stride, stride);
    dst->array = dst_view(af::seq(pad, columns - pad - 1), af::seq(pad, rows - pad - 1));
}

void d4af_api_version() {
    int api = AF_API_VERSION;
    std::cout << "API VERSION: " << api << std::endl;
}

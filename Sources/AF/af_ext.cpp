// needs to stay
#include "AF.hpp"
#include "arrayfire.h"
#include "af/array.h"
#include "af/gfor.h"
#include <csignal>

#include "arrayfire.h"

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
    dst->array.operator=(value);
}

void d4af_fill_64f(d4af_array dst, double value) {
    dst->array.operator=(value);
}

void d4af_fill_32s(d4af_array dst, int value) {
    dst->array.operator=(value);
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
    dst->array.operator=(-src->array);
}

void d4af_add(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array = (lhs->array + rhs->array);
}

void d4af_sub(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array.operator=(lhs->array - rhs->array);
}

void d4af_mul(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array.operator=(lhs->array * rhs->array);
}

void d4af_div(d4af_array dst, const d4af_array lhs, const d4af_array rhs) {
    dst->array.operator=(lhs->array / rhs->array);
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
    auto maxIdx = af::array();
    af::max(dst->array, maxIdx, (const af::array) src->array);
    maxIdx.eval();
    return maxIdx.scalar<int>();
}

int d4af_argmin(d4af_array dst, const d4af_array src) {
    auto minIdx = af::array();
    af::min(dst->array, minIdx, (const af::array) src->array);
    minIdx.eval();
    return minIdx.scalar<int>();
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
    dst->array = af::reorder(src->array, arangement[0], arangement[1], arangement[2], arangement[3]);
}

void d4af_permute_add(d4af_array dst, const d4af_array src, const dim_t dims, const dim_t* shape, const dim_t* arangement, const d4af_array add) {
    // arangement must be 1-padded permutation of length 4.
    auto src_view = af::moddims(src->array, dims, shape);
    dst->array = add->array + af::reorder(src->array, arangement[0], arangement[1], arangement[2], arangement[3]);
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
    
    dst->array = af::moddims(srcs[0]->array, shapes[00], shapes[1], shapes[2], shapes[3]);
    
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
                third = af::moddims(srcs[i]->array, shapes[i*4+4], shapes[i*4+5], shapes[i*4+6], shapes[i*4+7]);
                dst->array = af::join(dim, dst->array, second, third);
                break;
            case 3:
                second = af::moddims(srcs[i]->array, shapes[i*4], shapes[i*4+1], shapes[i*4+2], shapes[i*4+3]);
                third = af::moddims(srcs[i]->array, shapes[i*4+4], shapes[i*4+5], shapes[i*4+6], shapes[i*4+7]);
                fourth = af::moddims(srcs[i]->array, shapes[i*4+8], shapes[i*4+9], shapes[i*4+10], shapes[i*4+11]);
                dst->array = af::join(dim, dst->array, second, third, fourth);
                break;
            default:
                break;
        }
    }
}

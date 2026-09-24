//
//  reduction.metal
//  DL4S
//
//  Created by Palle Klewitz on 24.09.26.
//  Copyright (c) 2026 - Palle Klewitz
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

// Reductions of a tensor with the shape [outer, length, inner] along the middle axis.
//
// Rows (inner = 1) are reduced by a SIMD group or a threadgroup per row. Columns (inner > 1) are reduced by one thread
// per column, so that neighboring threads read neighboring elements. Both forms can split the reduced axis into segments,
// which a second pass combines, so that long reductions of few rows or columns use the whole GPU.

struct ReduceParameters {
    uint outer;
    uint length;
    uint inner;
    uint segmentLength;
    uint segments;
    float scale;
    // Whether the result is added to the current values of the result buffer.
    uint accumulate;
};

#define SUM_COMBINE(a, b) ((a) + (b))
#define MAX_COMBINE(a, b) max((a), (b))
#define MIN_COMBINE(a, b) min((a), (b))
#define SCALE_float(value, scale) ((value) * (scale))
#define SCALE_int(value, scale) (value)
#define NO_SCALE(value, scale) (value)

// Rows: the threadgroup has the size (width, height). A row segment is reduced by `width` threads: by one SIMD group
// when the width is 32, with `height` rows per threadgroup, or by the whole threadgroup when the width is larger, with one row.
#define REDUCE_ROWS(NAME, T, IDENTITY, COMBINE, SIMD_REDUCE, SCALE) \
kernel void reduce_rows_##NAME##_##T(device const T* values [[buffer(0)]], device T* result [[buffer(1)]], constant ReduceParameters& p [[buffer(2)]], \
    uint2 local [[thread_position_in_threadgroup]], uint2 size [[threads_per_threadgroup]], uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup T partial[32]; \
    uint output = group.y * size.y + local.y; \
    bool active = output < p.outer * p.segments; \
    T accumulator = IDENTITY; \
    if (active) { \
        uint row = output / p.segments, segment = output % p.segments; \
        uint start = segment * p.segmentLength, end = min(start + p.segmentLength, p.length); \
        device const T* values_row = values + ulong(row) * p.length; \
        for (uint i = start + local.x; i < end; i += size.x) { accumulator = COMBINE(accumulator, values_row[i]); } \
    } \
    accumulator = SIMD_REDUCE(accumulator); \
    if (size.x > 32) { \
        if (lane == 0) { partial[simd] = accumulator; } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        accumulator = lane < size.x / 32 ? partial[lane] : IDENTITY; \
        accumulator = SIMD_REDUCE(accumulator); \
    } \
    if (active && local.x == 0) { \
        T value = SCALE(accumulator, T(p.scale)); \
        result[output] = p.accumulate ? result[output] + value : value; \
    } \
}

// Columns: one thread per column and segment. The result has the shape [segments, outer, inner].
#define REDUCE_COLUMNS(NAME, T, IDENTITY, COMBINE, SCALE) \
kernel void reduce_columns_##NAME##_##T(device const T* values [[buffer(0)]], device T* result [[buffer(1)]], constant ReduceParameters& p [[buffer(2)]], uint3 position [[thread_position_in_grid]]) { \
    if (position.x >= p.inner || position.y >= p.outer || position.z >= p.segments) { return; } \
    uint start = position.z * p.segmentLength, end = min(start + p.segmentLength, p.length); \
    device const T* column = values + ulong(position.y) * p.length * p.inner + position.x; \
    T accumulator = IDENTITY; \
    for (uint k = start; k < end; k++) { accumulator = COMBINE(accumulator, column[ulong(k) * p.inner]); } \
    ulong output = (ulong(position.z) * p.outer + position.y) * p.inner + position.x; \
    T value = SCALE(accumulator, T(p.scale)); \
    result[output] = p.accumulate ? result[output] + value : value; \
}

// Arguments of the maximum or minimum: the index along the reduced axis of the first element with the extreme value.
#define REDUCE_ROWS_ARG(NAME, T, IDENTITY, BETTER, SIMD_REDUCE) \
kernel void reduce_rows_arg##NAME##_##T(device const T* values [[buffer(0)]], device T* result [[buffer(1)]], device int* context [[buffer(2)]], constant ReduceParameters& p [[buffer(3)]], \
    uint2 local [[thread_position_in_threadgroup]], uint2 size [[threads_per_threadgroup]], uint2 group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup T partialValues[32]; \
    threadgroup uint partialIndices[32]; \
    uint row = group.y * size.y + local.y; \
    bool active = row < p.outer; \
    T best = IDENTITY; \
    uint bestIndex = UINT_MAX; \
    if (active) { \
        device const T* values_row = values + ulong(row) * p.length; \
        for (uint i = local.x; i < p.length; i += size.x) { \
            T value = values_row[i]; \
            if (bestIndex == UINT_MAX || BETTER(value, best)) { best = value; bestIndex = i; } \
        } \
    } \
    T extreme = SIMD_REDUCE(best); \
    bestIndex = simd_min(best == extreme ? bestIndex : UINT_MAX); \
    best = extreme; \
    if (size.x > 32) { \
        if (lane == 0) { partialValues[simd] = best; partialIndices[simd] = bestIndex; } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        best = lane < size.x / 32 ? partialValues[lane] : IDENTITY; \
        bestIndex = lane < size.x / 32 ? partialIndices[lane] : UINT_MAX; \
        extreme = SIMD_REDUCE(best); \
        bestIndex = simd_min(best == extreme && bestIndex != UINT_MAX ? bestIndex : UINT_MAX); \
        best = extreme; \
    } \
    if (active && local.x == 0) { result[row] = best; context[row] = int(bestIndex); } \
}

#define REDUCE_COLUMNS_ARG(NAME, T, BETTER) \
kernel void reduce_columns_arg##NAME##_##T(device const T* values [[buffer(0)]], device T* result [[buffer(1)]], device int* context [[buffer(2)]], constant ReduceParameters& p [[buffer(3)]], uint3 position [[thread_position_in_grid]]) { \
    if (position.x >= p.inner || position.y >= p.outer) { return; } \
    device const T* column = values + ulong(position.y) * p.length * p.inner + position.x; \
    T best = column[0]; \
    int bestIndex = 0; \
    for (uint k = 1; k < p.length; k++) { \
        T value = column[ulong(k) * p.inner]; \
        if (BETTER(value, best)) { best = value; bestIndex = int(k); } \
    } \
    result[ulong(position.y) * p.inner + position.x] = best; \
    context[ulong(position.y) * p.inner + position.x] = bestIndex; \
}

#define GREATER(a, b) ((a) > (b))
#define LESS(a, b) ((a) < (b))

REDUCE_ROWS(sum, float, 0.0f, SUM_COMBINE, simd_sum, SCALE_float)
REDUCE_ROWS(max, float, -INFINITY, MAX_COMBINE, simd_max, NO_SCALE)
REDUCE_ROWS(min, float, INFINITY, MIN_COMBINE, simd_min, NO_SCALE)
REDUCE_ROWS(sum, int, 0, SUM_COMBINE, simd_sum, SCALE_int)
REDUCE_ROWS(max, int, INT_MIN, MAX_COMBINE, simd_max, NO_SCALE)
REDUCE_ROWS(min, int, INT_MAX, MIN_COMBINE, simd_min, NO_SCALE)

REDUCE_COLUMNS(sum, float, 0.0f, SUM_COMBINE, SCALE_float)
REDUCE_COLUMNS(max, float, -INFINITY, MAX_COMBINE, NO_SCALE)
REDUCE_COLUMNS(min, float, INFINITY, MIN_COMBINE, NO_SCALE)
REDUCE_COLUMNS(sum, int, 0, SUM_COMBINE, SCALE_int)
REDUCE_COLUMNS(max, int, INT_MIN, MAX_COMBINE, NO_SCALE)
REDUCE_COLUMNS(min, int, INT_MAX, MIN_COMBINE, NO_SCALE)

REDUCE_ROWS_ARG(max, float, -INFINITY, GREATER, simd_max)
REDUCE_ROWS_ARG(min, float, INFINITY, LESS, simd_min)
REDUCE_ROWS_ARG(max, int, INT_MIN, GREATER, simd_max)
REDUCE_ROWS_ARG(min, int, INT_MAX, LESS, simd_min)

REDUCE_COLUMNS_ARG(max, float, GREATER)
REDUCE_COLUMNS_ARG(min, float, LESS)
REDUCE_COLUMNS_ARG(max, int, GREATER)
REDUCE_COLUMNS_ARG(min, int, LESS)

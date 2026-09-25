//
//  elementwise.metal
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

// Element-wise kernels: fill, copy, unary functions, and binary operators with broadcasting.
//
// The kernels are limited by memory bandwidth. One thread processes one element: on Apple GPUs, scalar loads of
// neighboring threads reach the same bandwidth as vector loads, and they need no alignment.

kernel void fill_u32(device uint* result [[buffer(0)]], constant uint& value [[buffer(1)]], constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) { result[i] = value; }
}

kernel void copy_u32(device const uint* source [[buffer(0)]], device uint* result [[buffer(1)]], constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) { result[i] = source[i]; }
}

#define UNARY(NAME, T, EXPRESSION) \
kernel void NAME##_##T(device const T* values [[buffer(0)]], device T* result [[buffer(1)]], constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    T x = values[i]; \
    result[i] = (EXPRESSION); \
}

UNARY(neg, float, -x)
UNARY(neg, int, -x)
UNARY(relu, float, max(x, 0.0f))
UNARY(relu, int, max(x, 0))
UNARY(heaviside, float, x > 0.0f ? 1.0f : 0.0f)
UNARY(heaviside, int, x > 0 ? 1 : 0)
UNARY(exp, float, precise::exp(x))
UNARY(log, float, precise::log(x))
UNARY(sqrt, float, precise::sqrt(x))
UNARY(sin, float, precise::sin(x))
UNARY(cos, float, precise::cos(x))
UNARY(tan, float, precise::tan(x))
UNARY(sinh, float, precise::sinh(x))
UNARY(cosh, float, precise::cosh(x))
UNARY(tanh, float, precise::tanh(x))

// Binary operators in four forms: vector-vector, vector-scalar, scalar-vector, and broadcast with a layout.
// The scalar forms read the scalar from device memory, so the host does not wait for the GPU to compute it.
#define BINARY(NAME, T, EXPRESSION) \
kernel void NAME##_vv_##T(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* result [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    T x = a[i]; T y = b[i]; \
    result[i] = (EXPRESSION); \
} \
kernel void NAME##_vs_##T(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* result [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    T x = a[i]; T y = b[0]; \
    result[i] = (EXPRESSION); \
} \
kernel void NAME##_sv_##T(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* result [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    T x = a[0]; T y = b[i]; \
    result[i] = (EXPRESSION); \
} \
kernel void NAME##_bc_##T(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* result [[buffer(2)]], constant uint& count [[buffer(3)]], constant Layout& layout [[buffer(4)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    int3 offsets = layout_offsets(layout, i); \
    T x = a[offsets.x]; T y = b[offsets.y]; \
    result[i] = (EXPRESSION); \
}

BINARY(add, float, x + y)
BINARY(sub, float, x - y)
BINARY(mul, float, x * y)
BINARY(div, float, x / y)
BINARY(add, int, x + y)
BINARY(sub, int, x - y)
BINARY(mul, int, x * y)
BINARY(div, int, y == 0 ? 0 : x / y)
BINARY(max, float, max(x, y))
BINARY(min, float, min(x, y))
BINARY(max, int, max(x, y))
BINARY(min, int, min(x, y))

// Maximum and minimum that also record which operand they select: 0 for the first, 1 for the second.
#define SELECT(NAME, T, CONDITION) \
kernel void NAME##_context_##T(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* result [[buffer(2)]], device T* context [[buffer(3)]], constant uint& count [[buffer(4)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    T x = a[i]; T y = b[i]; \
    bool first = (CONDITION); \
    result[i] = first ? x : y; \
    context[i] = first ? T(0) : T(1); \
}

SELECT(max, float, x >= y)
SELECT(min, float, x <= y)
SELECT(max, int, x >= y)
SELECT(min, int, x <= y)

kernel void arange_float(device float* result [[buffer(0)]], constant float2& startAndIncrement [[buffer(1)]], constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) { result[i] = startAndIncrement.x + float(i) * startAndIncrement.y; }
}

// Copies the elements of a matrix in the band around the diagonal and leaves the other elements of the result unchanged.
kernel void band_u32(device const uint* values [[buffer(0)]], device uint* result [[buffer(1)]], constant int4& bounds [[buffer(2)]], uint2 position [[thread_position_in_grid]]) {
    int row = int(position.y), column = int(position.x);
    int rows = bounds.x, columns = bounds.y;
    if (row >= rows || column >= columns) { return; }
    if (column >= row - bounds.z && column <= row + bounds.w) {
        result[row * columns + column] = values[row * columns + column];
    }
}

// Writes values to the positions i * stride of the result, or reads them from there when the direction is 1.
kernel void diagonal_u32(device const uint* values [[buffer(0)]], device uint* result [[buffer(1)]], constant uint3& countStrideDirection [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= countStrideDirection.x) { return; }
    if (countStrideDirection.z == 0) {
        result[i * countStrideDirection.y] = values[i];
    } else {
        result[i] = values[i * countStrideDirection.y];
    }
}

kernel void fill_diagonal_u32(device uint* result [[buffer(0)]], constant uint3& countStrideValue [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < countStrideValue.x) { result[i * countStrideValue.y] = countStrideValue.z; }
}

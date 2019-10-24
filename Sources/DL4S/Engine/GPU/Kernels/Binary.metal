//
//  Binary.metal
//  DL4S
//
//  Created by Palle Klewitz on 24.10.19.
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

#include <metal_stdlib>
using namespace metal;

#include "Shape.hpp"


kernel void vAdd_Float32(
    const device float* lhs_vals [[buffer(0)]],
    const device float* rhs_vals [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] + rhs_vals[pos];
}

kernel void vSub_Float32(
    const device float* lhs_vals [[buffer(0)]],
    const device float* rhs_vals [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] - rhs_vals[pos];
}

kernel void vMul_Float32(
    const device float* lhs_vals [[buffer(0)]],
    const device float* rhs_vals [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] * rhs_vals[pos];
}

kernel void vDiv_Float32(
    const device float* lhs_vals [[buffer(0)]],
    const device float* rhs_vals [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] / rhs_vals[pos];
}

kernel void vsAdd_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant float &rhs_val [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] + rhs_val;
}

kernel void vsMul_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant float &rhs_val [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_vals[pos] * rhs_val;
}

kernel void svDiv_Float32(
    constant float &lhs_val [[buffer(0)]],
    const device float* rhs_vals [[buffer(1)]],
    device float* result_vals [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = lhs_val / rhs_vals[pos];
}

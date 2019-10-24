//
//  Unary.metal
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

#include "Tensor.hpp"
#include "Shape.hpp"


kernel void vNeg_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = -src_vals[pos];
}

kernel void vExp_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = exp(src_vals[pos]);
}

kernel void vLog_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = log(src_vals[pos]);
}

kernel void vSqrt_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = sqrt(src_vals[pos]);
}

kernel void vSin_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = sin(src_vals[pos]);
}

kernel void vCos_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = cos(src_vals[pos]);
}

kernel void vTan_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = tan(src_vals[pos]);
}

kernel void vSinh_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = sinh(src_vals[pos]);
}

kernel void vCosh_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = cosh(src_vals[pos]);
}

kernel void vTanh_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = tanh(src_vals[pos]);
}

kernel void vSquare_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = src_vals[pos] * src_vals[pos];
}

kernel void vRelu_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = max(src_vals[pos], 0.0f);
}

kernel void vHeaviside_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* result_vals [[buffer(1)]],
    uint pos [[thread_position_in_grid]]
) {
    result_vals[pos] = src_vals[pos] > 0.0f ? 1.0f : 0.0f;
}

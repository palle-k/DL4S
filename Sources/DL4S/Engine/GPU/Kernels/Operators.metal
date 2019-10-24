//
//  kernels.metal
//  DL4S
//
//  Created by Palle Klewitz on 10.03.19.
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
#include "Tensor.hpp"


kernel void vFill_Float32(
    constant float &value [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    destination[pos] = value;
}

kernel void vFill_Int32(
    constant int &value [[buffer(0)]],
    device int *destination [[buffer(1)]],
    constant int &count [[buffer(2)]],
    uint pos [[thread_position_in_grid]]
) {
    destination[pos] = value;
}

kernel void vFillStride_Float32(
    constant float &value [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant int &count [[buffer(2)]],
    constant int &stride [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    destination[pos * stride] = value;
}

kernel void vFillStride_Int32(
    constant int &value [[buffer(0)]],
    device int *destination [[buffer(1)]],
    constant int &count [[buffer(2)]],
    constant int &stride [[buffer(3)]],
    uint pos [[thread_position_in_grid]]
) {
    destination[pos * stride] = value;
}

kernel void vScatter_AddInPlace_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    const device int* src_ctx [[buffer(4)]],
    device float* dst_vals [[buffer(7)]],
    constant int &dst_dim [[buffer(8)]],
    const device int* dst_shape [[buffer(9)]],
    const device int* dst_strides [[buffer(10)]],
    constant int &expanded_axis [[buffer(11)]],
    uint pos [[thread_position_in_grid]]
) {
    uint posBefore = pos / dst_strides[expanded_axis];
    uint posAfter = pos % dst_strides[expanded_axis];
    
    uint offset = posAfter + (expanded_axis != 0 ? posBefore * dst_strides[expanded_axis - 1] : 0);
    
    dst_vals[offset + src_ctx[pos]] += src_vals[pos];
}

kernel void mMul_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant int &lhs_dim [[buffer(1)]],
    const device int* lhs_shape [[buffer(2)]],
    const device float* rhs_vals [[buffer(3)]],
    constant int &rhs_dim [[buffer(4)]],
    const device int* rhs_shape [[buffer(5)]],
    device float* result_vals [[buffer(6)]],
    constant int &result_dim [[buffer(7)]],
    const device int* result_shape [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]
) {
    int column = pos.x;
    int row = pos.y;
    int count = lhs_shape[1];
    int rhs_cols = rhs_shape[1];
    
    float result = 0.0f;
    
    for (int i = 0; i < count; i++) {
        int lhs_idx = row * count + i;
        int rhs_idx = rhs_cols * i + column;
        
        result += lhs_vals[lhs_idx] * rhs_vals[rhs_idx];
    }
    
    int dstIdx = result_shape[1] * row + column;
    result_vals[dstIdx] = result;
}

kernel void mTrans_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_width [[buffer(1)]],
    constant int &src_height [[buffer(2)]],
    device float* dst_vals [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]
) {
    int column = pos.x;
    int row = pos.y;
    
    int src_idx = src_width * row + column;
    int dst_idx = src_height * column + row;
    
    dst_vals[dst_idx] = src_vals[src_idx];
}

kernel void permuteAxes_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    device float* dst_vals [[buffer(3)]],
    constant int &dst_dim [[buffer(4)]],
    const device int* dst_shape [[buffer(5)]],
    const device int* arangement [[buffer(6)]],
    const device int* strides [[buffer(7)]],
    uint pos [[thread_position_in_grid]]
) {
    auto src_sh = Shape(src_dim, src_shape);
    uint dst_pos = src_sh.permute(pos, strides, arangement);
    dst_vals[dst_pos] = src_vals[pos];
}

kernel void permuteAxesAdd_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    device float* add_vals [[buffer(3)]],
    constant int &add_dim [[buffer(4)]],
    const device int* add_shape [[buffer(5)]],
    device float* dst_vals [[buffer(6)]],
    constant int &dst_dim [[buffer(7)]],
    const device int* dst_shape [[buffer(8)]],
    const device int* arangement [[buffer(9)]],
    const device int* strides [[buffer(10)]],
    uint pos [[thread_position_in_grid]]
) {
    auto src_sh = Shape(src_dim, src_shape);
    uint dst_pos = src_sh.permute(pos, strides, arangement);
    dst_vals[dst_pos] = src_vals[pos] + add_vals[dst_pos];
}

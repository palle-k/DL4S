//
//  Reduce.metal
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
#include "Tensor.hpp"

constexpr constant int REDUCE_AXIS_LIMIT = 8;

class ReduceContext {
public:
    const int reduceDim;
    const int threadOffset;
    thread int* reduceIndex;
    device const int* reducedAxes;
    device const int* sourceStrides;
    const Shape sourceShape;
    
    ReduceContext(int reduceDim, int threadOffset, thread int* reduceIndex, device const int* reducedAxes, const Shape sourceShape, device const int* sourceStrides):
        reduceDim(reduceDim), threadOffset(threadOffset), reduceIndex(reduceIndex), reducedAxes(reducedAxes), sourceShape(sourceShape), sourceStrides(sourceStrides) {}
    
    int linearSourceIndex();
    void advance();
};

void ReduceContext::advance() {
    int i = this->reduceDim - 1;
    
    for (; i >= 0; i--) {
        this->reduceIndex[i]++;
        
        if (this->reduceIndex[i] >= this->sourceShape.shape[this->reducedAxes[i]]) {
            this->reduceIndex[i] = 0;
        } else {
            break;
        }
    }
}

int ReduceContext::linearSourceIndex() {
    int linearIndex = this->threadOffset;
    for (int i = 0; i < this->reduceDim; i++) {
        linearIndex += this->reduceIndex[i] * this->sourceStrides[this->reducedAxes[i]];
    }
    return linearIndex;
}

kernel void vSum_Reduce_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    constant int &reduced_axis [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    float sum = 0;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        sum += src_vals[i * stride + offset];
    }
    
    dst_vals[pos] = sum;
}

kernel void vSum_ReduceMulti_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    constant int &reduced_axis_count [[buffer(7)]],
    const device int* reduced_axes [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    thread int reduce_idx[REDUCE_AXIS_LIMIT];
    
    const auto sourceShape = Shape(src_dim, src_shape);
    const auto dstShape = Shape(dst_dim, dst_shape);
    const int baseIdx = dstShape.indexWithInsertedAxes(pos, reduced_axes, sourceShape);
    auto context = ReduceContext(reduced_axis_count, baseIdx, reduce_idx, reduced_axes, sourceShape, src_strides);
    
    int reduceCount = 1;
    for (int i = 0; i < reduced_axis_count; i++) {
        reduceCount *= src_shape[reduced_axes[i]];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < reduceCount; i++) {
        int idx = context.linearSourceIndex();
        sum += src_vals[idx];
        context.advance();
    }
    
    dst_vals[pos] = sum;
}

kernel void vMul_Reduce_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    constant int &reduced_axis [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    float prod = 1;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        prod *= src_vals[i * stride + offset];
    }
    
    dst_vals[pos] = prod;
}

kernel void vMul_ReduceMulti_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    constant int &reduced_axis_count [[buffer(7)]],
    const device int* reduced_axes [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    thread int reduce_idx[REDUCE_AXIS_LIMIT];
    
    const auto sourceShape = Shape(src_dim, src_shape);
    const auto dstShape = Shape(dst_dim, dst_shape);
    const int baseIdx = dstShape.indexWithInsertedAxes(pos, reduced_axes, sourceShape);
    auto context = ReduceContext(reduced_axis_count, baseIdx, reduce_idx, reduced_axes, sourceShape, src_strides);
    
    int reduceCount = 1;
    for (int i = 0; i < reduced_axis_count; i++) {
        reduceCount *= src_shape[reduced_axes[i]];
    }
    
    float product = 1.0f;
    for (int i = 0; i < reduceCount; i++) {
        int idx = context.linearSourceIndex();
        product *= src_vals[idx];
        context.advance();
    }
    
    dst_vals[pos] = product;
}

kernel void vMax_ReduceCtx_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    device int* dst_ctx [[buffer(8)]],
    constant int &reduced_axis [[buffer(11)]],
    uint pos [[thread_position_in_grid]]
) {
    float maxVal = -INFINITY;
    int ctx = 0;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        auto v = src_vals[i * stride + offset];
        if (v > maxVal) {
            maxVal = v;
            ctx = i;
        }
    }
    
    dst_vals[pos] = maxVal;
    dst_ctx[pos] = ctx;
}

kernel void vMin_ReduceCtx_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    device int* dst_ctx [[buffer(8)]],
    constant int &reduced_axis [[buffer(11)]],
    uint pos [[thread_position_in_grid]]
) {
    float minVal = INFINITY;
    int ctx = 0;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        auto v = src_vals[i * stride + offset];
        if (v < minVal) {
            minVal = v;
            ctx = i;
        }
    }
    
    dst_vals[pos] = minVal;
    dst_ctx[pos] = ctx;
}

kernel void vMax_Reduce_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    constant int &reduced_axis [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    float maxVal = -INFINITY;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        auto v = src_vals[i * stride + offset];
        if (v > maxVal) {
            maxVal = v;
        }
    }
    
    dst_vals[pos] = maxVal;
}

kernel void vMin_Reduce_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    const device int* dst_strides [[buffer(7)]],
    constant int &reduced_axis [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    float minVal = INFINITY;
    
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    uint count = src_shape[reduced_axis];
    
    for (uint i = 0; i < count; i++) {
        auto v = src_vals[i * stride + offset];
        if (v < minVal) {
            minVal = v;
        }
    }
    
    dst_vals[pos] = minVal;
}

kernel void vScatter_Float32(
    const device float* src_vals [[buffer(0)]],
    device float* dst_vals [[buffer(3)]],
    constant int &dst_dim [[buffer(4)]],
    const device int* dst_shape [[buffer(5)]],
    const device int* dst_strides [[buffer(6)]],
    constant int &scatter_axis [[buffer(7)]],
    const device int* indices [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    uint posBefore = pos / dst_strides[scatter_axis];
    uint posAfter = pos % dst_strides[scatter_axis];
    
    uint offset = posAfter + (scatter_axis != 0 ? posBefore * dst_strides[scatter_axis - 1] : 0);
    uint stride = dst_strides[scatter_axis];
    
    auto i = indices[pos];
    dst_vals[i * stride + offset] = src_vals[pos];
}


kernel void vGather_Float32(
    const device float* src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    const device int* src_strides [[buffer(3)]],
    device float* dst_vals [[buffer(4)]],
    constant int &dst_dim [[buffer(5)]],
    const device int* dst_shape [[buffer(6)]],
    constant int &reduced_axis [[buffer(7)]],
    const device int* indices [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    uint posBefore = pos / src_strides[reduced_axis];
    uint posAfter = pos % src_strides[reduced_axis];
    
    uint offset = posAfter + (reduced_axis != 0 ? posBefore * src_strides[reduced_axis - 1] : 0);
    uint stride = src_strides[reduced_axis];
    
    auto i = indices[pos];
    dst_vals[pos] = src_vals[i * stride + offset];
}

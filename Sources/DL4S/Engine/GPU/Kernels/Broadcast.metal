//
//  Broadcast.metal
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

template <typename element_t>
class Tensor {
private:
    const device element_t* elements;
    
public:
    const Shape shape;
    Tensor(Shape s, const device element_t* el) : shape(s), elements(el) {}
    element_t get(int idx) const;
};

template <typename element_t>
class MutableTensor {
private:
    device element_t* elements;
    
public:
    const Shape shape;
    MutableTensor(Shape s, device element_t* el) : shape(s), elements(el) {}
    
    element_t get(int idx) const;
    void set(int idx, element_t value);
};
template <typename element_t>
element_t Tensor<element_t>::get(int idx) const {
    return this->elements[idx];
}

template <typename element_t>
element_t MutableTensor<element_t>::get(int idx) const {
    return this->elements[idx];
}

template <typename element_t>
void MutableTensor<element_t>::set(int idx, element_t value) {
    this->elements[idx] = value;
}

kernel void vAdd_Broadcast_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant int &lhs_dim [[buffer(1)]],
    const device int* lhs_shape [[buffer(2)]],
    const device float* rhs_vals [[buffer(3)]],
    constant int &rhs_dim [[buffer(4)]],
    const device int* rhs_shape [[buffer(5)]],
    device float* result_vals [[buffer(6)]],
    constant int &result_dim [[buffer(7)]],
    const device int* result_shape [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    auto lhs_sh = Shape(lhs_dim, lhs_shape);
    auto rhs_sh = Shape(rhs_dim, rhs_shape);
    auto result_sh = Shape(result_dim, result_shape);
    
    auto lhs = Tensor<float>(lhs_sh, lhs_vals);
    auto rhs = Tensor<float>(rhs_sh, rhs_vals);
    auto result = MutableTensor<float>(lhs.shape, result_vals);
    
    result.set(pos, lhs.get(result_sh.translate(pos, lhs_sh)) + rhs.get(result_sh.translate(pos, rhs_sh)));
}


kernel void vSub_Broadcast_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant int &lhs_dim [[buffer(1)]],
    const device int* lhs_shape [[buffer(2)]],
    const device float* rhs_vals [[buffer(3)]],
    constant int &rhs_dim [[buffer(4)]],
    const device int* rhs_shape [[buffer(5)]],
    device float* result_vals [[buffer(6)]],
    constant int &result_dim [[buffer(7)]],
    const device int* result_shape [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    auto lhs_sh = Shape(lhs_dim, lhs_shape);
    auto rhs_sh = Shape(rhs_dim, rhs_shape);
    auto result_sh = Shape(result_dim, result_shape);
    
    auto lhs = Tensor<float>(lhs_sh, lhs_vals);
    auto rhs = Tensor<float>(rhs_sh, rhs_vals);
    auto result = MutableTensor<float>(lhs.shape, result_vals);
    
    result.set(pos, lhs.get(result_sh.translate(pos, lhs_sh)) - rhs.get(result_sh.translate(pos, rhs_sh)));
}

kernel void vMul_Broadcast_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant int &lhs_dim [[buffer(1)]],
    const device int* lhs_shape [[buffer(2)]],
    const device float* rhs_vals [[buffer(3)]],
    constant int &rhs_dim [[buffer(4)]],
    const device int* rhs_shape [[buffer(5)]],
    device float* result_vals [[buffer(6)]],
    constant int &result_dim [[buffer(7)]],
    const device int* result_shape [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    auto lhs_sh = Shape(lhs_dim, lhs_shape);
    auto rhs_sh = Shape(rhs_dim, rhs_shape);
    auto result_sh = Shape(result_dim, result_shape);
    
    auto lhs = Tensor<float>(lhs_sh, lhs_vals);
    auto rhs = Tensor<float>(rhs_sh, rhs_vals);
    auto result = MutableTensor<float>(lhs.shape, result_vals);
    
    result.set(pos, lhs.get(result_sh.translate(pos, lhs_sh)) * rhs.get(result_sh.translate(pos, rhs_sh)));
}

kernel void vDiv_Broadcast_Float32(
    const device float* lhs_vals [[buffer(0)]],
    constant int &lhs_dim [[buffer(1)]],
    const device int* lhs_shape [[buffer(2)]],
    const device float* rhs_vals [[buffer(3)]],
    constant int &rhs_dim [[buffer(4)]],
    const device int* rhs_shape [[buffer(5)]],
    device float* result_vals [[buffer(6)]],
    constant int &result_dim [[buffer(7)]],
    const device int* result_shape [[buffer(8)]],
    uint pos [[thread_position_in_grid]]
) {
    auto lhs_sh = Shape(lhs_dim, lhs_shape);
    auto rhs_sh = Shape(rhs_dim, rhs_shape);
    auto result_sh = Shape(result_dim, result_shape);
    
    auto lhs = Tensor<float>(lhs_sh, lhs_vals);
    auto rhs = Tensor<float>(rhs_sh, rhs_vals);
    auto result = MutableTensor<float>(lhs.shape, result_vals);
    
    result.set(pos, lhs.get(result_sh.translate(pos, lhs_sh)) / rhs.get(result_sh.translate(pos, rhs_sh)));
}

kernel void subscriptRead_Float32(
    const device float *src_vals [[buffer(0)]],
    constant int &src_dim [[buffer(1)]],
    const device int* src_shape [[buffer(2)]],
    device float *dst_vals [[buffer(3)]],
    constant int &dst_dim [[buffer(4)]],
    const device int *dst_shape [[buffer(5)]],
    // constant int &idx_dim [[buffer(6)]],
    const device int *idx [[buffer(7)]],
    uint pos [[thread_position_in_grid]]
) {
    auto src_sh = Shape(dst_dim, src_shape);
    auto dst_sh = Shape(dst_dim, dst_shape);
    
    auto src = Tensor<float>(src_sh, src_vals);
    auto dst = MutableTensor<float>(dst_sh, dst_vals);
    
    // idx: integers, either -1 (copy axis) or ≥0 (copy value)
    // pos: position in dst
    uint src_pos = 0;
    
    
    
    dst.set(pos, src.get(src_pos));
}

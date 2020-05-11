let SHADER_SOURCE = """
//
//  Shape.h
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

#ifndef Shape_h
#define Shape_h

using namespace metal;

class Shape {
public:
    const int dim;
    const device int* shape;
    
    Shape(const int d, const device int* s): dim(d), shape(s) {}
    
    const int getCount() const;
    const int operator[](int index) const;
    const int broadcastIndex(int globalIndex) const;
    const int translate(int globalIndex, Shape subshape) const;
    const int permute(int index, device const int* strides, device const int* arangement) const;
    const int indexWithInsertedAxes(const int index, device const int* inserted_axes, const Shape dstShape) const;
};


#endif /* Shape_h */
//
//  Tensor.h
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

#ifndef Tensor_h
#define Tensor_h
using namespace metal;



#endif /* Tensor_h */
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
//
//  Shape.metal
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


const int Shape::getCount() const {
    int count = 1;
    for (int axis = 0; axis < this->dim; axis++) {
        count *= shape[axis];
    }
    return count;
}

const int Shape::operator[](int index) const {
    return this->shape[index];
}

const int Shape::broadcastIndex(int globalIndex) const {
    return globalIndex % this->getCount();
}

const int Shape::translate(int globalIndex, Shape subshape) const {
    int broadcastIdx = globalIndex;
    
    int dimOffset = this->dim - subshape.dim;
    int srcIdx = 0;
    int srcStride = 1;
    
    for (int i = subshape.dim - 1; i >= 0; i--) {
        int srcDim = subshape.shape[i];
        int dstDim = this->shape[i + dimOffset];
        
        if (srcDim != 1) {
            srcIdx += (broadcastIdx % dstDim) * srcStride;
        }
        broadcastIdx = broadcastIdx / dstDim;
        srcStride *= srcDim;
    }
    
    return srcIdx;
}

const int Shape::permute(int index, const device int *strides, const device int *arangement) const {
    int dstIdx = 0;
    for (int i = this->dim - 1; i >= 0; i--) {
        int dimSize = this->shape[i];
        int axisIdx = index % dimSize;
        index /= dimSize;
        dstIdx += axisIdx * strides[arangement[i]];
    }
    return dstIdx;
}

const int Shape::indexWithInsertedAxes(const int index, device const int* inserted_axes, const Shape dstShape) const {
    int dstIdx = 0;
    int src_i = 0;
    int sax_i = 0;
    int stride = 1;
    int idx = index;
    for (int i = 0; i < dstShape.dim; i++) {
        if (inserted_axes[sax_i] == i) {
            stride *= dstShape.shape[i];
        } else {
            dstIdx += stride * (idx % this->shape[src_i]);
            idx /= this->shape[src_i];
            stride *= dstShape.shape[i];
            src_i++;
        }
    }
    return dstIdx;
}
//
//  Tensor.metal
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
"""

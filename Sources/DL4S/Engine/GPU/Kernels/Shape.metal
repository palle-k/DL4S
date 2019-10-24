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

#include "Shape.hpp"

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
            dstIdx += stride;
        } else {
            dstIdx += stride * (idx % this->shape[src_i]);
            idx /= this->shape[src_i];
            stride *= dstShape.shape[i];
            src_i++;
        }
    }
    return dstIdx;
}

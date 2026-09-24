//
//  prelude.metal
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

// Declarations that all kernel groups share. The library prepends this file to every group before it compiles the group.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Shape and strides of a strided region with up to eight axes. Strides are in elements and can be negative.
struct Layout {
    int dim;
    int shape[8];
    int strides[3][8];
};

// Offsets of the element with the given linear index in the row-major order of the shape, for up to three operands.
inline int3 layout_offsets(constant Layout& layout, uint index) {
    int3 offsets = int3(0);
    uint rest = index;
    for (int axis = layout.dim - 1; axis >= 0; axis--) {
        uint size = uint(layout.shape[axis]);
        int coordinate = int(rest % size);
        rest /= size;
        offsets += coordinate * int3(layout.strides[0][axis], layout.strides[1][axis], layout.strides[2][axis]);
    }
    return offsets;
}

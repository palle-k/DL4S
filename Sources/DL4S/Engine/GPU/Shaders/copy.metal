//
//  copy.metal
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

// Kernels that move elements: strided copies, gather and scatter along an axis, and the window matrix of a convolution.
//
// Kernels that only move elements work on 32-bit words, so they serve all element types with GPU kernels.

// Copies a strided region. Operand 0 of the layout is the source, operand 1 the result.
kernel void strided_copy_u32(device const uint* source [[buffer(0)]], device uint* result [[buffer(1)]], constant Layout& layout [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) { return; }
    int3 offsets = layout_offsets(layout, i);
    result[offsets.y] = source[offsets.x];
}

// Copies a strided region and adds a second one. Operand 2 of the layout is the summand.
// Every element is read before it is written, so the summand can be the result.
#define STRIDED_ADD(T) \
kernel void strided_copy_add_##T(device const T* source [[buffer(0)]], device T* result [[buffer(1)]], device const T* summand [[buffer(2)]], constant Layout& layout [[buffer(3)]], constant uint& count [[buffer(4)]], uint i [[thread_position_in_grid]]) { \
    if (i >= count) { return; } \
    int3 offsets = layout_offsets(layout, i); \
    T value = source[offsets.x] + summand[offsets.z]; \
    result[offsets.y] = value; \
}

STRIDED_ADD(float)
STRIDED_ADD(int)

// Transposes the last two axes of a batch of matrices through threadgroup memory, so that reads and writes are contiguous.
// The source matrices have `rows` rows and `columns` columns.
kernel void transpose_u32(device const uint* source [[buffer(0)]], device uint* result [[buffer(1)]], constant uint2& size [[buffer(2)]],
                          uint3 group [[threadgroup_position_in_grid]], uint3 local [[thread_position_in_threadgroup]]) {
    threadgroup uint tile[32][33];
    uint rows = size.x, columns = size.y;
    ulong matrix = ulong(group.z) * rows * columns;
    uint row0 = group.y * 32, column0 = group.x * 32;
    for (uint r = local.y; r < 32; r += 8) {
        uint row = row0 + r, column = column0 + local.x;
        if (row < rows && column < columns) { tile[r][local.x] = source[matrix + ulong(row) * columns + column]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = local.y; c < 32; c += 8) {
        uint column = column0 + c, row = row0 + local.x;
        if (row < rows && column < columns) { result[matrix + ulong(column) * rows + row] = tile[local.x][c]; }
    }
}

// Gathers along an axis. The source has the shape [outer, axisSize, inner], the indices and the result have the shape [outer, inner].
kernel void gather_u32(device const uint* source [[buffer(0)]], device const int* indices [[buffer(1)]], device uint* result [[buffer(2)]], constant int4& sizes [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    int count = sizes.x, axisSize = sizes.y, inner = sizes.z, ignoreIndex = sizes.w;
    if (int(i) >= count) { return; }
    int index = indices[i];
    if (index == ignoreIndex) { result[i] = 0; return; }
    int outer = int(i) / inner, position = int(i) % inner;
    result[i] = source[(outer * axisSize + index) * inner + position];
}

// Scatters along an axis into a result that is filled with zeros. The result has the shape [outer, axisSize, inner].
kernel void scatter_u32(device const uint* values [[buffer(0)]], device const int* indices [[buffer(1)]], device uint* result [[buffer(2)]], constant int4& sizes [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    int count = sizes.x, axisSize = sizes.y, inner = sizes.z, ignoreIndex = sizes.w;
    if (int(i) >= count) { return; }
    int index = indices[i];
    if (index == ignoreIndex) { return; }
    int outer = int(i) / inner, position = int(i) % inner;
    result[(outer * axisSize + index) * inner + position] = values[i];
}

struct WindowGeometry {
    int batchSize, channels, height, width;
    int kernelHeight, kernelWidth, padding, stride;
    int outputHeight, outputWidth;
};

// Writes the window matrix of a convolution. Row (channel, kernelRow, kernelColumn), column (image, outputRow, outputColumn).
kernel void img2col_u32(device const uint* image [[buffer(0)]], device uint* result [[buffer(1)]], constant WindowGeometry& g [[buffer(2)]], uint2 position [[thread_position_in_grid]]) {
    int windows = g.outputHeight * g.outputWidth;
    int columns = g.batchSize * windows;
    int column = int(position.x), row = int(position.y);
    if (column >= columns || row >= g.channels * g.kernelHeight * g.kernelWidth) { return; }
    int kernelColumn = row % g.kernelWidth;
    int kernelRow = (row / g.kernelWidth) % g.kernelHeight;
    int channel = row / (g.kernelWidth * g.kernelHeight);
    int imageIndex = column / windows;
    int window = column % windows;
    int y = (window / g.outputWidth) * g.stride - g.padding + kernelRow;
    int x = (window % g.outputWidth) * g.stride - g.padding + kernelColumn;
    uint value = 0;
    if (y >= 0 && y < g.height && x >= 0 && x < g.width) {
        value = image[((imageIndex * g.channels + channel) * g.height + y) * g.width + x];
    }
    result[ulong(row) * columns + column] = value;
}

// Adds the windows of a window matrix to the image elements that they cover. One thread computes one image element,
// so no two threads write the same element.
kernel void col2img_float(device const float* matrix [[buffer(0)]], device float* image [[buffer(1)]], constant WindowGeometry& g [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    int elements = g.batchSize * g.channels * g.height * g.width;
    if (int(i) >= elements) { return; }
    int x = int(i) % g.width;
    int y = (int(i) / g.width) % g.height;
    int channel = (int(i) / (g.width * g.height)) % g.channels;
    int imageIndex = int(i) / (g.width * g.height * g.channels);
    int windows = g.outputHeight * g.outputWidth;
    int columns = g.batchSize * windows;
    float sum = 0;
    for (int kernelRow = 0; kernelRow < g.kernelHeight; kernelRow++) {
        int shiftedY = y + g.padding - kernelRow;
        if (shiftedY < 0 || shiftedY % g.stride != 0) { continue; }
        int outputRow = shiftedY / g.stride;
        if (outputRow >= g.outputHeight) { continue; }
        for (int kernelColumn = 0; kernelColumn < g.kernelWidth; kernelColumn++) {
            int shiftedX = x + g.padding - kernelColumn;
            if (shiftedX < 0 || shiftedX % g.stride != 0) { continue; }
            int outputColumn = shiftedX / g.stride;
            if (outputColumn >= g.outputWidth) { continue; }
            int row = (channel * g.kernelHeight + kernelRow) * g.kernelWidth + kernelColumn;
            sum += matrix[ulong(row) * columns + imageIndex * windows + outputRow * g.outputWidth + outputColumn];
        }
    }
    image[i] = sum;
}

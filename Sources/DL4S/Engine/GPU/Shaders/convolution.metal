//
//  convolution.metal
//  DL4S
//
//  Created by Palle Klewitz on 29.09.26.
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

// Convolutions as implicit matrix products: C[m, n] = sum over k of A[m, k] * B[k, n], where the threadgroups gather the
// tiles of B from the images when they load them, so no window matrix is written to memory.
//
// A threadgroup of four or eight SIMD groups computes a 64 x 64 or 128 x 64 tile of C: every SIMD group computes 32 x 32
// elements, as in the matrix kernels. The taller tile uses every loaded element of B for twice as many products, which
// matters because the elements of B are gathered one at a time. The columns n enumerate the
// pixels of a grid of `heightN` x `widthN` per image. The rows of A are contiguous. A row k of B has an entry in a table: the
// offset of its element from the element of a column, and the displacement of its position in the image, which decides
// whether it lies in the padding.
// Every thread loads the same column of every tile of B, so it decomposes its column once.
//
// The forward pass uses A = filters, [outputChannels, inputChannels * kernelHeight * kernelWidth], and B = the windows of
// the input. The data gradient of a strided convolution runs once for every phase of the stride: the pixels of the input
// image whose position modulo the stride is the same use the same filter taps, so B contains only taps that reach them.

struct ImplicitGemmParameters {
    int M, N, K;
    int aRowStride;
    int heightN, widthN;
    // Position of the element of column (b, uy, ux) in the image of B: y = uy * bStrideY + bOffsetY, x = ux * bStrideX + bOffsetX
    int bBatch, bStrideY, bOffsetY, bStrideX, bOffsetX, bHeight, bWidth;
    // Element of row m and column (b, uy, ux) of C: m * cRowStride + b * cBatch + (uy * cStrideY + cOffsetY) * cWidth + ux * cStrideX + cOffsetX
    int cRowStride, cBatch, cStrideY, cOffsetY, cStrideX, cOffsetX, cWidth;
    int accumulate, hasBias;
};

constant constexpr int TILE = 64;
constant constexpr int BK = 32;
constant constexpr int BLOCKS = 4;
constant constexpr int A_STRIDE = BK + 4;
constant constexpr int B_STRIDE = TILE + 4;

// TM is the height of the tile, 64 or 128, with TM * 2 threads.
template <int TM>
kernel void implicit_gemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
                          device const int4* bTable [[buffer(3)]], device const float* bias [[buffer(4)]], constant ImplicitGemmParameters& p [[buffer(5)]],
                          uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int THREADS = TM * 2;
    threadgroup float As[TM * A_STRIDE];
    threadgroup float Bs[BK * B_STRIDE];
    // The table entries of the rows of the current tile, loaded once per tile instead of once per element.
    threadgroup int4 rowEntries[BK];
    const int m0 = int(group.y) * TM, n0 = int(group.x) * TILE;
    const int sm = int(simd / 2) * 32, sn = int(simd % 2) * 32;
    const int pixels = p.heightN * p.widthN;

    // The column of B that this thread loads, decomposed once.
    const int bColumn = int(thread_index) % TILE;
    const int n = n0 + bColumn;
    const bool nValid = n < p.N;
    const int b = nValid ? n / pixels : 0;
    const int pixel = n - b * pixels;
    const int uy = pixel / p.widthN, ux = pixel - uy * p.widthN;
    const int ny = uy * p.bStrideY + p.bOffsetY, nx = ux * p.bStrideX + p.bOffsetX;
    const long bBase = long(b) * p.bBatch + long(ny) * p.bWidth + nx;
    // The column of A that this thread loads.
    const int aColumn = int(thread_index) % BK;

    simdgroup_float8x8 accumulators[BLOCKS][BLOCKS];
    UNROLL for (int i = 0; i < BLOCKS; i++) {
        UNROLL for (int j = 0; j < BLOCKS; j++) {
            accumulators[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = 0; k0 < p.K; k0 += BK) {
        if (thread_index < uint(BK)) {
            int k = k0 + int(thread_index);
            rowEntries[thread_index] = k < p.K ? bTable[k] : int4(0, INT_MIN / 2, 0, 0);
        }
        if (m0 + TM <= p.M && k0 + BK <= p.K) {
            // Tiles in the interior of A: every thread loads four elements at a time.
            UNROLL for (int e = int(thread_index) * 4; e < TM * BK; e += THREADS * 4) {
                int r = e / BK, c = e % BK;
                float4 v = float4(*(device const packed_float4*)(A + long(m0 + r) * p.aRowStride + k0 + c));
                threadgroup float* target = As + r * A_STRIDE + c;
                target[0] = v.x; target[1] = v.y; target[2] = v.z; target[3] = v.w;
            }
        } else {
            const int ka = k0 + aColumn;
            UNROLL for (int step = 0; step < TM * BK / THREADS; step++) {
                int r = int(thread_index) / BK + step * (THREADS / BK);
                int m = m0 + r;
                As[r * A_STRIDE + aColumn] = m < p.M && ka < p.K ? A[long(m) * p.aRowStride + ka] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (int step = 0; step < TILE * BK / THREADS; step++) {
            int r = int(thread_index) / TILE + step * (THREADS / TILE);
            // Rows after the end of K have a displacement that puts every element into the padding.
            int4 entry = rowEntries[r];
            int y = ny + entry.y, x = nx + entry.z;
            bool inside = nValid && y >= 0 && y < p.bHeight && x >= 0 && x < p.bWidth;
            Bs[r * B_STRIDE + bColumn] = inside ? B[bBase + entry.x] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[BLOCKS], bm[BLOCKS];
            UNROLL for (int i = 0; i < BLOCKS; i++) {
                simdgroup_load(a[i], As + (sm + i * 8) * A_STRIDE + kk, A_STRIDE);
            }
            UNROLL for (int j = 0; j < BLOCKS; j++) {
                simdgroup_load(bm[j], Bs + kk * B_STRIDE + sn + j * 8, B_STRIDE);
            }
            UNROLL for (int i = 0; i < BLOCKS; i++) {
                UNROLL for (int j = 0; j < BLOCKS; j++) {
                    simdgroup_multiply_accumulate(accumulators[i][j], a[i], bm[j], accumulators[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Every thread owns two neighboring elements of each 8 x 8 block, in row `blockRow` and columns `blockColumn` and `blockColumn + 1`.
    const int quad = int(lane) / 4;
    const int blockRow = (quad & 4) + (int(lane) / 2) % 4;
    const int blockColumn = (quad & 2) * 2 + (int(lane) % 2) * 2;
    UNROLL for (int j = 0; j < BLOCKS; j++) {
        UNROLL for (int e = 0; e < 2; e++) {
            int column = n0 + sn + j * 8 + blockColumn + e;
            if (column >= p.N) { continue; }
            int cb = column / pixels;
            int cp = column - cb * pixels;
            int cy = cp / p.widthN, cx = cp - cy * p.widthN;
            long columnOffset = long(cb) * p.cBatch + long(cy * p.cStrideY + p.cOffsetY) * p.cWidth + cx * p.cStrideX + p.cOffsetX;
            UNROLL for (int i = 0; i < BLOCKS; i++) {
                int row = m0 + sm + i * 8 + blockRow;
                if (row >= p.M) { continue; }
                device float* target = C + long(row) * p.cRowStride + columnOffset;
                float value = accumulators[i][j].thread_elements()[e] + (p.hasBias ? bias[row] : 0.0f);
                *target = p.accumulate ? *target + value : value;
            }
        }
    }
}

template [[host_name("implicit_gemm_64")]] kernel void implicit_gemm<64>(device const float*, device const float*, device float*, device const int4*, device const float*, constant ImplicitGemmParameters&, uint3, uint, uint, uint);
template [[host_name("implicit_gemm_128")]] kernel void implicit_gemm<128>(device const float*, device const float*, device float*, device const int4*, device const float*, constant ImplicitGemmParameters&, uint3, uint, uint, uint);

// Writes the rows of a matrix that the tables of the implicit products describe as a contiguous matrix:
// result[m, k] = source[m * rowStride + table[k]]. The data gradient uses it for the filters of every phase of the stride.
kernel void gather_columns(device const float* source [[buffer(0)]], device const int* table [[buffer(1)]], device float* result [[buffer(2)]],
                           constant int3& p [[buffer(3)]], uint2 position [[thread_position_in_grid]]) {
    // p: rows, columns, row stride of the source
    if (int(position.x) >= p.y || int(position.y) >= p.x) { return; }
    result[position.y * uint(p.y) + position.x] = source[long(position.y) * p.z + table[position.x]];
}

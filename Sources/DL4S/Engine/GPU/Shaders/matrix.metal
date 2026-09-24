//
//  matrix.metal
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

// Matrix multiplication with SIMD group matrices, for a batch of matrices with constant strides between them.
//
// A threadgroup of four SIMD groups in a 2 x 2 arrangement computes a tile of TILE x TILE elements of the result,
// where TILE is 64 or 32. Every SIMD group computes its quarter of the tile as blocks of 8 x 8 elements.
// The threadgroup loads tiles of TILE x 32 elements of both operands into threadgroup memory, in the layout in which
// they are stored, and a transposed operand is transposed when the SIMD groups load their blocks.
// Tiles in the interior of the matrices are loaded with vector loads and no bounds checks.
//
// The loops over the blocks are unrolled: when the compiler keeps them as loops, the arrays of SIMD group matrices
// are placed in memory instead of registers, and the kernel reaches less than a fifth of its throughput.
//
// The kernel for a single row computes a vector-matrix product, which is limited by the bandwidth of the reads of the matrix.

struct GemmParameters {
    int M, N, K;
    int lda, ldb, ldc;
    int transposeA, transposeB;
    float alpha, beta;
    long batchStrideA, batchStrideB, batchStrideC;
    // Number of parts of the inner axis and the length of a part. With more than one part, the threadgroups along z
    // compute the products of the parts, and matrix z of C receives the product of part z % splits of batch z / splits.
    int splits, splitLength;
};

#define UNROLL _Pragma("clang loop unroll(full)")

constant constexpr int BK = 32;

template <bool TA, bool TB, int TILE>
kernel void gemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmParameters& p [[buffer(3)]],
                 uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    // Blocks of 8 x 8 elements per SIMD group along each axis.
    constexpr int BLOCKS = TILE / 16;
    // Layout of the tiles in threadgroup memory: A as TILE x BK (or BK x TILE when transposed), B as BK x TILE (or TILE x BK).
    // The padding spreads the rows over the memory banks.
    constexpr int A_ROWS = TA ? BK : TILE;
    constexpr int A_COLUMNS = TA ? TILE : BK;
    constexpr int B_ROWS = TB ? TILE : BK;
    constexpr int B_COLUMNS = TB ? BK : TILE;
    constexpr int A_STRIDE = A_COLUMNS + 4;
    constexpr int B_STRIDE = B_COLUMNS + 4;
    threadgroup float As[A_ROWS * A_STRIDE];
    threadgroup float Bs[B_ROWS * B_STRIDE];

    const int batch = int(group.z) / p.splits;
    const int kBegin = (int(group.z) % p.splits) * p.splitLength;
    const int kEnd = min(p.K, kBegin + p.splitLength);
    A += long(batch) * p.batchStrideA;
    B += long(batch) * p.batchStrideB;
    C += long(group.z) * p.batchStrideC;
    const int m0 = int(group.y) * TILE, n0 = int(group.x) * TILE;
    const int sm = int(simd / 2) * (TILE / 2), sn = int(simd % 2) * (TILE / 2);
    // Rows and columns of the tile in the stored layout of the operands.
    const int aRow0 = TA ? 0 : m0, aColumn0 = TA ? m0 : 0;
    const int bRow0 = TB ? n0 : 0, bColumn0 = TB ? 0 : n0;
    const int aRowLimit = TA ? kEnd : p.M, aColumnLimit = TA ? p.M : kEnd;
    const int bRowLimit = TB ? p.N : kEnd, bColumnLimit = TB ? kEnd : p.N;
    const bool interior = m0 + TILE <= p.M && n0 + TILE <= p.N;

    simdgroup_float8x8 accumulators[BLOCKS][BLOCKS];
    UNROLL for (int i = 0; i < BLOCKS; i++) {
        UNROLL for (int j = 0; j < BLOCKS; j++) {
            accumulators[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (int k0 = kBegin; k0 < kEnd; k0 += BK) {
        const int aRowStart = aRow0 + (TA ? k0 : 0), aColumnStart = aColumn0 + (TA ? 0 : k0);
        const int bRowStart = bRow0 + (TB ? 0 : k0), bColumnStart = bColumn0 + (TB ? k0 : 0);
        if (interior && k0 + BK <= kEnd) {
            // Every thread loads four elements at a time. The element counts of both tiles are multiples of 4 * 128.
            UNROLL for (int e = int(thread_index) * 4; e < A_ROWS * A_COLUMNS; e += 128 * 4) {
                int r = e / A_COLUMNS, c = e % A_COLUMNS;
                float4 v = float4(*(device const packed_float4*)(A + long(aRowStart + r) * p.lda + aColumnStart + c));
                threadgroup float* target = As + r * A_STRIDE + c;
                target[0] = v.x; target[1] = v.y; target[2] = v.z; target[3] = v.w;
            }
            UNROLL for (int e = int(thread_index) * 4; e < B_ROWS * B_COLUMNS; e += 128 * 4) {
                int r = e / B_COLUMNS, c = e % B_COLUMNS;
                float4 v = float4(*(device const packed_float4*)(B + long(bRowStart + r) * p.ldb + bColumnStart + c));
                threadgroup float* target = Bs + r * B_STRIDE + c;
                target[0] = v.x; target[1] = v.y; target[2] = v.z; target[3] = v.w;
            }
        } else {
            for (int e = int(thread_index); e < A_ROWS * A_COLUMNS; e += 128) {
                int r = e / A_COLUMNS, c = e % A_COLUMNS;
                int row = aRowStart + r, column = aColumnStart + c;
                As[r * A_STRIDE + c] = row < aRowLimit && column < aColumnLimit ? A[long(row) * p.lda + column] : 0.0f;
            }
            for (int e = int(thread_index); e < B_ROWS * B_COLUMNS; e += 128) {
                int r = e / B_COLUMNS, c = e % B_COLUMNS;
                int row = bRowStart + r, column = bColumnStart + c;
                Bs[r * B_STRIDE + c] = row < bRowLimit && column < bColumnLimit ? B[long(row) * p.ldb + column] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        UNROLL for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[BLOCKS], b[BLOCKS];
            UNROLL for (int i = 0; i < BLOCKS; i++) {
                if (TA) {
                    simdgroup_load(a[i], As + kk * A_STRIDE + sm + i * 8, A_STRIDE, 0, true);
                } else {
                    simdgroup_load(a[i], As + (sm + i * 8) * A_STRIDE + kk, A_STRIDE);
                }
            }
            UNROLL for (int j = 0; j < BLOCKS; j++) {
                if (TB) {
                    simdgroup_load(b[j], Bs + (sn + j * 8) * B_STRIDE + kk, B_STRIDE, 0, true);
                } else {
                    simdgroup_load(b[j], Bs + kk * B_STRIDE + sn + j * 8, B_STRIDE);
                }
            }
            UNROLL for (int i = 0; i < BLOCKS; i++) {
                UNROLL for (int j = 0; j < BLOCKS; j++) {
                    simdgroup_multiply_accumulate(accumulators[i][j], a[i], b[j], accumulators[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (interior && p.alpha == 1.0f && p.beta == 0.0f) {
        UNROLL for (int i = 0; i < BLOCKS; i++) {
            UNROLL for (int j = 0; j < BLOCKS; j++) {
                simdgroup_store(accumulators[i][j], C + long(m0 + sm + i * 8) * p.ldc + n0 + sn + j * 8, p.ldc);
            }
        }
        return;
    }
    // Every thread owns two neighboring elements of each 8 x 8 block, in row `blockRow` and columns `blockColumn` and `blockColumn + 1`.
    const int quad = int(lane) / 4;
    const int blockRow = (quad & 4) + (int(lane) / 2) % 4;
    const int blockColumn = (quad & 2) * 2 + (int(lane) % 2) * 2;
    UNROLL for (int i = 0; i < BLOCKS; i++) {
        UNROLL for (int j = 0; j < BLOCKS; j++) {
            thread auto& elements = accumulators[i][j].thread_elements();
            int row = m0 + sm + i * 8 + blockRow;
            UNROLL for (int e = 0; e < 2; e++) {
                int column = n0 + sn + j * 8 + blockColumn + e;
                if (row < p.M && column < p.N) {
                    device float* target = C + long(row) * p.ldc + column;
                    float value = p.alpha * elements[e];
                    if (p.beta != 0.0f) {
                        value += p.beta * *target;
                    }
                    *target = value;
                }
            }
        }
    }
}

#define GEMM(TA, TB, TILE, NAME) \
template [[host_name(NAME)]] kernel void gemm<TA, TB, TILE>(device const float*, device const float*, device float*, constant GemmParameters&, uint3, uint, uint, uint);

GEMM(false, false, 64, "gemm_nn_64")
GEMM(false, true, 64, "gemm_nt_64")
GEMM(true, false, 64, "gemm_tn_64")
GEMM(true, true, 64, "gemm_tt_64")
GEMM(false, false, 32, "gemm_nn_32")
GEMM(false, true, 32, "gemm_nt_32")
GEMM(true, false, 32, "gemm_tn_32")
GEMM(true, true, 32, "gemm_tt_32")

// Adds the products of the parts of the inner axis: C = alpha * sum of the parts + beta * C.
// The parts of batch b are the matrices b * splits ..< (b + 1) * splits of the partial products.
kernel void gemm_split_sum(device const float* partial [[buffer(0)]], device float* C [[buffer(1)]], constant GemmParameters& p [[buffer(2)]],
                           uint3 position [[thread_position_in_grid]]) {
    int column = int(position.x), row = int(position.y), batch = int(position.z);
    if (column >= p.N || row >= p.M) { return; }
    long matrix = long(p.M) * p.N;
    device const float* parts = partial + long(batch) * p.splits * matrix + long(row) * p.N + column;
    float sum = 0.0f;
    for (int part = 0; part < p.splits; part++) {
        sum += parts[long(part) * matrix];
    }
    device float* target = C + long(batch) * p.batchStrideC + long(row) * p.ldc + column;
    float value = p.alpha * sum;
    if (p.beta != 0.0f) { value += p.beta * *target; }
    *target = value;
}

// Vector-matrix product for results with one row: every thread computes one element of the result.
// For a matrix that is not transposed, neighboring threads read neighboring elements of a row of B.
// For a transposed matrix, a SIMD group computes one element as the dot product of a row of B with the vector.
kernel void gemv_n(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmParameters& p [[buffer(3)]],
                   uint2 position [[thread_position_in_grid]]) {
    int column = int(position.x);
    if (column >= p.N) { return; }
    A += long(position.y) * p.batchStrideA;
    B += long(position.y) * p.batchStrideB;
    C += long(position.y) * p.batchStrideC;
    // The row vector is A with the stride 1, or the first column of a transposed A with the stride lda.
    int aStride = p.transposeA ? p.lda : 1;
    float sum = 0.0f;
    for (int k = 0; k < p.K; k++) {
        sum = fma(A[long(k) * aStride], B[long(k) * p.ldb + column], sum);
    }
    float value = p.alpha * sum;
    if (p.beta != 0.0f) { value += p.beta * C[column]; }
    C[column] = value;
}

kernel void gemv_t(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]], constant GemmParameters& p [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    int column = int(group.x) * 8 + int(simd);
    if (column >= p.N) { return; }
    A += long(group.y) * p.batchStrideA;
    B += long(group.y) * p.batchStrideB;
    C += long(group.y) * p.batchStrideC;
    int aStride = p.transposeA ? p.lda : 1;
    device const float* row = B + long(column) * p.ldb;
    float sum = 0.0f;
    for (int k = int(lane); k < p.K; k += 32) {
        sum = fma(A[long(k) * aStride], row[k], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        float value = p.alpha * sum;
        if (p.beta != 0.0f) { value += p.beta * C[column]; }
        C[column] = value;
    }
}

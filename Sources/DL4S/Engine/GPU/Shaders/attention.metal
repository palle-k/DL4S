//
//  attention.metal
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

// Scaled dot product attention without the matrix of the scores in memory, after FlashAttention-2
// (https://arxiv.org/abs/2307.08691).
//
// The forward kernel gives every SIMD group R rows of queries and loops over blocks of BC keys, which the threadgroup
// loads into threadgroup memory. The SIMD group keeps its queries, its scores, and its accumulated result in SIMD group
// matrices, and keeps the softmax online: it tracks the largest score of every row and the sum of the exponentials.
// The scores are in the base 2 domain, `log2(e) * (q · k / temperature - 10⁹ * mask)`, so the exponentials are exp2.
//
// The result is rescaled only when the largest score of a row grows by more than RESCALE_THRESHOLD, after the lazy
// rescaling of FlashAttention-4. Until then the exponentials use the old maximum and are at most
// 2^RESCALE_THRESHOLD, so no float overflows. The final division uses the same maximum, so the result is exact.
//
// The backward pass computes the result again with the forward kernel, which also writes the largest score m and the
// inverse sum of the exponentials 1 / l of every row, and D = rowsum(outputGradient * result). The weights are then
// exp2(score - m) / l. A log-sum-exp m + log2(l) would lose log2(l) in a row in which every score is masked, because
// the scores of such a row are about -1.4 * 10⁹, where floats are 128 apart. One kernel computes the key and value
// gradients: every SIMD group keeps R keys and loops over the query blocks of every query head that uses them. With the
// head sizes 32 and 64, the same kernel multiplies the score gradient of every block with the keys of the threadgroup and
// adds the product to the query gradient with atomics, so that the backward pass computes the weights once: the query
// gradient is then not bit-for-bit reproducible, because the order of the additions changes. With the head size 128, the
// threadgroup memory does not hold the keys, and a second kernel computes the query gradient from the weights again.
// When few threadgroups would compute the key and value gradients, as with few key heads, several threadgroups share the
// query blocks of a block of keys and add the key and value gradients with atomics.
//
// The strides of the operands are parameters, so that multi-head attention reads the heads from its projections, in
// which the heads of a position are next to each other, and writes its result in the same layout.
//
// With a mask, a SIMD group skips the products of a block in which every score is masked, when every row of the block
// already has an unmasked score: the weights of the block are then exactly 0. A threadgroup in which every SIMD group
// skips a block does not load it. A causal mask skips about half of the blocks. A row in which every score is masked is
// computed like any other row. The votes read `attention_mask_tiles`, one byte for every 8 x 8 scores.
//
// The kernels use the layout of the elements of a SIMD group matrix in the threads: thread `lane` holds the elements
// (row, column) and (row, column + 1), where `matrix_position(lane)` is (column, row). Metal does not document the
// layout; the GPU tests of attention compare the results with the CPU and fail when it changes.

struct AttentionParameters {
    int queryCount;
    int keyCount;
    // Number of query heads, and the number of query heads that share a key head and a value head.
    int heads;
    int keyGroup;
    int batchSize;
    // log2(e) / temperature, 10⁹ * log2(e), and 1 / temperature
    float scale;
    float maskScale;
    float gradientScale;
    // Strides of the batch, head, and row axes in elements: of the queries, the result, and their gradients, and of the keys,
    // the values, and their gradients. The keys and the values have the batch stride 0 when they broadcast along the batch.
    int queryStrides[3];
    int keyStrides[3];
    // Strides of the mask along the batch, head, query, and key axes, 0 for axes along which it broadcasts, and the strides
    // of its tiles of 8 x 8 scores in `attention_mask_tiles`.
    int maskStrides[4];
    int tileStrides[4];
    int hasMask;
    // Whether a kernel adds to the query, key, and value gradients, and which of the key and value gradients it computes.
    int accumulate[3];
    int computes[3];
    // Number of threadgroups that share the query blocks of a block of keys in the key and value kernel. With more than one,
    // they add the key and value gradients with atomics.
    int splits;
};

constant constexpr float RESCALE_THRESHOLD = 8.0f;

// The outputs of the forward kernel.
constant constexpr int RESULT = 1;
constant constexpr int STATISTICS = 2;
constant constexpr int ZEROS = 4;

// A score whose mask is at least MASKED is at most log2(e) * (score - 5 * 10⁸). A row whose largest score is above
// ATTENDS has an unmasked score, so the weight of the masked score, exp2(score - maximum), is exactly 0 in float.
constant constexpr float MASKED = 0.5f;
constant constexpr float ATTENDS = -1e6f;

// First column and row of the two elements that a thread holds in an 8 x 8 SIMD group matrix.
inline ushort2 matrix_position(ushort lane) {
    ushort quad = lane / 4;
    return ushort2((quad & 2) * 2 + (lane % 2) * 2, (quad / 4) * 4 + (lane / 2) % 4);
}

// The two elements of a thread in a SIMD group matrix, which are the first two elements of its storage.
inline thread float2& elements(thread simdgroup_float8x8& matrix) {
    return *reinterpret_cast<thread float2*>(&matrix.thread_elements());
}

// The four threads that hold the elements of one row of a matrix differ in the bits 0 and 3 of their lane.
inline float row_maximum(float x) {
    x = max(x, simd_shuffle_xor(x, 1));
    return max(x, simd_shuffle_xor(x, 8));
}

inline float row_sum(float x) {
    x += simd_shuffle_xor(x, 1);
    return x + simd_shuffle_xor(x, 8);
}

// The mask of one row of the scores: of a query in the forward and query kernels, of a key in the key and value kernel.
struct MaskRow {
    device const float* elements;
    int stride;
    int last;

    // The mask of the columns `column` and `column + 1`. A column after the last one has the mask of the last one.
    float2 pair(int column) const {
        return float2(elements[min(column, last) * stride], elements[min(column + 1, last) * stride]);
    }
};

inline MaskRow query_mask(device const float* M, constant AttentionParameters& p, int batch, int head, int query) {
    long offset = long(batch) * p.maskStrides[0] + long(head) * p.maskStrides[1] + long(query) * p.maskStrides[2];
    return MaskRow{M + offset, p.maskStrides[3], p.keyCount - 1};
}

inline MaskRow key_mask(device const float* M, constant AttentionParameters& p, int batch, int head, int key) {
    long offset = long(batch) * p.maskStrides[0] + long(head) * p.maskStrides[1] + long(key) * p.maskStrides[3];
    return MaskRow{M + offset, p.maskStrides[2], p.queryCount - 1};
}

// Whether every score of the mask tile at the given tile row and column is masked. Tiles after the last have the value
// of the last one.
inline bool tile_masked(device const uchar* tiles, constant AttentionParameters& p, int batch, int head, int queryTile, int keyTile) {
    int lastQueryTile = (p.queryCount - 1) / 8, lastKeyTile = (p.keyCount - 1) / 8;
    long offset = long(batch) * p.tileStrides[0] + long(head) * p.tileStrides[1] + long(min(queryTile, lastQueryTile)) * p.tileStrides[2] + min(keyTile, lastKeyTile) * p.tileStrides[3];
    return tiles[offset] != 0;
}

// Writes 1 for every tile of 8 x 8 scores of the mask in which every score is masked, and 0 for the others. The tiles
// have the shape of the mask with the query and key axes divided by 8, and the threads along x, y, and z cover the key
// tiles, the query tiles, and the batch and head axes of the mask.
kernel void attention_mask_tiles(device const float* M [[buffer(0)]], device uchar* tiles [[buffer(1)]], constant AttentionParameters& p [[buffer(2)]],
                                 constant int2& maskShape [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    int queryCount = p.maskStrides[2] == 0 ? 1 : p.queryCount, keyCount = p.maskStrides[3] == 0 ? 1 : p.keyCount;
    if (int(index.x) * 8 >= keyCount || int(index.y) * 8 >= queryCount) {
        return;
    }
    int batch = int(index.z) / maskShape.y, head = int(index.z) % maskShape.y;
    device const float* m = M + long(batch) * p.maskStrides[0] + long(head) * p.maskStrides[1];
    bool masked = true;
    for (int query = int(index.y) * 8; query < min(int(index.y) * 8 + 8, queryCount); query++) {
        for (int key = int(index.x) * 8; key < min(int(index.x) * 8 + 8, keyCount); key++) {
            masked = masked && m[long(query) * p.maskStrides[2] + long(key) * p.maskStrides[3]] >= MASKED;
        }
    }
    tiles[long(batch) * p.tileStrides[0] + long(head) * p.tileStrides[1] + long(index.y) * p.tileStrides[2] + index.x * p.tileStrides[3]] = masked ? 1 : 0;
}

// Whether every SIMD group of the threadgroup skips a block, so that the threadgroup does not load it. Every thread of
// the threadgroup calls it with the vote of its SIMD group.
template <int SIMDS>
inline bool threadgroup_skips(bool skips, threadgroup bool* votes, uint simd, uint lane) {
    if (lane == 0) {
        votes[simd] = skips;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool all = true;
    UNROLL for (int i = 0; i < SIMDS; i++) {
        all = all && votes[i];
    }
    return all;
}

// Loads rows [first, first + ROWS) of a matrix with `count` rows of D elements, `stride` elements apart, into threadgroup
// memory, with zeros for the rows after the last.
template <int ROWS, int D, int THREADS>
inline void load_rows(threadgroup float* target, device const float* source, int stride, int first, int count, uint thread_index) {
    UNROLL for (int e = int(thread_index) * 4; e < ROWS * D; e += THREADS * 4) {
        int row = e / D;
        float4 x = first + row < count ? *(device const float4*)(source + long(first + row) * stride + e % D) : float4(0.0f);
        *(threadgroup float4*)(target + e) = x;
    }
}

// Forward pass. The threadgroups along x, y, and z cover the query blocks, the query heads, and the batch.
// The bits of `outputs` select what the kernel writes: RESULT the result, STATISTICS m, 1 / l, and D, for which it reads
// the gradient of the result, and ZEROS zeros into Z, which has the layout of the result, for the query gradient that the
// key and value kernel adds to with atomics.
template <int D, int R, int BC, int SIMDS>
kernel void attention_forward(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], device const float* V [[buffer(2)]],
                              device const float* M [[buffer(3)]], device float* O [[buffer(4)]],
                              device float* Mx [[buffer(5)]], device float* Inv [[buffer(6)]], device float* Dsum [[buffer(7)]], device const float* G [[buffer(8)]],
                              constant AttentionParameters& p [[buffer(9)]], constant int& outputs [[buffer(10)]], device const uchar* T [[buffer(11)]],
                              device float* Z [[buffer(12)]],
                              uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int RB = R / 8, CB = BC / 8, DB = D / 8;
    threadgroup float Ks[BC * D];
    threadgroup float Vs[BC * D];
    threadgroup bool votes[SIMDS];

    const int batch = group.z, head = group.y;
    const int keyHead = head / p.keyGroup;
    const long slice = long(batch) * p.heads + head;
    device const float* q = Q + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
    device const float* k = K + long(batch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    device const float* v = V + long(batch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    const ushort2 position = matrix_position(lane);
    const int firstRow = int(group.x) * R * SIMDS + int(simd) * R;

    simdgroup_float8x8 queries[RB][DB], result[RB][DB];
    float maximum[RB], sum[RB];
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        UNROLL for (int db = 0; db < DB; db++) {
            // The queries are scaled once, so that the products are the scores in the base 2 domain.
            elements(queries[rb][db]) = row < p.queryCount ? *(device const float2*)(q + long(row) * p.queryStrides[2] + db * 8 + position.x) * p.scale : float2(0.0f);
            result[rb][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        maximum[rb] = -INFINITY;
        sum[rb] = 0.0f;
    }

    for (int firstKey = 0; firstKey < p.keyCount; firstKey += BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool skips = false;
        if (p.hasMask) {
            skips = int(lane) >= RB * CB || tile_masked(T, p, batch, head, firstRow / 8 + int(lane) / CB, firstKey / 8 + int(lane) % CB);
            UNROLL for (int rb = 0; rb < RB; rb++) {
                skips = skips && maximum[rb] > ATTENDS;
            }
            skips = simd_all(skips);
            if (threadgroup_skips<SIMDS>(skips, votes, simd, lane)) {
                continue;
            }
        }
        load_rows<BC, D, SIMDS * 32>(Ks, k, p.keyStrides[2], firstKey, p.keyCount, thread_index);
        load_rows<BC, D, SIMDS * 32>(Vs, v, p.keyStrides[2], firstKey, p.keyCount, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (skips) {
            continue;
        }

        simdgroup_float8x8 scores[RB][CB];
        UNROLL for (int rb = 0; rb < RB; rb++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                scores[rb][cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        UNROLL for (int db = 0; db < DB; db++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                simdgroup_float8x8 keys;
                simdgroup_load(keys, Ks + cb * 8 * D + db * 8, D, 0, true);
                UNROLL for (int rb = 0; rb < RB; rb++) {
                    simdgroup_multiply_accumulate(scores[rb][cb], queries[rb][db], keys, scores[rb][cb]);
                }
            }
        }

        const bool lastBlock = firstKey + BC > p.keyCount;
        bool rescales = false;
        float blockMaximum[RB];
        UNROLL for (int rb = 0; rb < RB; rb++) {
            // The mask is read after the products: registers that keep it through the products slow down the kernel.
            MaskRow mask = query_mask(M, p, batch, head, min(firstRow + rb * 8 + position.y, p.queryCount - 1));
            float rowMaximum = -INFINITY;
            UNROLL for (int cb = 0; cb < CB; cb++) {
                int column = firstKey + cb * 8 + position.x;
                float2 s = elements(scores[rb][cb]);
                if (p.hasMask) {
                    s -= p.maskScale * mask.pair(column);
                }
                if (lastBlock) {
                    s.x = column < p.keyCount ? s.x : -INFINITY;
                    s.y = column + 1 < p.keyCount ? s.y : -INFINITY;
                }
                elements(scores[rb][cb]) = s;
                rowMaximum = max(rowMaximum, max(s.x, s.y));
            }
            blockMaximum[rb] = row_maximum(rowMaximum);
            rescales = rescales || blockMaximum[rb] > maximum[rb] + RESCALE_THRESHOLD;
        }
        if (simd_any(rescales)) {
            UNROLL for (int rb = 0; rb < RB; rb++) {
                float newMaximum = max(maximum[rb], blockMaximum[rb]);
                float factor = exp2(maximum[rb] - newMaximum);
                maximum[rb] = newMaximum;
                sum[rb] *= factor;
                UNROLL for (int db = 0; db < DB; db++) {
                    elements(result[rb][db]) *= factor;
                }
            }
        }
        UNROLL for (int rb = 0; rb < RB; rb++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                float2 e = exp2(elements(scores[rb][cb]) - maximum[rb]);
                elements(scores[rb][cb]) = e;
                sum[rb] += e.x + e.y;
            }
        }

        UNROLL for (int cb = 0; cb < CB; cb++) {
            UNROLL for (int db = 0; db < DB; db++) {
                simdgroup_float8x8 values;
                simdgroup_load(values, Vs + cb * 8 * D + db * 8, D);
                UNROLL for (int rb = 0; rb < RB; rb++) {
                    simdgroup_multiply_accumulate(result[rb][db], scores[rb][cb], values, result[rb][db]);
                }
            }
        }
    }

    const long queryOffset = long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
    const bool statistics = outputs & STATISTICS;
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        float total = row_sum(sum[rb]);
        float inverse = 1.0f / total;
        float gradientDot = 0.0f;
        UNROLL for (int db = 0; db < DB; db++) {
            float2 x = elements(result[rb][db]) * inverse;
            if (row < p.queryCount) {
                long offset = queryOffset + long(row) * p.queryStrides[2] + db * 8 + position.x;
                if (outputs & RESULT) {
                    *(device float2*)(O + offset) = x;
                }
                if (statistics) {
                    float2 y = *(device const float2*)(G + offset);
                    gradientDot += x.x * y.x + x.y * y.y;
                }
                if (outputs & ZEROS) {
                    *(device float2*)(Z + offset) = float2(0.0f);
                }
            }
        }
        if (statistics) {
            gradientDot = row_sum(gradientDot);
            if (row < p.queryCount && position.x == 0) {
                Mx[slice * p.queryCount + row] = maximum[rb];
                Inv[slice * p.queryCount + row] = inverse;
                Dsum[slice * p.queryCount + row] = gradientDot;
            }
        }
    }
}

// Gradients of the keys and the values. The threadgroups along x, y, and z cover the key blocks, the key heads, and
// the batch of the keys. Every threadgroup loops over the query heads of its group and, for keys that broadcast along
// the batch, over the batch.
template <int D, int R, int BQ, int SIMDS, bool QG>
kernel void attention_key_value_gradient(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], device const float* V [[buffer(2)]],
                                         device const float* M [[buffer(3)]], device const float* Mx [[buffer(4)]], device const float* Inv [[buffer(5)]],
                                         device const float* Dsum [[buffer(6)]], device const float* G [[buffer(7)]], device float* dK [[buffer(8)]], device float* dV [[buffer(9)]],
                                         constant AttentionParameters& p [[buffer(10)]], device const uchar* T [[buffer(11)]], device atomic_float* dQ [[buffer(12)]],
                                         uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                                         uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int RB = R / 8, CB = BQ / 8, DB = D / 8, KEYS = R * SIMDS;
    static_assert(!QG || DB % SIMDS == 0, "Every SIMD group computes the same number of columns of the query gradient.");
    threadgroup float Qs[BQ * D];
    threadgroup float Gs[BQ * D];
    // With QG, the unscaled keys of the threadgroup and the score gradient of a block, [BQ, KEYS], for the query gradient
    threadgroup float Ks[QG ? KEYS * D : 1];
    threadgroup float Ss[QG ? BQ * KEYS : 1];
    threadgroup float Ms[BQ];
    threadgroup float Is[BQ];
    threadgroup float Ds[BQ];
    threadgroup bool votes[SIMDS];

    const int keyBatch = int(group.z) / p.splits, split = int(group.z) % p.splits, keyHead = group.y;
    device const float* k = K + long(keyBatch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    device const float* v = V + long(keyBatch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    const ushort2 position = matrix_position(lane);
    const int firstRow = int(group.x) * R * SIMDS + int(simd) * R;
    const bool computesKeys = p.computes[1];

    simdgroup_float8x8 keys[RB][DB], values[RB][DB], keyGradient[RB][DB], valueGradient[RB][DB];
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        UNROLL for (int db = 0; db < DB; db++) {
            elements(keys[rb][db]) = row < p.keyCount ? *(device const float2*)(k + long(row) * p.keyStrides[2] + db * 8 + position.x) * p.scale : float2(0.0f);
            elements(values[rb][db]) = row < p.keyCount ? *(device const float2*)(v + long(row) * p.keyStrides[2] + db * 8 + position.x) : float2(0.0f);
            keyGradient[rb][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            valueGradient[rb][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    if (QG && p.computes[0]) {
        // The first barrier of the loop makes the keys visible to every SIMD group.
        load_rows<KEYS, D, SIMDS * 32>(Ks, k, p.keyStrides[2], int(group.x) * KEYS, p.keyCount, thread_index);
    }

    // The query blocks of every batch and query head that use the keys, of which the threadgroup processes its part.
    const int firstBatch = p.keyStrides[0] == 0 ? 0 : keyBatch;
    const int batches = p.keyStrides[0] == 0 ? p.batchSize : 1;
    const int queryBlocks = (p.queryCount + BQ - 1) / BQ;
    const int items = batches * p.keyGroup * queryBlocks;
    const int lastItem = (split + 1) * items / p.splits;
    for (int item = split * items / p.splits; item < lastItem; item++) {
        const int batch = firstBatch + item / (p.keyGroup * queryBlocks);
        const int head = keyHead * p.keyGroup + (item / queryBlocks) % p.keyGroup;
        const long slice = long(batch) * p.heads + head;
        device const float* q = Q + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
        device const float* g = G + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
        const int firstQuery = (item % queryBlocks) * BQ;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool skips = false;
        if (p.hasMask) {
            skips = int(lane) >= RB * CB || tile_masked(T, p, batch, head, firstQuery / 8 + int(lane) % CB, firstRow / 8 + int(lane) / CB);
            // Every lane checks the largest scores of one query of the block.
            skips = skips && (int(lane) >= BQ || Mx[slice * p.queryCount + min(firstQuery + int(lane), p.queryCount - 1)] > ATTENDS);
            skips = simd_all(skips);
            if (threadgroup_skips<SIMDS>(skips, votes, simd, lane)) {
                continue;
            }
        }
        load_rows<BQ, D, SIMDS * 32>(Qs, q, p.queryStrides[2], firstQuery, p.queryCount, thread_index);
        load_rows<BQ, D, SIMDS * 32>(Gs, g, p.queryStrides[2], firstQuery, p.queryCount, thread_index);
        for (int i = int(thread_index); i < BQ; i += SIMDS * 32) {
            // A query after the last one has the largest score infinity, so that its weights are 0.
            bool valid = firstQuery + i < p.queryCount;
            Ms[i] = valid ? Mx[slice * p.queryCount + firstQuery + i] : INFINITY;
            Is[i] = valid ? Inv[slice * p.queryCount + firstQuery + i] : 0.0f;
            Ds[i] = valid ? Dsum[slice * p.queryCount + firstQuery + i] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A SIMD group that skips the block writes zeros for the query gradient, because the product of the query
        // gradient needs every SIMD group of the threadgroup.
        const bool computesQueries = QG && p.computes[0];
        simdgroup_float8x8 scoreGradient[RB][CB];
        UNROLL for (int rb = 0; rb < RB; rb++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                scoreGradient[rb][cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        if (!skips) {
            // The transposed weights, Pᵀ = exp2(K Qᵀ * scale - mask - m) / l.
            simdgroup_float8x8 weights[RB][CB];
            UNROLL for (int rb = 0; rb < RB; rb++) {
                UNROLL for (int cb = 0; cb < CB; cb++) {
                    weights[rb][cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            UNROLL for (int db = 0; db < DB; db++) {
                UNROLL for (int cb = 0; cb < CB; cb++) {
                    simdgroup_float8x8 queries;
                    simdgroup_load(queries, Qs + cb * 8 * D + db * 8, D, 0, true);
                    UNROLL for (int rb = 0; rb < RB; rb++) {
                        simdgroup_multiply_accumulate(weights[rb][cb], keys[rb][db], queries, weights[rb][cb]);
                    }
                }
            }
            UNROLL for (int rb = 0; rb < RB; rb++) {
                MaskRow mask = key_mask(M, p, batch, head, min(firstRow + rb * 8 + position.y, p.keyCount - 1));
                UNROLL for (int cb = 0; cb < CB; cb++) {
                    int column = cb * 8 + position.x;
                    float2 s = elements(weights[rb][cb]);
                    if (p.hasMask) {
                        s -= p.maskScale * mask.pair(firstQuery + column);
                    }
                    elements(weights[rb][cb]) = exp2(s - float2(Ms[column], Ms[column + 1])) * float2(Is[column], Is[column + 1]);
                }
            }

            // dV += Pᵀ G
            UNROLL for (int cb = 0; cb < CB; cb++) {
                UNROLL for (int db = 0; db < DB; db++) {
                    simdgroup_float8x8 gradient;
                    simdgroup_load(gradient, Gs + cb * 8 * D + db * 8, D);
                    UNROLL for (int rb = 0; rb < RB; rb++) {
                        simdgroup_multiply_accumulate(valueGradient[rb][db], weights[rb][cb], gradient, valueGradient[rb][db]);
                    }
                }
            }

            if (computesKeys || computesQueries) {
                // dPᵀ = V Gᵀ, and the transposed score gradient dSᵀ = Pᵀ * (dPᵀ - D).
                UNROLL for (int db = 0; db < DB; db++) {
                    UNROLL for (int cb = 0; cb < CB; cb++) {
                        simdgroup_float8x8 gradient;
                        simdgroup_load(gradient, Gs + cb * 8 * D + db * 8, D, 0, true);
                        UNROLL for (int rb = 0; rb < RB; rb++) {
                            simdgroup_multiply_accumulate(scoreGradient[rb][cb], values[rb][db], gradient, scoreGradient[rb][cb]);
                        }
                    }
                }
                UNROLL for (int rb = 0; rb < RB; rb++) {
                    UNROLL for (int cb = 0; cb < CB; cb++) {
                        int column = cb * 8 + position.x;
                        float2 difference = elements(scoreGradient[rb][cb]) - float2(Ds[column], Ds[column + 1]);
                        elements(scoreGradient[rb][cb]) = elements(weights[rb][cb]) * difference;
                    }
                }
            }

            // dK += dSᵀ Q
            if (computesKeys) {
                UNROLL for (int cb = 0; cb < CB; cb++) {
                    UNROLL for (int db = 0; db < DB; db++) {
                        simdgroup_float8x8 queries;
                        simdgroup_load(queries, Qs + cb * 8 * D + db * 8, D);
                        UNROLL for (int rb = 0; rb < RB; rb++) {
                            simdgroup_multiply_accumulate(keyGradient[rb][db], scoreGradient[rb][cb], queries, keyGradient[rb][db]);
                        }
                    }
                }
            }
        }

        // dQ += dS K for the keys of the threadgroup, added to the query gradient with atomics. SIMD group s computes the
        // columns s, s + SIMDS, ... of 8 elements of the [BQ, D] product from dS in threadgroup memory, so that it loads
        // every block of the keys once for all rows of the product.
        if (computesQueries) {
            UNROLL for (int rb = 0; rb < RB; rb++) {
                UNROLL for (int cb = 0; cb < CB; cb++) {
                    simdgroup_store(scoreGradient[rb][cb], Ss + cb * 8 * KEYS + int(simd) * R + rb * 8, KEYS, 0, true);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            UNROLL for (int i = 0; i < DB / SIMDS; i++) {
                int db = int(simd) + i * SIMDS;
                simdgroup_float8x8 product[CB];
                UNROLL for (int qb = 0; qb < CB; qb++) {
                    product[qb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
                UNROLL for (int kb = 0; kb < KEYS / 8; kb++) {
                    simdgroup_float8x8 keyBlock;
                    simdgroup_load(keyBlock, Ks + kb * 8 * D + db * 8, D);
                    UNROLL for (int qb = 0; qb < CB; qb++) {
                        simdgroup_float8x8 scores;
                        simdgroup_load(scores, Ss + qb * 8 * KEYS + kb * 8, KEYS);
                        simdgroup_multiply_accumulate(product[qb], scores, keyBlock, product[qb]);
                    }
                }
                UNROLL for (int qb = 0; qb < CB; qb++) {
                    int query = firstQuery + qb * 8 + position.y;
                    if (query < p.queryCount) {
                        float2 x = elements(product[qb]) * p.gradientScale;
                        device atomic_float* target = dQ + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1] + long(query) * p.queryStrides[2] + db * 8 + position.x;
                        atomic_fetch_add_explicit(target, x.x, memory_order_relaxed);
                        atomic_fetch_add_explicit(target + 1, x.y, memory_order_relaxed);
                    }
                }
            }
        }
    }

    device float* dk = dK + long(keyBatch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    device float* dv = dV + long(keyBatch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        if (row >= p.keyCount) {
            continue;
        }
        UNROLL for (int db = 0; db < DB; db++) {
            long offset = long(row) * p.keyStrides[2] + db * 8 + position.x;
            float2 valueElements = elements(valueGradient[rb][db]);
            float2 keyElements = elements(keyGradient[rb][db]) * p.gradientScale;
            if (p.splits > 1) {
                // The gradients start at zero or hold the accumulated gradient.
                if (p.computes[2]) {
                    atomic_fetch_add_explicit((device atomic_float*)(dv + offset), valueElements.x, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_float*)(dv + offset + 1), valueElements.y, memory_order_relaxed);
                }
                if (computesKeys) {
                    atomic_fetch_add_explicit((device atomic_float*)(dk + offset), keyElements.x, memory_order_relaxed);
                    atomic_fetch_add_explicit((device atomic_float*)(dk + offset + 1), keyElements.y, memory_order_relaxed);
                }
                continue;
            }
            if (p.computes[2]) {
                *(device float2*)(dv + offset) = p.accumulate[2] ? *(device float2*)(dv + offset) + valueElements : valueElements;
            }
            if (computesKeys) {
                *(device float2*)(dk + offset) = p.accumulate[1] ? *(device float2*)(dk + offset) + keyElements : keyElements;
            }
        }
    }
}

// Gradient of the queries. The threadgroups along x, y, and z cover the query blocks, the query heads, and the batch.
template <int D, int R, int BC, int SIMDS>
kernel void attention_query_gradient(device const float* Q [[buffer(0)]], device const float* K [[buffer(1)]], device const float* V [[buffer(2)]],
                                     device const float* M [[buffer(3)]], device const float* Mx [[buffer(4)]], device const float* Inv [[buffer(5)]],
                                     device const float* Dsum [[buffer(6)]], device const float* G [[buffer(7)]], device float* dQ [[buffer(8)]],
                                     constant AttentionParameters& p [[buffer(9)]], device const uchar* T [[buffer(10)]],
                                     uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int RB = R / 8, CB = BC / 8, DB = D / 8;
    threadgroup float Ks[BC * D];
    threadgroup float Vs[BC * D];
    threadgroup bool votes[SIMDS];

    const int batch = group.z, head = group.y;
    const int keyHead = head / p.keyGroup;
    const long slice = long(batch) * p.heads + head;
    device const float* q = Q + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
    device const float* g = G + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
    device const float* k = K + long(batch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    device const float* v = V + long(batch) * p.keyStrides[0] + long(keyHead) * p.keyStrides[1];
    const ushort2 position = matrix_position(lane);
    const int firstRow = int(group.x) * R * SIMDS + int(simd) * R;

    simdgroup_float8x8 queries[RB][DB], gradients[RB][DB], queryGradient[RB][DB];
    float rowMaximum[RB], inverseSum[RB], rowDot[RB];
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        bool valid = row < p.queryCount;
        UNROLL for (int db = 0; db < DB; db++) {
            long offset = long(row) * p.queryStrides[2] + db * 8 + position.x;
            elements(queries[rb][db]) = valid ? *(device const float2*)(q + offset) * p.scale : float2(0.0f);
            elements(gradients[rb][db]) = valid ? *(device const float2*)(g + offset) : float2(0.0f);
            queryGradient[rb][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        rowMaximum[rb] = valid ? Mx[slice * p.queryCount + row] : INFINITY;
        inverseSum[rb] = valid ? Inv[slice * p.queryCount + row] : 0.0f;
        rowDot[rb] = valid ? Dsum[slice * p.queryCount + row] : 0.0f;
    }

    for (int firstKey = 0; firstKey < p.keyCount; firstKey += BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool skips = false;
        if (p.hasMask) {
            skips = int(lane) >= RB * CB || tile_masked(T, p, batch, head, firstRow / 8 + int(lane) / CB, firstKey / 8 + int(lane) % CB);
            UNROLL for (int rb = 0; rb < RB; rb++) {
                skips = skips && rowMaximum[rb] > ATTENDS;
            }
            skips = simd_all(skips);
            if (threadgroup_skips<SIMDS>(skips, votes, simd, lane)) {
                continue;
            }
        }
        load_rows<BC, D, SIMDS * 32>(Ks, k, p.keyStrides[2], firstKey, p.keyCount, thread_index);
        load_rows<BC, D, SIMDS * 32>(Vs, v, p.keyStrides[2], firstKey, p.keyCount, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (skips) {
            continue;
        }

        // P = exp2(Q Kᵀ * scale - mask - m) / l
        simdgroup_float8x8 weights[RB][CB], scoreGradient[RB][CB];
        UNROLL for (int rb = 0; rb < RB; rb++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                weights[rb][cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                scoreGradient[rb][cb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        UNROLL for (int db = 0; db < DB; db++) {
            UNROLL for (int cb = 0; cb < CB; cb++) {
                simdgroup_float8x8 keys, values;
                simdgroup_load(keys, Ks + cb * 8 * D + db * 8, D, 0, true);
                simdgroup_load(values, Vs + cb * 8 * D + db * 8, D, 0, true);
                UNROLL for (int rb = 0; rb < RB; rb++) {
                    simdgroup_multiply_accumulate(weights[rb][cb], queries[rb][db], keys, weights[rb][cb]);
                    simdgroup_multiply_accumulate(scoreGradient[rb][cb], gradients[rb][db], values, scoreGradient[rb][cb]);
                }
            }
        }
        // dS = P * (G Vᵀ - D)
        const bool lastBlock = firstKey + BC > p.keyCount;
        UNROLL for (int rb = 0; rb < RB; rb++) {
            MaskRow mask = query_mask(M, p, batch, head, min(firstRow + rb * 8 + position.y, p.queryCount - 1));
            UNROLL for (int cb = 0; cb < CB; cb++) {
                int column = firstKey + cb * 8 + position.x;
                float2 s = elements(weights[rb][cb]);
                if (p.hasMask) {
                    s -= p.maskScale * mask.pair(column);
                }
                float2 w = exp2(s - rowMaximum[rb]) * inverseSum[rb];
                if (lastBlock) {
                    w.x = column < p.keyCount ? w.x : 0.0f;
                    w.y = column + 1 < p.keyCount ? w.y : 0.0f;
                }
                elements(scoreGradient[rb][cb]) = w * (elements(scoreGradient[rb][cb]) - rowDot[rb]);
            }
        }

        // dQ += dS K
        UNROLL for (int cb = 0; cb < CB; cb++) {
            UNROLL for (int db = 0; db < DB; db++) {
                simdgroup_float8x8 keys;
                simdgroup_load(keys, Ks + cb * 8 * D + db * 8, D);
                UNROLL for (int rb = 0; rb < RB; rb++) {
                    simdgroup_multiply_accumulate(queryGradient[rb][db], scoreGradient[rb][cb], keys, queryGradient[rb][db]);
                }
            }
        }
    }

    device float* dq = dQ + long(batch) * p.queryStrides[0] + long(head) * p.queryStrides[1];
    UNROLL for (int rb = 0; rb < RB; rb++) {
        int row = firstRow + rb * 8 + position.y;
        if (row >= p.queryCount) {
            continue;
        }
        UNROLL for (int db = 0; db < DB; db++) {
            long offset = long(row) * p.queryStrides[2] + db * 8 + position.x;
            float2 x = elements(queryGradient[rb][db]) * p.gradientScale;
            *(device float2*)(dq + offset) = p.accumulate[0] ? *(device float2*)(dq + offset) + x : x;
        }
    }
}

// Instantiates the kernels for a head size with the rows per SIMD group, the block length, and the number of SIMD groups
// of the forward kernel (F), the key and value kernel (K), and the query kernel (Q). `GPUFusedAttention.swift` has the
// same configurations.
#define ATTENTION_KERNELS(D, FR, FB, FS, KR, KB, KS, QR, QB, QS) \
template [[host_name("attention_forward_" #D)]] kernel void attention_forward<D, FR, FB, FS>(device const float*, device const float*, device const float*, device const float*, device float*, device float*, device float*, device float*, device const float*, constant AttentionParameters&, constant int&, device const uchar*, device float*, uint3, uint, uint, uint); \
template [[host_name("attention_key_value_gradient_" #D)]] kernel void attention_key_value_gradient<D, KR, KB, KS, false>(device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device float*, device float*, constant AttentionParameters&, device const uchar*, device atomic_float*, uint3, uint, uint, uint); \
template [[host_name("attention_query_gradient_" #D)]] kernel void attention_query_gradient<D, QR, QB, QS>(device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device float*, constant AttentionParameters&, device const uchar*, uint3, uint, uint, uint);

// The key and value kernel that also adds the query gradient, with the configuration of the key and value kernel. With the
// head size 128, its threadgroup memory exceeds 32 KB.
#define ATTENTION_GRADIENTS(D, R, B, S) \
template [[host_name("attention_gradients_" #D)]] kernel void attention_key_value_gradient<D, R, B, S, true>(device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device const float*, device float*, device float*, constant AttentionParameters&, device const uchar*, device atomic_float*, uint3, uint, uint, uint);

ATTENTION_KERNELS(32, 16, 32, 4, 16, 16, 4, 16, 16, 4)
ATTENTION_KERNELS(64, 16, 32, 4, 8, 16, 8, 8, 32, 8)
ATTENTION_KERNELS(128, 16, 16, 4, 8, 16, 8, 8, 16, 8)
ATTENTION_GRADIENTS(32, 16, 16, 4)
ATTENTION_GRADIENTS(64, 8, 16, 8)

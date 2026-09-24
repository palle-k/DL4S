//
//  fused.metal
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

// Kernels of the fused operations of the GPU, for floats.
//
// Backward kernels either store the gradient or add it to the accumulated gradient in the result buffer,
// selected by the `accumulate` parameter, so that no separate addition is necessary.

inline float sigmoid_(float x) { return 0.5f * precise::tanh(0.5f * x) + 0.5f; }
// log(1 + exp(x)) without an overflow for large x.
inline float softplus_(float x) { return max(x, 0.0f) + precise::log(1.0f + precise::exp(-abs(x))); }

inline void store(device float* result, uint i, float value, uint accumulate) {
    result[i] = accumulate ? result[i] + value : value;
}

// MARK: Activations

// Parameters of element-wise kernels. The parameter of an activation, such as the leakage, is read from a buffer:
// element i uses parameter i % parameterLength, so the parameter can be a scalar or have the shape of the last axes.
struct ElementwiseParameters {
    uint count;
    uint parameterLength;
    uint accumulate;
};

#define ACTIVATION_FORWARD(NAME, EXPRESSION) \
kernel void NAME##_forward(device const float* input [[buffer(0)]], device float* result [[buffer(1)]], device const float* parameters [[buffer(2)]], \
                           constant ElementwiseParameters& p [[buffer(3)]], uint i [[thread_position_in_grid]]) { \
    if (i >= p.count) { return; } \
    float x = input[i]; \
    float a = parameters[i % p.parameterLength]; \
    (void)a; \
    result[i] = (EXPRESSION); \
}

// The value `x` is the input of the forward operation, or its result for activations whose gradient uses the result.
#define ACTIVATION_BACKWARD(NAME, EXPRESSION) \
kernel void NAME##_backward(device const float* input [[buffer(0)]], device const float* outputGradient [[buffer(1)]], device float* result [[buffer(2)]], \
                            device const float* parameters [[buffer(3)]], constant ElementwiseParameters& p [[buffer(4)]], uint i [[thread_position_in_grid]]) { \
    if (i >= p.count) { return; } \
    float x = input[i]; \
    float g = outputGradient[i]; \
    float a = parameters[i % p.parameterLength]; \
    (void)a; \
    store(result, i, (EXPRESSION), p.accumulate); \
}

ACTIVATION_BACKWARD(tanh, (1.0f - x * x) * g)
ACTIVATION_BACKWARD(relu, x > 0.0f ? g : 0.0f)
ACTIVATION_FORWARD(sigmoid, sigmoid_(x))
ACTIVATION_BACKWARD(sigmoid, x * (1.0f - x) * g)
ACTIVATION_FORWARD(leaky_relu, x > 0.0f ? x : a * x)
ACTIVATION_BACKWARD(leaky_relu, (x > 0.0f ? 1.0f : (x < 0.0f ? a : 0.0f)) * g)
ACTIVATION_FORWARD(gelu, x * sigmoid_(1.702f * x))
ACTIVATION_BACKWARD(gelu, (sigmoid_(1.702f * x) + 1.702f * x * sigmoid_(1.702f * x) * (1.0f - sigmoid_(1.702f * x))) * g)
ACTIVATION_FORWARD(swish, x * sigmoid_(a * x))
ACTIVATION_BACKWARD(swish, (sigmoid_(a * x) + a * x * sigmoid_(a * x) * (1.0f - sigmoid_(a * x))) * g)
ACTIVATION_FORWARD(mish, x * precise::tanh(softplus_(x)))
ACTIVATION_BACKWARD(mish, (precise::tanh(softplus_(x)) + x * (1.0f - precise::tanh(softplus_(x)) * precise::tanh(softplus_(x))) * sigmoid_(x)) * g)
ACTIVATION_FORWARD(lisht, x * precise::tanh(x))
ACTIVATION_BACKWARD(lisht, (precise::tanh(x) + x * (1.0f - precise::tanh(x) * precise::tanh(x))) * g)
ACTIVATION_FORWARD(elu, x > 0.0f ? x : a * (precise::exp(x) - 1.0f))
ACTIVATION_BACKWARD(elu, (x > 0.0f ? 1.0f : a * precise::exp(min(x, 0.0f))) * g)
ACTIVATION_FORWARD(softplus, softplus_(x))
ACTIVATION_BACKWARD(softplus, sigmoid_(x) * g)
ACTIVATION_FORWARD(squareplus, 0.5f * (x + precise::sqrt(x * x + 4.0f)))
ACTIVATION_BACKWARD(squareplus, 0.5f * (1.0f + x / precise::sqrt(x * x + 4.0f)) * g)

// MARK: Rows

// Parameters of kernels that process rows. One threadgroup processes one row.
struct RowParameters {
    uint length;
    uint accumulate;
    float epsilon;
};

// Sum and maximum of the values of all threads of a threadgroup. Every thread receives the result.
inline float group_sum(float value, threadgroup float* scratch, uint lane, uint simd, uint simds) {
    value = simd_sum(value);
    if (simds == 1) { return value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { scratch[simd] = value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_sum(lane < simds ? scratch[lane] : 0.0f);
}

inline float group_max(float value, threadgroup float* scratch, uint lane, uint simd, uint simds) {
    value = simd_max(value);
    if (simds == 1) { return value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) { scratch[simd] = value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_max(lane < simds ? scratch[lane] : -INFINITY);
}

#define ROW_ARGUMENTS \
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], uint size [[threads_per_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]

kernel void softmax_forward(device const float* input [[buffer(0)]], device float* result [[buffer(1)]], constant RowParameters& p [[buffer(2)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    device const float* x = input + ulong(row) * p.length;
    device float* y = result + ulong(row) * p.length;
    float maximum = -INFINITY;
    for (uint i = t; i < p.length; i += size) { maximum = max(maximum, x[i]); }
    maximum = group_max(maximum, scratch, lane, simd, simds);
    float sum = 0.0f;
    for (uint i = t; i < p.length; i += size) { sum += precise::exp(x[i] - maximum); }
    float inverse = 1.0f / group_sum(sum, scratch, lane, simd, simds);
    for (uint i = t; i < p.length; i += size) { y[i] = precise::exp(x[i] - maximum) * inverse; }
}

kernel void log_softmax_forward(device const float* input [[buffer(0)]], device float* result [[buffer(1)]], constant RowParameters& p [[buffer(2)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    device const float* x = input + ulong(row) * p.length;
    device float* y = result + ulong(row) * p.length;
    float maximum = -INFINITY;
    for (uint i = t; i < p.length; i += size) { maximum = max(maximum, x[i]); }
    maximum = group_max(maximum, scratch, lane, simd, simds);
    float sum = 0.0f;
    for (uint i = t; i < p.length; i += size) { sum += precise::exp(x[i] - maximum); }
    float offset = maximum + precise::log(group_sum(sum, scratch, lane, simd, simds));
    for (uint i = t; i < p.length; i += size) { y[i] = x[i] - offset; }
}

// Gradient of the softmax from its result y: y * (g - sum(g * y)).
kernel void softmax_backward(device const float* output [[buffer(0)]], device const float* outputGradient [[buffer(1)]], device float* result [[buffer(2)]], constant RowParameters& p [[buffer(3)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    ulong offset = ulong(row) * p.length;
    device const float* y = output + offset;
    device const float* g = outputGradient + offset;
    float dot = 0.0f;
    for (uint i = t; i < p.length; i += size) { dot += g[i] * y[i]; }
    dot = group_sum(dot, scratch, lane, simd, simds);
    for (uint i = t; i < p.length; i += size) { store(result, offset + i, y[i] * (g[i] - dot), p.accumulate); }
}

// Gradient of the log softmax from its result y: g - exp(y) * sum(g).
kernel void log_softmax_backward(device const float* output [[buffer(0)]], device const float* outputGradient [[buffer(1)]], device float* result [[buffer(2)]], constant RowParameters& p [[buffer(3)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    ulong offset = ulong(row) * p.length;
    device const float* y = output + offset;
    device const float* g = outputGradient + offset;
    float sum = 0.0f;
    for (uint i = t; i < p.length; i += size) { sum += g[i]; }
    sum = group_sum(sum, scratch, lane, simd, simds);
    for (uint i = t; i < p.length; i += size) { store(result, offset + i, g[i] - precise::exp(y[i]) * sum, p.accumulate); }
}

// Layer normalization of a row: (x - mean) / (sqrt(variance) + epsilon) * scale + shift.
kernel void layer_norm_forward(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]], device const float* shift [[buffer(2)]],
                               device float* result [[buffer(3)]], constant RowParameters& p [[buffer(4)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    device const float* x = input + ulong(row) * p.length;
    device float* y = result + ulong(row) * p.length;
    float sum = 0.0f;
    for (uint i = t; i < p.length; i += size) { sum += x[i]; }
    float mean = group_sum(sum, scratch, lane, simd, simds) / float(p.length);
    float squares = 0.0f;
    for (uint i = t; i < p.length; i += size) { float centered = x[i] - mean; squares += centered * centered; }
    float variance = group_sum(squares, scratch, lane, simd, simds) / float(p.length);
    float inverse = 1.0f / (precise::sqrt(variance) + p.epsilon);
    for (uint i = t; i < p.length; i += size) { y[i] = (x[i] - mean) * inverse * scale[i] + shift[i]; }
}

// Gradients of layer normalization. With d = sqrt(variance) + epsilon, n = (x - mean) / d, and dn = g * scale:
// dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d.
// The kernel writes g * n for the gradient of the scale into `scaleTerms` when that buffer is used.
kernel void layer_norm_backward(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]], device const float* outputGradient [[buffer(2)]],
                                device float* inputGradient [[buffer(3)]], device float* scaleTerms [[buffer(4)]], constant RowParameters& p [[buffer(5)]],
                                constant uint2& outputs [[buffer(6)]], ROW_ARGUMENTS) {
    threadgroup float scratch[32];
    uint simds = (size + 31) / 32;
    ulong offset = ulong(row) * p.length;
    device const float* x = input + offset;
    device const float* g = outputGradient + offset;
    float sum = 0.0f;
    for (uint i = t; i < p.length; i += size) { sum += x[i]; }
    float mean = group_sum(sum, scratch, lane, simd, simds) / float(p.length);
    float squares = 0.0f;
    for (uint i = t; i < p.length; i += size) { float centered = x[i] - mean; squares += centered * centered; }
    float deviation = precise::sqrt(group_sum(squares, scratch, lane, simd, simds) / float(p.length));
    float divisor = deviation + p.epsilon;
    float inverse = 1.0f / divisor;
    if (outputs.y != 0) {
        for (uint i = t; i < p.length; i += size) { scaleTerms[offset + i] = g[i] * (x[i] - mean) * inverse; }
    }
    if (outputs.x == 0) { return; }
    float gradientSum = 0.0f, correlation = 0.0f;
    for (uint i = t; i < p.length; i += size) {
        float normalizedGradient = g[i] * scale[i];
        gradientSum += normalizedGradient;
        correlation += normalizedGradient * (x[i] - mean) * inverse;
    }
    float meanGradient = group_sum(gradientSum, scratch, lane, simd, simds) / float(p.length);
    correlation = group_sum(correlation, scratch, lane, simd, simds) / float(p.length) * divisor / deviation;
    for (uint i = t; i < p.length; i += size) {
        float normalized = (x[i] - mean) * inverse;
        store(inputGradient, offset + i, (g[i] * scale[i] - meanGradient - normalized * correlation) * inverse, p.accumulate);
    }
}

// MARK: Batch normalization

// Batch normalization normalizes every column of a matrix [rows, columns] with the statistics of the column.
// One thread processes one column, so that neighboring threads read neighboring elements of a row.
struct ColumnParameters {
    uint rows;
    uint columns;
    uint accumulate;
    float epsilon;
};

// Normalizes with the statistics of the batch and writes the mean and the biased variance of every column.
kernel void batch_norm_forward(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]], device const float* shift [[buffer(2)]],
                               device float* result [[buffer(3)]], device float* mean [[buffer(4)]], device float* variance [[buffer(5)]],
                               constant ColumnParameters& p [[buffer(6)]], uint j [[thread_position_in_grid]]) {
    if (j >= p.columns) { return; }
    float sum = 0.0f;
    for (uint row = 0; row < p.rows; row++) { sum += input[ulong(row) * p.columns + j]; }
    float mu = sum / float(p.rows);
    float squares = 0.0f;
    for (uint row = 0; row < p.rows; row++) { float centered = input[ulong(row) * p.columns + j] - mu; squares += centered * centered; }
    float sigma2 = squares / float(p.rows);
    mean[j] = mu;
    variance[j] = sigma2;
    float factor = scale[j] / (precise::sqrt(sigma2) + p.epsilon);
    float offset = shift[j] - mu * factor;
    for (uint row = 0; row < p.rows; row++) {
        ulong index = ulong(row) * p.columns + j;
        result[index] = input[index] * factor + offset;
    }
}

// Gradients of batch normalization with the statistics of the batch. With d = sqrt(variance) + epsilon,
// n = (x - mean) / d, and dn = g * scale: dx = (dn - mean(dn) - n * mean(dn * n) * d / sqrt(variance)) / d.
// The sums of g * n and g of every column are written for the gradients of the scale and the shift.
kernel void batch_norm_backward(device const float* input [[buffer(0)]], device const float* scale [[buffer(1)]], device const float* outputGradient [[buffer(2)]],
                                device float* inputGradient [[buffer(3)]], device float* scaleSums [[buffer(4)]], device float* shiftSums [[buffer(5)]],
                                constant ColumnParameters& p [[buffer(6)]], constant uint& computesInput [[buffer(7)]], uint j [[thread_position_in_grid]]) {
    if (j >= p.columns) { return; }
    float sum = 0.0f;
    for (uint row = 0; row < p.rows; row++) { sum += input[ulong(row) * p.columns + j]; }
    float mu = sum / float(p.rows);
    float squares = 0.0f;
    for (uint row = 0; row < p.rows; row++) { float centered = input[ulong(row) * p.columns + j] - mu; squares += centered * centered; }
    float deviation = precise::sqrt(squares / float(p.rows));
    float divisor = deviation + p.epsilon;
    float inverse = 1.0f / divisor;
    float gamma = scale[j];
    float gradientSum = 0.0f, productSum = 0.0f, scaleSum = 0.0f, shiftSum = 0.0f;
    for (uint row = 0; row < p.rows; row++) {
        ulong index = ulong(row) * p.columns + j;
        float normalized = (input[index] - mu) * inverse;
        float g = outputGradient[index];
        gradientSum += g * gamma;
        productSum += g * gamma * normalized;
        scaleSum += g * normalized;
        shiftSum += g;
    }
    scaleSums[j] = scaleSum;
    shiftSums[j] = shiftSum;
    if (computesInput == 0) { return; }
    float meanGradient = gradientSum / float(p.rows);
    float correlation = productSum / float(p.rows) * divisor / deviation;
    for (uint row = 0; row < p.rows; row++) {
        ulong index = ulong(row) * p.columns + j;
        float normalized = (input[index] - mu) * inverse;
        store(inputGradient, uint(index), (outputGradient[index] * gamma - meanGradient - normalized * correlation) * inverse, p.accumulate);
    }
}

// Normalization with fixed statistics: y = x * factor + offset, with the factor and the offset of the column.
kernel void affine_columns(device const float* input [[buffer(0)]], device const float* factors [[buffer(1)]], device const float* offsets [[buffer(2)]],
                           device float* result [[buffer(3)]], constant ColumnParameters& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.columns) { return; }
    uint j = i % p.columns;
    result[i] = input[i] * factors[j] + offsets[j];
}

// Gradients of the normalization with fixed statistics: dx = g * factor, and the sums of g * (x - mean) / d and g of every column.
kernel void batch_norm_fixed_backward(device const float* input [[buffer(0)]], device const float* outputGradient [[buffer(1)]], device const float* factors [[buffer(2)]],
                                      device const float* inverseDivisors [[buffer(3)]], device const float* means [[buffer(4)]], device float* inputGradient [[buffer(5)]],
                                      device float* scaleSums [[buffer(6)]], device float* shiftSums [[buffer(7)]], constant ColumnParameters& p [[buffer(8)]],
                                      constant uint& computesInput [[buffer(9)]], uint j [[thread_position_in_grid]]) {
    if (j >= p.columns) { return; }
    float factor = factors[j], inverse = inverseDivisors[j], mu = means[j];
    float scaleSum = 0.0f, shiftSum = 0.0f;
    for (uint row = 0; row < p.rows; row++) {
        ulong index = ulong(row) * p.columns + j;
        float g = outputGradient[index];
        scaleSum += g * (input[index] - mu) * inverse;
        shiftSum += g;
        if (computesInput != 0) { store(inputGradient, uint(index), g * factor, p.accumulate); }
    }
    scaleSums[j] = scaleSum;
    shiftSums[j] = shiftSum;
}

// MARK: Optimizers

struct AdamParameters {
    uint count;
    uint amsgrad;
    float learningRate, beta1, beta2, epsilon;
    float firstCorrection, secondCorrection;
};

// One Adam step. The moments are updated in place, the updated parameter is written to the result.
kernel void adam_update(device const float* parameter [[buffer(0)]], device const float* gradient [[buffer(1)]], device float* firstMoment [[buffer(2)]],
                        device float* secondMoment [[buffer(3)]], device float* secondMomentMax [[buffer(4)]], device float* result [[buffer(5)]],
                        constant AdamParameters& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) { return; }
    float g = gradient[i];
    float m = p.beta1 * firstMoment[i] + (1.0f - p.beta1) * g;
    float v = p.beta2 * secondMoment[i] + (1.0f - p.beta2) * g * g;
    firstMoment[i] = m;
    secondMoment[i] = v;
    if (p.amsgrad) {
        v = max(secondMomentMax[i], v);
        secondMomentMax[i] = v;
    }
    result[i] = parameter[i] - p.learningRate / (precise::sqrt(v * p.secondCorrection) + p.epsilon) * (m * p.firstCorrection);
}

// MARK: Dropout

// A hash of 32 bits with a low bias (https://nullprogram.com/blog/2018/07/31/). Two rounds with the seed give every
// element an independent random number, so the mask does not depend on the order in which the threads run.
inline uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

struct DropoutParameters {
    uint count;
    uint threshold;
    uint2 seed;
};

kernel void dropout_forward(device const float* input [[buffer(0)]], device float* result [[buffer(1)]], device float* mask [[buffer(2)]],
                            constant DropoutParameters& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) { return; }
    float keep = hash32(hash32(i ^ p.seed.x) + p.seed.y) < p.threshold ? 1.0f : 0.0f;
    mask[i] = keep;
    result[i] = input[i] * keep;
}

ACTIVATION_BACKWARD(dropout, x * g)

// MARK: Gated recurrent unit

// Gates of a step from the projections of the input and the products of the state with the weights.
// The products of the update and reset gates are replaced by the gates, and the reset state is written.
kernel void gru_gates(device const float* updateInput [[buffer(0)]], device float* update [[buffer(1)]], device const float* resetInput [[buffer(2)]],
                      device float* reset [[buffer(3)]], device const float* state [[buffer(4)]], device float* resetState [[buffer(5)]],
                      constant uint& count [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) { return; }
    float z = sigmoid_(updateInput[i] + update[i]);
    float r = sigmoid_(resetInput[i] + reset[i]);
    update[i] = z;
    reset[i] = r;
    resetState[i] = r * state[i];
}

// New state: state + z * (tanh(candidateInput + candidateProduct) - state).
kernel void gru_state(device const float* update [[buffer(0)]], device const float* candidateInput [[buffer(1)]], device const float* candidateProduct [[buffer(2)]],
                      device const float* state [[buffer(3)]], device float* result [[buffer(4)]], constant uint& count [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) { return; }
    float s = state[i];
    result[i] = s + update[i] * (precise::tanh(candidateInput[i] + candidateProduct[i]) - s);
}

// Gradients of the pre-activations of the update gate and the candidate, and the part of the state gradient that flows
// through (1 - z) * state.
kernel void gru_backward_gates(device const float* update [[buffer(0)]], device const float* candidateInput [[buffer(1)]], device const float* candidateProduct [[buffer(2)]],
                               device const float* state [[buffer(3)]], device const float* outputGradient [[buffer(4)]], device float* updateGradient [[buffer(5)]],
                               device float* candidateGradient [[buffer(6)]], device float* stateGradient [[buffer(7)]], constant uint& count [[buffer(8)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= count) { return; }
    float z = update[i];
    float c = precise::tanh(candidateInput[i] + candidateProduct[i]);
    float g = outputGradient[i];
    updateGradient[i] = g * (c - state[i]) * z * (1.0f - z);
    candidateGradient[i] = g * z * (1.0f - c * c);
    stateGradient[i] = g * (1.0f - z);
}

// Gradient of the pre-activation of the reset gate, and the part of the state gradient that flows through reset * state.
kernel void gru_backward_reset(device const float* reset [[buffer(0)]], device const float* state [[buffer(1)]], device const float* resetStateGradient [[buffer(2)]],
                               device float* resetGradient [[buffer(3)]], device float* stateGradient [[buffer(4)]], constant uint& count [[buffer(5)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= count) { return; }
    float r = reset[i];
    float gradient = resetStateGradient[i];
    resetGradient[i] = gradient * state[i] * r * (1.0f - r);
    stateGradient[i] += gradient * r;
}

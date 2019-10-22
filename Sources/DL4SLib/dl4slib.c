#include "dl4slib.h"
#include <immintrin.h>

void avxcpy(void* dst, const void* src, size_t count) {
#ifdef __AVX2__
    const __m256i *pSrc = src;
    __m256i *pDest = dst;
    size_t nVects = count / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
        const __m256i loaded = _mm256_stream_load_si256(pSrc);
        _mm256_stream_si256(pDest, loaded);
    }
    _mm_sfence();
#else
    const __m128 *pSrc = src;
    __m128 *pDest = dst;
    size_t nVects = count / sizeof(*pSrc);
    for (; nVects > 0; nVects--, pSrc++, pDest++) {
      __m128 buffer = _mm_load_ps(pSrc);
      _mm_store_ps(pDest, buffer);
    }
    _mm_sfence();
#endif
}

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>

void d4lib_sfill(const float* src, float* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vfill(src, dst, dst_stride, length); }
void d4lib_dfill(const double* src, double* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vfillD(src, dst, dst_stride, length); }
void d4lib_ifill(const int* src, int* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vfilli(src, dst, dst_stride, length); }

// Vector square
void d4lib_ssquare(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vsq(src, src_stride, dst, dst_stride, length); }
void d4lib_dsquare(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vsqD(src, src_stride, dst, dst_stride, length); }
void d4lib_isquare(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = src[i * src_stride] * src[i * src_stride];
    }
}

// Vector threshold
void d4lib_sthreshold(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dthreshold(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ithreshold(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector negate
void d4lib_sneg(const float* src, d4lib_stride src_stride, float* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vneg(src, src_stride, dst, dst_stride, length); }
void d4lib_dneg(const double* src, d4lib_stride src_stride, double* dst, d4lib_stride dst_stride, d4lib_length length)  { vDSP_vnegD(src, src_stride, dst, dst_stride, length); }
void d4lib_ineg(const int* src, d4lib_stride src_stride, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = -src[i * src_stride];
    }
}

// Vector add
void d4lib_saddv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vadd(lhs, lhs_stride, rhs, rhs_stride, dst, dst_stride, length); }
void d4lib_daddv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length) { vDSP_vaddD(lhs, lhs_stride, rhs, rhs_stride, dst, dst_stride, length); }
void d4lib_iaddv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        dst[i * dst_stride] = lhs[i * lhs_stride] + rhs[i * rhs_stride];
    }
}

// Vector scalar add
void d4lib_saddvs(const float* lhs, d4lib_stride lhs_stride, const float* rhs , float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_daddvs(const double* lhs, d4lib_stride lhs_stride, const double* rhs , double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_iaddvs(const int* lhs, d4lib_stride lhs_stride, const int* rhs , int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector subtract
void d4lib_ssubv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dsubv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_isubv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Scalar vector subtract
void d4lib_ssubsv(const float* lhs, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dsubsv(const double* lhs, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_isubsv(const int* lhs, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector multiply
void d4lib_smulv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dmulv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_imulv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector scalar multiply
void d4lib_smulvs(const float* lhs, d4lib_stride lhs_stride, const float* rhs , float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dmulvs(const double* lhs, d4lib_stride lhs_stride, const double* rhs , double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_imulvs(const int* lhs, d4lib_stride lhs_stride, const int* rhs , int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector divide
void d4lib_sdivv(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ddivv(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_idivv(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Scalar vector divide
void d4lib_sdivsv(const float* lhs, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_ddivsv(const double* lhs, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_idivsv(const int* lhs, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_stride dst_stride, d4lib_length length);

// Vector Sum
void d4lib_ssum(const float* src, d4lib_stride src_stride, float* dst, d4lib_length length);
void d4lib_dsum(const double* src, d4lib_stride src_stride, double* dst, d4lib_length length);
void d4lib_isum(const int* src, d4lib_stride src_stride, int* dst, d4lib_length length);

// Vector dot product
void d4lib_sdot(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length length);
void d4lib_ddot(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length length);
void d4lib_idot(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length length);

// Vector maximum value and index
void d4lib_smaxi(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_dmaxi(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_imaxi(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length* dst_idx, d4lib_length length);

// Vector minimum value and index
void d4lib_smini(const float* lhs, d4lib_stride lhs_stride, const float* rhs, d4lib_stride rhs_stride, float* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_dmini(const double* lhs, d4lib_stride lhs_stride, const double* rhs, d4lib_stride rhs_stride, double* dst, d4lib_length* dst_idx, d4lib_length length);
void d4lib_imini(const int* lhs, d4lib_stride lhs_stride, const int* rhs, d4lib_stride rhs_stride, int* dst, d4lib_length* dst_idx, d4lib_length length);

// Vector ramp
void d4lib_sramp(const float* start, const float* increment, float* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_dramp(const double* start, const double* increment, double* dst, d4lib_stride dst_stride, d4lib_length length);
void d4lib_iramp(const int* start, const int* increment, int* dst, d4lib_stride dst_stride, d4lib_length length);

// single vector math functions
void d4lib_stanh(const float* start, float* dst, d4lib_length length);
void d4lib_dtanh(const double* start, double* dst, d4lib_length length);

void d4lib_sexp(const float* start, float* dst, d4lib_length length);
void d4lib_dexp(const double* start, double* dst, d4lib_length length);

void d4lib_slog(const float* start, float* dst, d4lib_length length);
void d4lib_dlog(const double* start, double* dst, d4lib_length length);

void d4lib_ssqrt(const float* start, float* dst, d4lib_length length);
void d4lib_dsqrt(const double* start, double* dst, d4lib_length length);

void d4lib_ssin(const float* start, float* dst, d4lib_length length);
void d4lib_dsin(const double* start, double* dst, d4lib_length length);

void d4lib_scos(const float* start, float* dst, d4lib_length length);
void d4lib_dcos(const double* start, double* dst, d4lib_length length);

void d4lib_stan(const float* start, float* dst, d4lib_length length);
void d4lib_dtan(const double* start, double* dst, d4lib_length length);

void d4lib_scopysign(const float* start, float* dst, d4lib_length length);
void d4lib_dcopysign(const double* start, double* dst, d4lib_length length);

void d4lib_sheaviside(const float* src, float* dst, d4lib_length length) {
    for (int i = 0; i < length; i++) {
        float s = src[i];
        dst[i] = s > 0 ? 1 : 0;
    }
}
void d4lib_dheaviside(const double* src, double* dst, d4lib_length length);

#elif defined __INTEL_MKL__

#else

#endif

void d4lib_img2col(const float* src, float* dst, D4LIB_Img2ColSetup setup) {
    const int vertical_stride = setup.width;
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    const int output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int rows = setup.kernel_width * setup.kernel_height * setup.channels;
    const int cols = output_width * output_height * setup.batch_size;
    const int patches_per_kernel = output_width * output_height;
    
    int* row_buffer = malloc(sizeof(int) * rows);
    int* col_buffer = malloc(sizeof(int) * cols);
    int* chan_depthstride_buffer = malloc(sizeof(int) * rows);
    int* rcols_buffer = malloc(sizeof(int) * rows);
    
    for (int r = 0; r < rows; r++) {
        row_buffer[r] = r / setup.kernel_width;
        col_buffer[r] = r % setup.kernel_height;
        chan_depthstride_buffer[r] = (r / (setup.kernel_width * setup.kernel_height)) * depth_stride;
        rcols_buffer[r] = r * cols;
    }
    
    for (int c = 0; c < cols; c++) {
        int patch_idx = c % patches_per_kernel;
        int featuremap_idx = c / patches_per_kernel;
        
        int base_dst_col = patch_idx % output_width;
        int base_dst_row = (patch_idx / output_width) % output_height;
        
        int base_src_col = base_dst_col * setup.stride - setup.padding;
        int base_src_row = base_dst_row * setup.stride - setup.padding;
        
        const float* feature_map = &src[featuremap_stride * featuremap_idx];
        
        for (int r = 0; r < rows; r++) {
            int row = base_src_row + row_buffer[r];
            int col = base_src_col + col_buffer[r];
            
            if (row < 0 || row >= setup.height || col < 0 || col >= setup.width) {
                dst[rcols_buffer[r] + c] = 0;
            } else {
                dst[rcols_buffer[r] + c] = feature_map[chan_depthstride_buffer[r] + row * vertical_stride + col];
            }
        }
    }
    
    free(row_buffer);
    free(col_buffer);
    free(chan_depthstride_buffer);
    free(rcols_buffer);
}

void d4lib_col2img(const float* src, float* dst, D4LIB_Img2ColSetup setup) {
    const int vertical_stride = setup.width;
    const int depth_stride = setup.width * setup.height;
    const int featuremap_stride = depth_stride * setup.channels;
    const int output_height = (setup.height + 2 * setup.padding - setup.kernel_height) / setup.stride + 1;
    const int output_width = (setup.width + 2 * setup.padding - setup.kernel_width) / setup.stride + 1;
    const int rows = setup.kernel_width * setup.kernel_height * setup.channels;
    const int cols = output_width * output_height * setup.batch_size;
    const int patches_per_kernel = output_width * output_height;
    
    int* row_buffer = malloc(sizeof(int) * rows);
    int* col_buffer = malloc(sizeof(int) * cols);
    int* chan_depthstride_buffer = malloc(sizeof(int) * rows);
    int* rcols_buffer = malloc(sizeof(int) * rows);
    
    int total_count = setup.width * setup.height * setup.channels * setup.batch_size;
    const float zero = 0;
    d4lib_sfill(&zero, dst, 1, total_count);
    
    for (int r = 0; r < rows; r++) {
        row_buffer[r] = r / setup.kernel_width;
        col_buffer[r] = r % setup.kernel_height;
        chan_depthstride_buffer[r] = (r / (setup.kernel_width * setup.kernel_height)) * depth_stride;
        rcols_buffer[r] = r * cols;
    }
    
    for (int c = 0; c < cols; c++) {
        int patch_idx = c % patches_per_kernel;
        int featuremap_idx = c / patches_per_kernel;
        
        int base_dst_col = patch_idx % output_width;
        int base_dst_row = (patch_idx / output_width) % output_height;
        
        int base_src_col = base_dst_col * setup.stride - setup.padding;
        int base_src_row = base_dst_row * setup.stride - setup.padding;
        
        float* feature_map = &dst[featuremap_stride * featuremap_idx];
        
        for (int r = 0; r < rows; r++) {
            int row = base_src_row + row_buffer[r];
            int col = base_src_col + col_buffer[r];
            
            if (row >= 0 && row < setup.height && col >= 0 && col < setup.width) {
                feature_map[chan_depthstride_buffer[r] + row * vertical_stride + col] += src[rcols_buffer[r] + c];
            }
        }
    }
    
    free(row_buffer);
    free(col_buffer);
    free(chan_depthstride_buffer);
    free(rcols_buffer);
}

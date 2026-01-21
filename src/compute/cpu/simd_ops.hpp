#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(ODI_HAS_AVX2) || defined(ODI_HAS_AVX512)
#include <immintrin.h>
#endif

#if defined(ODI_HAS_NEON)
#include <arm_neon.h>
#endif

namespace odi {

// ============================================================================
// Scalar fallback implementations
// ============================================================================

inline void simd_add_scalar_fallback(float* out, const float* a, float b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b;
    }
}

inline void simd_add_fallback(float* out, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void simd_mul_scalar_fallback(float* out, const float* a, float b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b;
    }
}

inline void simd_mul_fallback(float* out, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

inline float simd_dot_fallback(const float* a, const float* b, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void simd_silu_fallback(float* out, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        out[i] = xi / (1.0f + std::exp(-xi));
    }
}

inline void simd_gelu_fallback(float* out, const float* x, int64_t n) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coef = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float x3 = xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + std::tanh(sqrt_2_over_pi * (xi + coef * x3)));
    }
}

inline void simd_relu_fallback(float* out, const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

inline void simd_softmax_fallback(float* out, const float* x, int64_t n) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int64_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::exp(x[i] - max_val);
        sum += out[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

// ============================================================================
// AVX2 implementations
// ============================================================================

#if defined(ODI_HAS_AVX2)

inline void simd_add_scalar_avx2(float* out, const float* a, float b, int64_t n) {
    __m256 vb = _mm256_set1_ps(b);
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b;
    }
}

inline void simd_add_avx2(float* out, const float* a, const float* b, int64_t n) {
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void simd_mul_scalar_avx2(float* out, const float* a, float b, int64_t n) {
    __m256 vb = _mm256_set1_ps(b);
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] * b;
    }
}

inline void simd_mul_avx2(float* out, const float* a, const float* b, int64_t n) {
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

inline float simd_dot_avx2(const float* a, const float* b, int64_t n) {
    __m256 sum = _mm256_setzero_ps();
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

inline void simd_relu_avx2(float* out, const float* x, int64_t n) {
    __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vr = _mm256_max_ps(vx, zero);
        _mm256_storeu_ps(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

#endif // ODI_HAS_AVX2

// ============================================================================
// NEON implementations
// ============================================================================

#if defined(ODI_HAS_NEON)

inline void simd_add_scalar_neon(float* out, const float* a, float b, int64_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b;
    }
}

inline void simd_add_neon(float* out, const float* a, const float* b, int64_t n) {
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void simd_mul_scalar_neon(float* out, const float* a, float b, int64_t n) {
    float32x4_t vb = vdupq_n_f32(b);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] * b;
    }
}

inline void simd_mul_neon(float* out, const float* a, const float* b, int64_t n) {
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

inline float simd_dot_neon(const float* a, const float* b, int64_t n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, va, vb);
    }

    // Horizontal sum
    float32x2_t sum2 = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    sum2 = vpadd_f32(sum2, sum2);
    float result = vget_lane_f32(sum2, 0);

    // Handle remainder
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

inline void simd_relu_neon(float* out, const float* x, int64_t n) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int64_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vr = vmaxq_f32(vx, zero);
        vst1q_f32(out + i, vr);
    }

    for (; i < n; ++i) {
        out[i] = std::max(0.0f, x[i]);
    }
}

#endif // ODI_HAS_NEON

// ============================================================================
// Dispatch functions (select best implementation at runtime)
// ============================================================================

inline void simd_add_scalar(float* out, const float* a, float b, int64_t n) {
#if defined(ODI_HAS_AVX2)
    simd_add_scalar_avx2(out, a, b, n);
#elif defined(ODI_HAS_NEON)
    simd_add_scalar_neon(out, a, b, n);
#else
    simd_add_scalar_fallback(out, a, b, n);
#endif
}

inline void simd_add(float* out, const float* a, const float* b, int64_t n) {
#if defined(ODI_HAS_AVX2)
    simd_add_avx2(out, a, b, n);
#elif defined(ODI_HAS_NEON)
    simd_add_neon(out, a, b, n);
#else
    simd_add_fallback(out, a, b, n);
#endif
}

inline void simd_mul_scalar(float* out, const float* a, float b, int64_t n) {
#if defined(ODI_HAS_AVX2)
    simd_mul_scalar_avx2(out, a, b, n);
#elif defined(ODI_HAS_NEON)
    simd_mul_scalar_neon(out, a, b, n);
#else
    simd_mul_scalar_fallback(out, a, b, n);
#endif
}

inline void simd_mul(float* out, const float* a, const float* b, int64_t n) {
#if defined(ODI_HAS_AVX2)
    simd_mul_avx2(out, a, b, n);
#elif defined(ODI_HAS_NEON)
    simd_mul_neon(out, a, b, n);
#else
    simd_mul_fallback(out, a, b, n);
#endif
}

inline float simd_dot(const float* a, const float* b, int64_t n) {
#if defined(ODI_HAS_AVX2)
    return simd_dot_avx2(a, b, n);
#elif defined(ODI_HAS_NEON)
    return simd_dot_neon(a, b, n);
#else
    return simd_dot_fallback(a, b, n);
#endif
}

inline void simd_silu(float* out, const float* x, int64_t n) {
    // SiLU doesn't have a simple SIMD implementation, use fallback
    simd_silu_fallback(out, x, n);
}

inline void simd_gelu(float* out, const float* x, int64_t n) {
    // GELU doesn't have a simple SIMD implementation, use fallback
    simd_gelu_fallback(out, x, n);
}

inline void simd_relu(float* out, const float* x, int64_t n) {
#if defined(ODI_HAS_AVX2)
    simd_relu_avx2(out, x, n);
#elif defined(ODI_HAS_NEON)
    simd_relu_neon(out, x, n);
#else
    simd_relu_fallback(out, x, n);
#endif
}

inline void simd_softmax(float* out, const float* x, int64_t n) {
    // Softmax is complex, use fallback
    simd_softmax_fallback(out, x, n);
}

} // namespace odi

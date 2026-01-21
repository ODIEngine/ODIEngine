#include "cpu_backend.hpp"
#include "simd_ops.hpp"
#include "../../format/gguf_types.hpp"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace odi {

// ============================================================================
// SIMD Capabilities Detection
// ============================================================================

SIMDCapabilities SIMDCapabilities::detect() {
    SIMDCapabilities caps;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 CPUID
    #ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    caps.has_fma = (cpuInfo[2] & (1 << 12)) != 0;
    caps.has_f16c = (cpuInfo[2] & (1 << 29)) != 0;

    __cpuidex(cpuInfo, 7, 0);
    caps.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    caps.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
    #else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps.has_fma = (ecx & (1 << 12)) != 0;
        caps.has_f16c = (ecx & (1 << 29)) != 0;
    }
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps.has_avx2 = (ebx & (1 << 5)) != 0;
        caps.has_avx512 = (ebx & (1 << 16)) != 0;
    }
    #endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    // ARM64 always has NEON
    caps.has_neon = true;
    // Check for FP16 support (typically available on ARMv8.2+)
    #ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    caps.has_neon_fp16 = true;
    #endif
#endif

    return caps;
}

std::string SIMDCapabilities::to_string() const {
    std::string result;
    if (has_avx512) result += "AVX-512 ";
    if (has_avx2) result += "AVX2 ";
    if (has_fma) result += "FMA ";
    if (has_f16c) result += "F16C ";
    if (has_neon) result += "NEON ";
    if (has_neon_fp16) result += "NEON-FP16 ";
    if (result.empty()) result = "None";
    return result;
}

// ============================================================================
// CPUBackend Implementation
// ============================================================================

CPUBackend::CPUBackend(int num_threads) {
    simd_caps_ = SIMDCapabilities::detect();

    if (num_threads <= 0) {
        num_threads_ = static_cast<int>(std::thread::hardware_concurrency());
        if (num_threads_ <= 0) num_threads_ = 4;
    } else {
        num_threads_ = num_threads;
    }
}

CPUBackend::~CPUBackend() = default;

std::string CPUBackend::device_info() const {
    std::string info = "CPU (" + std::to_string(num_threads_) + " threads)\n";
    info += "SIMD: " + simd_caps_.to_string();
    return info;
}

size_t CPUBackend::available_memory() const {
#ifdef __APPLE__
    int64_t mem;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
    return static_cast<size_t>(mem);
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullAvailPhys);
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages * page_size);
#endif
}

size_t CPUBackend::used_memory() const {
    return allocated_memory_.load();
}

bool CPUBackend::supports_dtype(odi_dtype_t dtype) const {
    switch (dtype) {
        case ODI_DTYPE_F32:
        case ODI_DTYPE_F16:
        case ODI_DTYPE_I32:
        case ODI_DTYPE_I16:
        case ODI_DTYPE_I8:
        case ODI_DTYPE_Q4_0:
        case ODI_DTYPE_Q4_1:
        case ODI_DTYPE_Q5_0:
        case ODI_DTYPE_Q5_1:
        case ODI_DTYPE_Q8_0:
        case ODI_DTYPE_Q4_K:
        case ODI_DTYPE_Q5_K:
        case ODI_DTYPE_Q6_K:
        case ODI_DTYPE_Q8_K:
            return true;
        default:
            return false;
    }
}

void CPUBackend::parallel_for(int64_t start, int64_t end, const std::function<void(int64_t, int64_t)>& fn) {
    int64_t total = end - start;
    if (total <= 0) return;

    if (num_threads_ <= 1 || total < num_threads_ * 4) {
        fn(start, end);
        return;
    }

    int64_t chunk_size = (total + num_threads_ - 1) / num_threads_;
    std::vector<std::thread> threads;
    threads.reserve(num_threads_);

    for (int t = 0; t < num_threads_; ++t) {
        int64_t t_start = start + t * chunk_size;
        int64_t t_end = std::min(t_start + chunk_size, end);
        if (t_start >= end) break;

        threads.emplace_back([&fn, t_start, t_end]() {
            fn(t_start, t_end);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

// ============================================================================
// Matrix Multiplication
// ============================================================================

void CPUBackend::matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    // Validate shapes
    if (a.ndim() < 1 || b.ndim() < 1) {
        throw std::runtime_error("matmul: invalid tensor dimensions");
    }

    // Get dimensions
    int M = static_cast<int>(a.shape(-2));
    int K = static_cast<int>(a.shape(-1));
    int N = static_cast<int>(b.shape(-1));

    if (a.ndim() == 1) M = 1;
    if (b.ndim() == 1) {
        N = 1;
        if (K != b.shape(0)) {
            throw std::runtime_error("matmul: dimension mismatch");
        }
    } else if (K != b.shape(-2)) {
        throw std::runtime_error("matmul: dimension mismatch");
    }

    // Handle quantized weights
    if (dtype_is_quantized(b.dtype())) {
        if (a.dtype() != ODI_DTYPE_F32) {
            throw std::runtime_error("matmul: quantized weights require F32 activations");
        }

        switch (b.dtype()) {
            case ODI_DTYPE_Q4_0:
                matmul_q4_0(out.data_ptr<float>(), a.data_ptr<float>(), b.data(), M, K, N);
                return;
            case ODI_DTYPE_Q8_0:
                matmul_q8_0(out.data_ptr<float>(), a.data_ptr<float>(), b.data(), M, K, N);
                return;
            default:
                // Dequantize and do F32 matmul
                Tensor b_f32 = empty({K, N}, ODI_DTYPE_F32);
                dequantize(b_f32, b);
                matmul_f32(out.data_ptr<float>(), a.data_ptr<float>(), b_f32.data_ptr<float>(), M, K, N);
                return;
        }
    }

    // F32 matmul
    if (a.dtype() == ODI_DTYPE_F32 && b.dtype() == ODI_DTYPE_F32) {
        matmul_f32(out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), M, K, N);
        return;
    }

    throw std::runtime_error("matmul: unsupported dtype combination");
}

void CPUBackend::matmul_f32(float* out, const float* a, const float* b, int M, int K, int N) {
#if defined(ODI_HAS_AVX2)
    if (simd_caps_.has_avx2) {
        matmul_f32_avx2(out, a, b, M, K, N);
        return;
    }
#endif

#if defined(ODI_HAS_NEON)
    if (simd_caps_.has_neon) {
        matmul_f32_neon(out, a, b, M, K, N);
        return;
    }
#endif

    // Fallback: naive implementation with cache blocking
    constexpr int BLOCK = 64;

    std::memset(out, 0, M * N * sizeof(float));

    parallel_for(0, M, [&](int64_t m_start, int64_t m_end) {
        for (int64_t m = m_start; m < m_end; ++m) {
            for (int kb = 0; kb < K; kb += BLOCK) {
                int k_end = std::min(kb + BLOCK, K);
                for (int nb = 0; nb < N; nb += BLOCK) {
                    int n_end = std::min(nb + BLOCK, N);
                    for (int k = kb; k < k_end; ++k) {
                        float a_mk = a[m * K + k];
                        for (int n = nb; n < n_end; ++n) {
                            out[m * N + n] += a_mk * b[k * N + n];
                        }
                    }
                }
            }
        }
    });
}

void CPUBackend::batched_matmul(Tensor& out, const Tensor& a, const Tensor& b) {
    // For now, iterate over batches
    int batch_size = static_cast<int>(a.shape(0));
    int M = static_cast<int>(a.shape(1));
    int K = static_cast<int>(a.shape(2));
    int N = static_cast<int>(b.shape(-1));

    bool broadcast_b = (b.ndim() == 2);

    parallel_for(0, batch_size, [&](int64_t start, int64_t end) {
        for (int64_t batch = start; batch < end; ++batch) {
            const float* a_ptr = a.data_ptr<float>() + batch * M * K;
            const float* b_ptr = broadcast_b ? b.data_ptr<float>() : b.data_ptr<float>() + batch * K * N;
            float* out_ptr = out.data_ptr<float>() + batch * M * N;

            // Use inner matmul
            matmul_f32(out_ptr, a_ptr, b_ptr, M, K, N);
        }
    });
}

// ============================================================================
// Element-wise Operations
// ============================================================================

void CPUBackend::add(Tensor& out, const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    // Handle broadcasting (simple case: b is smaller)
    int64_t b_n = b.numel();
    if (b_n == 1) {
        float bv = b_ptr[0];
        parallel_for(0, n, [&](int64_t start, int64_t end) {
            simd_add_scalar(out_ptr + start, a_ptr + start, bv, end - start);
        });
    } else if (b_n == n) {
        parallel_for(0, n, [&](int64_t start, int64_t end) {
            simd_add(out_ptr + start, a_ptr + start, b_ptr + start, end - start);
        });
    } else {
        // General broadcast
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] + b_ptr[i % b_n];
        }
    }
}

void CPUBackend::mul(Tensor& out, const Tensor& a, const Tensor& b) {
    int64_t n = a.numel();
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t b_n = b.numel();
    if (b_n == 1) {
        float bv = b_ptr[0];
        parallel_for(0, n, [&](int64_t start, int64_t end) {
            simd_mul_scalar(out_ptr + start, a_ptr + start, bv, end - start);
        });
    } else if (b_n == n) {
        parallel_for(0, n, [&](int64_t start, int64_t end) {
            simd_mul(out_ptr + start, a_ptr + start, b_ptr + start, end - start);
        });
    } else {
        for (int64_t i = 0; i < n; ++i) {
            out_ptr[i] = a_ptr[i] * b_ptr[i % b_n];
        }
    }
}

void CPUBackend::scale(Tensor& out, const Tensor& a, float scalar) {
    int64_t n = a.numel();
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, n, [&](int64_t start, int64_t end) {
        simd_mul_scalar(out_ptr + start, a_ptr + start, scalar, end - start);
    });
}

// ============================================================================
// Activation Functions
// ============================================================================

void CPUBackend::silu(Tensor& out, const Tensor& x) {
    int64_t n = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, n, [&](int64_t start, int64_t end) {
        simd_silu(out_ptr + start, x_ptr + start, end - start);
    });
}

void CPUBackend::gelu(Tensor& out, const Tensor& x) {
    int64_t n = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, n, [&](int64_t start, int64_t end) {
        simd_gelu(out_ptr + start, x_ptr + start, end - start);
    });
}

void CPUBackend::relu(Tensor& out, const Tensor& x) {
    int64_t n = x.numel();
    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, n, [&](int64_t start, int64_t end) {
        simd_relu(out_ptr + start, x_ptr + start, end - start);
    });
}

// ============================================================================
// Normalization
// ============================================================================

void CPUBackend::rms_norm(Tensor& out, const Tensor& x, const Tensor& weight, float eps) {
    // x: [..., hidden_size]
    // weight: [hidden_size]
    // out: same shape as x

    int64_t hidden_size = x.shape(-1);
    int64_t num_vectors = x.numel() / hidden_size;

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, num_vectors, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const float* xi = x_ptr + i * hidden_size;
            float* oi = out_ptr + i * hidden_size;

            // Compute mean of squares
            float sum_sq = simd_dot(xi, xi, hidden_size);
            float rms = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

            // Normalize and scale by weight
            for (int64_t j = 0; j < hidden_size; ++j) {
                oi[j] = xi[j] * rms * w_ptr[j];
            }
        }
    });
}

void CPUBackend::layer_norm(Tensor& out, const Tensor& x,
                           const Tensor& weight, const Tensor& bias, float eps) {
    int64_t hidden_size = x.shape(-1);
    int64_t num_vectors = x.numel() / hidden_size;

    const float* x_ptr = x.data_ptr<float>();
    const float* w_ptr = weight.data_ptr<float>();
    const float* b_ptr = bias.is_valid() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, num_vectors, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            const float* xi = x_ptr + i * hidden_size;
            float* oi = out_ptr + i * hidden_size;

            // Compute mean
            float sum = 0.0f;
            for (int64_t j = 0; j < hidden_size; ++j) {
                sum += xi[j];
            }
            float mean = sum / hidden_size;

            // Compute variance
            float var_sum = 0.0f;
            for (int64_t j = 0; j < hidden_size; ++j) {
                float d = xi[j] - mean;
                var_sum += d * d;
            }
            float inv_std = 1.0f / std::sqrt(var_sum / hidden_size + eps);

            // Normalize
            for (int64_t j = 0; j < hidden_size; ++j) {
                oi[j] = (xi[j] - mean) * inv_std * w_ptr[j];
                if (b_ptr) oi[j] += b_ptr[j];
            }
        }
    });
}

// ============================================================================
// Attention Operations
// ============================================================================

void CPUBackend::softmax(Tensor& out, const Tensor& x) {
    // Softmax along last dimension
    int64_t last_dim = x.shape(-1);
    int64_t num_vectors = x.numel() / last_dim;

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, num_vectors, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            simd_softmax(out_ptr + i * last_dim, x_ptr + i * last_dim, last_dim);
        }
    });
}

void CPUBackend::masked_softmax(Tensor& out, const Tensor& x, int seq_len) {
    // Apply causal mask then softmax
    int64_t last_dim = x.shape(-1);
    int64_t num_vectors = x.numel() / last_dim;

    const float* x_ptr = x.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    parallel_for(0, num_vectors, [&](int64_t start, int64_t end) {
        std::vector<float> temp(last_dim);
        for (int64_t i = start; i < end; ++i) {
            const float* xi = x_ptr + i * last_dim;
            float* oi = out_ptr + i * last_dim;

            // Apply mask
            int row = static_cast<int>(i % seq_len);
            for (int64_t j = 0; j < last_dim; ++j) {
                temp[j] = (j <= row) ? xi[j] : -std::numeric_limits<float>::infinity();
            }

            simd_softmax(oi, temp.data(), last_dim);
        }
    });
}

void CPUBackend::rope(Tensor& q, Tensor& k, int start_pos, float theta_base) {
    // Apply rotary position embedding
    // q, k: [..., seq_len, head_dim]
    int64_t head_dim = q.shape(-1);
    int64_t seq_len = q.shape(-2);

    float* q_ptr = q.data_ptr<float>();
    float* k_ptr = k.data_ptr<float>();

    int64_t num_heads = q.numel() / (seq_len * head_dim);

    parallel_for(0, num_heads, [&](int64_t h_start, int64_t h_end) {
        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t pos = 0; pos < seq_len; ++pos) {
                int actual_pos = start_pos + static_cast<int>(pos);
                float* qh = q_ptr + h * seq_len * head_dim + pos * head_dim;
                float* kh = k_ptr + h * seq_len * head_dim + pos * head_dim;

                for (int64_t i = 0; i < head_dim / 2; ++i) {
                    float freq = 1.0f / std::pow(theta_base, 2.0f * i / head_dim);
                    float val = actual_pos * freq;
                    float cos_val = std::cos(val);
                    float sin_val = std::sin(val);

                    // Apply rotation to q
                    float q0 = qh[i];
                    float q1 = qh[i + head_dim / 2];
                    qh[i] = q0 * cos_val - q1 * sin_val;
                    qh[i + head_dim / 2] = q0 * sin_val + q1 * cos_val;

                    // Apply rotation to k
                    float k0 = kh[i];
                    float k1 = kh[i + head_dim / 2];
                    kh[i] = k0 * cos_val - k1 * sin_val;
                    kh[i + head_dim / 2] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
    });
}

void CPUBackend::attention(Tensor& out, const Tensor& q, const Tensor& k, const Tensor& v,
                          float scale, bool causal) {
    // Scaled dot-product attention
    // q: [batch, heads, seq_len, head_dim]
    // k, v: [batch, heads, kv_len, head_dim]
    // out: [batch, heads, seq_len, head_dim]

    int batch = static_cast<int>(q.shape(0));
    int heads = static_cast<int>(q.shape(1));
    int seq_len = static_cast<int>(q.shape(2));
    int head_dim = static_cast<int>(q.shape(3));
    int kv_len = static_cast<int>(k.shape(2));

    const float* q_ptr = q.data_ptr<float>();
    const float* k_ptr = k.data_ptr<float>();
    const float* v_ptr = v.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    int64_t total_heads = batch * heads;

    parallel_for(0, total_heads, [&](int64_t start, int64_t end) {
        std::vector<float> scores(seq_len * kv_len);

        for (int64_t bh = start; bh < end; ++bh) {
            const float* qh = q_ptr + bh * seq_len * head_dim;
            const float* kh = k_ptr + bh * kv_len * head_dim;
            const float* vh = v_ptr + bh * kv_len * head_dim;
            float* oh = out_ptr + bh * seq_len * head_dim;

            // Compute Q @ K^T
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < kv_len; ++j) {
                    float dot = simd_dot(qh + i * head_dim, kh + j * head_dim, head_dim);
                    scores[i * kv_len + j] = dot * scale;
                }
            }

            // Apply causal mask and softmax
            for (int i = 0; i < seq_len; ++i) {
                float* row = scores.data() + i * kv_len;

                if (causal) {
                    // Mask future positions
                    for (int j = i + 1; j < kv_len; ++j) {
                        row[j] = -std::numeric_limits<float>::infinity();
                    }
                }

                simd_softmax(row, row, kv_len);
            }

            // Compute scores @ V
            for (int i = 0; i < seq_len; ++i) {
                std::memset(oh + i * head_dim, 0, head_dim * sizeof(float));
                for (int j = 0; j < kv_len; ++j) {
                    float s = scores[i * kv_len + j];
                    for (int d = 0; d < head_dim; ++d) {
                        oh[i * head_dim + d] += s * vh[j * head_dim + d];
                    }
                }
            }
        }
    });
}

// ============================================================================
// Embedding
// ============================================================================

void CPUBackend::embedding(Tensor& out, const Tensor& weights, const int32_t* tokens, int num_tokens) {
    int64_t embed_dim = weights.shape(-1);
    const float* w_ptr = weights.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    for (int i = 0; i < num_tokens; ++i) {
        int32_t token = tokens[i];
        std::memcpy(out_ptr + i * embed_dim, w_ptr + token * embed_dim, embed_dim * sizeof(float));
    }
}

// ============================================================================
// Quantization
// ============================================================================

void CPUBackend::dequantize(Tensor& out, const Tensor& x) {
    int64_t n = x.numel();
    float* out_ptr = out.data_ptr<float>();

    switch (x.dtype()) {
        case ODI_DTYPE_Q4_0:
            dequantize_q4_0(out_ptr, x.data(), n);
            break;
        case ODI_DTYPE_Q4_1:
            dequantize_q4_1(out_ptr, x.data(), n);
            break;
        case ODI_DTYPE_Q5_0:
            dequantize_q5_0(out_ptr, x.data(), n);
            break;
        case ODI_DTYPE_Q5_1:
            dequantize_q5_1(out_ptr, x.data(), n);
            break;
        case ODI_DTYPE_Q8_0:
            dequantize_q8_0(out_ptr, x.data(), n);
            break;
        default:
            throw std::runtime_error("Unsupported quantization type for dequantize");
    }
}

void CPUBackend::quantized_matmul(Tensor& out, const Tensor& a, const Tensor& b_quant) {
    // a is F32, b_quant is quantized
    matmul(out, a, b_quant);
}

// ============================================================================
// Dequantization Implementations
// ============================================================================

// Helper: convert FP16 to FP32
inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);

    if (exp == 0) {
        if (mant == 0) {
            uint32_t f = sign;
            return *reinterpret_cast<float*>(&f);
        }
        exp = 1;
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FF;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp + 127 - 15;
    }

    uint32_t f = sign | (exp << 23) | (mant << 13);
    return *reinterpret_cast<float*>(&f);
}

void CPUBackend::dequantize_q4_0(float* out, const void* x, int64_t n) {
    const BlockQ4_0* blocks = static_cast<const BlockQ4_0*>(x);
    int64_t num_blocks = (n + 31) / 32;

    for (int64_t i = 0; i < num_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);

        for (int j = 0; j < 16; ++j) {
            uint8_t byte = blocks[i].qs[j];
            int q0 = (byte & 0x0F) - 8;
            int q1 = (byte >> 4) - 8;

            out[i * 32 + j] = q0 * d;
            out[i * 32 + j + 16] = q1 * d;
        }
    }
}

void CPUBackend::dequantize_q4_1(float* out, const void* x, int64_t n) {
    const BlockQ4_1* blocks = static_cast<const BlockQ4_1*>(x);
    int64_t num_blocks = (n + 31) / 32;

    for (int64_t i = 0; i < num_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);
        float m = fp16_to_fp32(blocks[i].m);

        for (int j = 0; j < 16; ++j) {
            uint8_t byte = blocks[i].qs[j];
            int q0 = byte & 0x0F;
            int q1 = byte >> 4;

            out[i * 32 + j] = q0 * d + m;
            out[i * 32 + j + 16] = q1 * d + m;
        }
    }
}

void CPUBackend::dequantize_q5_0(float* out, const void* x, int64_t n) {
    const BlockQ5_0* blocks = static_cast<const BlockQ5_0*>(x);
    int64_t num_blocks = (n + 31) / 32;

    for (int64_t i = 0; i < num_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);

        uint32_t qh;
        std::memcpy(&qh, blocks[i].qh, sizeof(qh));

        for (int j = 0; j < 16; ++j) {
            uint8_t byte = blocks[i].qs[j];
            int q0 = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int q1 = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);

            q0 -= 16;
            q1 -= 16;

            out[i * 32 + j] = q0 * d;
            out[i * 32 + j + 16] = q1 * d;
        }
    }
}

void CPUBackend::dequantize_q5_1(float* out, const void* x, int64_t n) {
    const BlockQ5_1* blocks = static_cast<const BlockQ5_1*>(x);
    int64_t num_blocks = (n + 31) / 32;

    for (int64_t i = 0; i < num_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);
        float m = fp16_to_fp32(blocks[i].m);

        uint32_t qh;
        std::memcpy(&qh, blocks[i].qh, sizeof(qh));

        for (int j = 0; j < 16; ++j) {
            uint8_t byte = blocks[i].qs[j];
            int q0 = (byte & 0x0F) | (((qh >> j) & 1) << 4);
            int q1 = (byte >> 4) | (((qh >> (j + 16)) & 1) << 4);

            out[i * 32 + j] = q0 * d + m;
            out[i * 32 + j + 16] = q1 * d + m;
        }
    }
}

void CPUBackend::dequantize_q8_0(float* out, const void* x, int64_t n) {
    const BlockQ8_0* blocks = static_cast<const BlockQ8_0*>(x);
    int64_t num_blocks = (n + 31) / 32;

    for (int64_t i = 0; i < num_blocks; ++i) {
        float d = fp16_to_fp32(blocks[i].d);

        for (int j = 0; j < 32; ++j) {
            out[i * 32 + j] = blocks[i].qs[j] * d;
        }
    }
}

// ============================================================================
// Quantized MatMul
// ============================================================================

void CPUBackend::matmul_q4_0(float* out, const float* a, const void* b, int M, int K, int N) {
    // a: [M, K] F32
    // b: [K, N] Q4_0
    // out: [M, N] F32

    const int block_size = 32;
    int64_t k_blocks = (K + block_size - 1) / block_size;
    const BlockQ4_0* b_blocks = static_cast<const BlockQ4_0*>(b);

    parallel_for(0, M, [&](int64_t m_start, int64_t m_end) {
        std::vector<float> temp(block_size);

        for (int64_t m = m_start; m < m_end; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;

                for (int64_t kb = 0; kb < k_blocks; ++kb) {
                    // Get block for this position
                    const BlockQ4_0& block = b_blocks[n * k_blocks + kb];
                    float d = fp16_to_fp32(block.d);

                    // Dequantize block
                    for (int j = 0; j < 16; ++j) {
                        uint8_t byte = block.qs[j];
                        temp[j] = ((byte & 0x0F) - 8) * d;
                        temp[j + 16] = ((byte >> 4) - 8) * d;
                    }

                    // Dot product with input
                    int k_start = static_cast<int>(kb * block_size);
                    int k_end = std::min(k_start + block_size, K);
                    for (int k = k_start; k < k_end; ++k) {
                        sum += a[m * K + k] * temp[k - k_start];
                    }
                }

                out[m * N + n] = sum;
            }
        }
    });
}

void CPUBackend::matmul_q8_0(float* out, const float* a, const void* b, int M, int K, int N) {
    const int block_size = 32;
    int64_t k_blocks = (K + block_size - 1) / block_size;
    const BlockQ8_0* b_blocks = static_cast<const BlockQ8_0*>(b);

    parallel_for(0, M, [&](int64_t m_start, int64_t m_end) {
        for (int64_t m = m_start; m < m_end; ++m) {
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;

                for (int64_t kb = 0; kb < k_blocks; ++kb) {
                    const BlockQ8_0& block = b_blocks[n * k_blocks + kb];
                    float d = fp16_to_fp32(block.d);

                    int k_start = static_cast<int>(kb * block_size);
                    int k_end = std::min(k_start + block_size, K);

                    for (int k = k_start; k < k_end; ++k) {
                        sum += a[m * K + k] * block.qs[k - k_start] * d;
                    }
                }

                out[m * N + n] = sum;
            }
        }
    });
}

// ============================================================================
// Utility Operations
// ============================================================================

void CPUBackend::copy(Tensor& dst, const Tensor& src) {
    if (dst.nbytes() != src.nbytes()) {
        throw std::runtime_error("copy: size mismatch");
    }
    std::memcpy(dst.data(), src.data(), src.nbytes());
}

void CPUBackend::fill(Tensor& dst, float value) {
    dst.fill(value);
}

void CPUBackend::concat(Tensor& out, const std::vector<Tensor>& inputs, int dim) {
    if (inputs.empty()) return;

    // Simple implementation for last dimension concat
    if (dim == -1 || dim == inputs[0].ndim() - 1) {
        int64_t stride = 1;
        for (int i = 0; i < inputs[0].ndim() - 1; ++i) {
            stride *= inputs[0].shape(i);
        }

        float* out_ptr = out.data_ptr<float>();
        int64_t offset = 0;

        for (int64_t s = 0; s < stride; ++s) {
            for (const auto& input : inputs) {
                int64_t size = input.shape(-1);
                const float* in_ptr = input.data_ptr<float>() + s * size;
                std::memcpy(out_ptr + offset, in_ptr, size * sizeof(float));
                offset += size;
            }
        }
    } else {
        throw std::runtime_error("concat: only last dimension supported currently");
    }
}

void CPUBackend::sync() {
    // CPU is synchronous, nothing to do
}

// ============================================================================
// AVX2 Optimized MatMul
// ============================================================================

#if defined(ODI_HAS_AVX2)
#include <immintrin.h>

void CPUBackend::matmul_f32_avx2(float* out, const float* a, const float* b, int M, int K, int N) {
    constexpr int BLOCK_M = 6;
    constexpr int BLOCK_N = 16;
    constexpr int BLOCK_K = 256;

    std::memset(out, 0, M * N * sizeof(float));

    parallel_for(0, M, [&](int64_t m_start, int64_t m_end) {
        for (int64_t m = m_start; m < m_end; ++m) {
            for (int nb = 0; nb < N; nb += BLOCK_N) {
                int n_end = std::min(nb + BLOCK_N, N);

                for (int kb = 0; kb < K; kb += BLOCK_K) {
                    int k_end = std::min(kb + BLOCK_K, K);

                    for (int k = kb; k < k_end; ++k) {
                        float a_val = a[m * K + k];
                        __m256 va = _mm256_set1_ps(a_val);

                        int n = nb;
                        for (; n + 8 <= n_end; n += 8) {
                            __m256 vb = _mm256_loadu_ps(&b[k * N + n]);
                            __m256 vc = _mm256_loadu_ps(&out[m * N + n]);
                            vc = _mm256_fmadd_ps(va, vb, vc);
                            _mm256_storeu_ps(&out[m * N + n], vc);
                        }

                        // Handle remainder
                        for (; n < n_end; ++n) {
                            out[m * N + n] += a_val * b[k * N + n];
                        }
                    }
                }
            }
        }
    });
}
#else
void CPUBackend::matmul_f32_avx2(float*, const float*, const float*, int, int, int) {
    // Stub for non-AVX2 builds
}
#endif

// ============================================================================
// NEON Optimized MatMul
// ============================================================================

#if defined(ODI_HAS_NEON)
#include <arm_neon.h>

void CPUBackend::matmul_f32_neon(float* out, const float* a, const float* b, int M, int K, int N) {
    std::memset(out, 0, M * N * sizeof(float));

    parallel_for(0, M, [&](int64_t m_start, int64_t m_end) {
        for (int64_t m = m_start; m < m_end; ++m) {
            for (int k = 0; k < K; ++k) {
                float32x4_t va = vdupq_n_f32(a[m * K + k]);

                int n = 0;
                for (; n + 4 <= N; n += 4) {
                    float32x4_t vb = vld1q_f32(&b[k * N + n]);
                    float32x4_t vc = vld1q_f32(&out[m * N + n]);
                    vc = vmlaq_f32(vc, va, vb);
                    vst1q_f32(&out[m * N + n], vc);
                }

                // Handle remainder
                for (; n < N; ++n) {
                    out[m * N + n] += a[m * K + k] * b[k * N + n];
                }
            }
        }
    });
}
#else
void CPUBackend::matmul_f32_neon(float*, const float*, const float*, int, int, int) {
    // Stub for non-NEON builds
}
#endif

// ============================================================================
// Backend Factory
// ============================================================================

std::unique_ptr<Backend> create_backend(odi_backend_type_t type, int num_threads) {
    switch (type) {
        case ODI_BACKEND_AUTO:
        case ODI_BACKEND_CPU:
            return std::make_unique<CPUBackend>(num_threads);

#ifdef ODI_HAS_METAL
        case ODI_BACKEND_METAL:
            // TODO: return std::make_unique<MetalBackend>();
            return std::make_unique<CPUBackend>(num_threads);
#endif

#ifdef ODI_HAS_VULKAN
        case ODI_BACKEND_VULKAN:
            // TODO: return std::make_unique<VulkanBackend>();
            return std::make_unique<CPUBackend>(num_threads);
#endif

#ifdef ODI_HAS_CUDA
        case ODI_BACKEND_CUDA:
            // TODO: return std::make_unique<CUDABackend>();
            return std::make_unique<CPUBackend>(num_threads);
#endif

        default:
            return nullptr;
    }
}

std::vector<odi_backend_type_t> available_backends() {
    std::vector<odi_backend_type_t> backends;
    backends.push_back(ODI_BACKEND_CPU);

#ifdef ODI_HAS_METAL
    backends.push_back(ODI_BACKEND_METAL);
#endif

#ifdef ODI_HAS_VULKAN
    backends.push_back(ODI_BACKEND_VULKAN);
#endif

#ifdef ODI_HAS_CUDA
    backends.push_back(ODI_BACKEND_CUDA);
#endif

    return backends;
}

bool is_backend_available(odi_backend_type_t type) {
    switch (type) {
        case ODI_BACKEND_AUTO:
        case ODI_BACKEND_CPU:
            return true;

#ifdef ODI_HAS_METAL
        case ODI_BACKEND_METAL:
            return true;
#endif

#ifdef ODI_HAS_VULKAN
        case ODI_BACKEND_VULKAN:
            return true;
#endif

#ifdef ODI_HAS_CUDA
        case ODI_BACKEND_CUDA:
            return true;
#endif

        default:
            return false;
    }
}

} // namespace odi

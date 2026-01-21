#pragma once

#include "../backend.hpp"
#include <thread>
#include <vector>
#include <functional>
#include <atomic>

namespace odi {

// SIMD capabilities detected at runtime
struct SIMDCapabilities {
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_fma = false;
    bool has_f16c = false;
    bool has_neon = false;
    bool has_neon_fp16 = false;

    static SIMDCapabilities detect();
    std::string to_string() const;
};

/**
 * CPU Backend
 *
 * Uses SIMD instructions (AVX2, AVX-512, NEON) for acceleration.
 * Supports quantized operations for INT4/INT8.
 */
class CPUBackend : public Backend {
public:
    explicit CPUBackend(int num_threads = 0);
    ~CPUBackend() override;

    // Backend interface
    odi_backend_type_t type() const override { return ODI_BACKEND_CPU; }
    std::string name() const override { return "CPU"; }
    std::string device_info() const override;

    size_t available_memory() const override;
    size_t used_memory() const override;

    bool supports_dtype(odi_dtype_t dtype) const override;

    // Core operations
    void matmul(Tensor& out, const Tensor& a, const Tensor& b) override;
    void batched_matmul(Tensor& out, const Tensor& a, const Tensor& b) override;
    void add(Tensor& out, const Tensor& a, const Tensor& b) override;
    void mul(Tensor& out, const Tensor& a, const Tensor& b) override;
    void scale(Tensor& out, const Tensor& a, float scalar) override;

    // Activations
    void silu(Tensor& out, const Tensor& x) override;
    void gelu(Tensor& out, const Tensor& x) override;
    void relu(Tensor& out, const Tensor& x) override;

    // Normalization
    void rms_norm(Tensor& out, const Tensor& x, const Tensor& weight, float eps) override;
    void layer_norm(Tensor& out, const Tensor& x,
                   const Tensor& weight, const Tensor& bias, float eps) override;

    // Attention
    void softmax(Tensor& out, const Tensor& x) override;
    void masked_softmax(Tensor& out, const Tensor& x, int seq_len) override;
    void rope(Tensor& q, Tensor& k, int start_pos, float theta_base) override;
    void attention(Tensor& out, const Tensor& q, const Tensor& k, const Tensor& v,
                  float scale, bool causal) override;

    // Embedding
    void embedding(Tensor& out, const Tensor& weights, const int32_t* tokens, int num_tokens) override;

    // Quantization
    void dequantize(Tensor& out, const Tensor& x) override;
    void quantized_matmul(Tensor& out, const Tensor& a, const Tensor& b_quant) override;

    // Utility
    void copy(Tensor& dst, const Tensor& src) override;
    void fill(Tensor& dst, float value) override;
    void concat(Tensor& out, const std::vector<Tensor>& inputs, int dim) override;
    void sync() override;

    // Thread pool access
    int num_threads() const { return num_threads_; }
    const SIMDCapabilities& simd_caps() const { return simd_caps_; }

    // Parallel execution helper
    void parallel_for(int64_t start, int64_t end, const std::function<void(int64_t, int64_t)>& fn);

private:
    int num_threads_;
    SIMDCapabilities simd_caps_;
    std::atomic<size_t> allocated_memory_{0};

    // Internal implementation functions
    void matmul_f32(float* out, const float* a, const float* b, int M, int K, int N);
    void matmul_f32_avx2(float* out, const float* a, const float* b, int M, int K, int N);
    void matmul_f32_neon(float* out, const float* a, const float* b, int M, int K, int N);

    void matmul_q4_0(float* out, const float* a, const void* b, int M, int K, int N);
    void matmul_q8_0(float* out, const float* a, const void* b, int M, int K, int N);

    void dequantize_q4_0(float* out, const void* x, int64_t n);
    void dequantize_q4_1(float* out, const void* x, int64_t n);
    void dequantize_q5_0(float* out, const void* x, int64_t n);
    void dequantize_q5_1(float* out, const void* x, int64_t n);
    void dequantize_q8_0(float* out, const void* x, int64_t n);
};

} // namespace odi

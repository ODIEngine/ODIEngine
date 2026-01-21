#pragma once

#include "../tensor/tensor.hpp"
#include <odi/odi_types.h>
#include <memory>
#include <string>

namespace odi {

/**
 * Backend - Abstract interface for compute backends
 *
 * Provides operations needed for LLM inference.
 * Implementations: CPUBackend, MetalBackend, VulkanBackend, CUDABackend
 */
class Backend {
public:
    virtual ~Backend() = default;

    // Backend info
    virtual odi_backend_type_t type() const = 0;
    virtual std::string name() const = 0;
    virtual std::string device_info() const = 0;

    // Memory management
    virtual size_t available_memory() const = 0;
    virtual size_t used_memory() const = 0;

    // Check if backend supports a dtype
    virtual bool supports_dtype(odi_dtype_t dtype) const = 0;

    // ========================================================================
    // Core tensor operations
    // ========================================================================

    // Matrix multiplication: out = a @ b
    // a: [M, K], b: [K, N] -> out: [M, N]
    virtual void matmul(Tensor& out, const Tensor& a, const Tensor& b) = 0;

    // Batched matrix multiplication: out = a @ b
    // a: [B, M, K], b: [B, K, N] or [K, N] -> out: [B, M, N]
    virtual void batched_matmul(Tensor& out, const Tensor& a, const Tensor& b) = 0;

    // Element-wise addition: out = a + b
    virtual void add(Tensor& out, const Tensor& a, const Tensor& b) = 0;

    // Element-wise multiplication: out = a * b
    virtual void mul(Tensor& out, const Tensor& a, const Tensor& b) = 0;

    // Scale: out = a * scalar
    virtual void scale(Tensor& out, const Tensor& a, float scalar) = 0;

    // ========================================================================
    // Activation functions
    // ========================================================================

    // SiLU (Swish): out = x * sigmoid(x)
    virtual void silu(Tensor& out, const Tensor& x) = 0;

    // GELU: out = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    virtual void gelu(Tensor& out, const Tensor& x) = 0;

    // ReLU: out = max(0, x)
    virtual void relu(Tensor& out, const Tensor& x) = 0;

    // ========================================================================
    // Normalization
    // ========================================================================

    // RMS Normalization: out = x / sqrt(mean(x^2) + eps) * weight
    virtual void rms_norm(Tensor& out, const Tensor& x, const Tensor& weight, float eps) = 0;

    // Layer Normalization: out = (x - mean) / sqrt(var + eps) * weight + bias
    virtual void layer_norm(Tensor& out, const Tensor& x,
                           const Tensor& weight, const Tensor& bias, float eps) = 0;

    // ========================================================================
    // Attention operations
    // ========================================================================

    // Softmax along last dimension
    virtual void softmax(Tensor& out, const Tensor& x) = 0;

    // Masked softmax for causal attention
    virtual void masked_softmax(Tensor& out, const Tensor& x, int seq_len) = 0;

    // Rotary Position Embedding
    // Applies RoPE to query and key tensors
    virtual void rope(Tensor& q, Tensor& k, int start_pos, float theta_base) = 0;

    // Scaled dot-product attention
    // q: [batch, heads, seq_len, head_dim]
    // k: [batch, heads, kv_len, head_dim]
    // v: [batch, heads, kv_len, head_dim]
    // out: [batch, heads, seq_len, head_dim]
    virtual void attention(Tensor& out, const Tensor& q, const Tensor& k, const Tensor& v,
                          float scale, bool causal) = 0;

    // ========================================================================
    // Embedding operations
    // ========================================================================

    // Token embedding lookup
    virtual void embedding(Tensor& out, const Tensor& weights, const int32_t* tokens, int num_tokens) = 0;

    // ========================================================================
    // Quantization operations
    // ========================================================================

    // Dequantize tensor to FP32
    virtual void dequantize(Tensor& out, const Tensor& x) = 0;

    // Quantized matrix multiplication (keeps weights quantized)
    virtual void quantized_matmul(Tensor& out, const Tensor& a, const Tensor& b_quant) = 0;

    // ========================================================================
    // Utility operations
    // ========================================================================

    // Copy tensor
    virtual void copy(Tensor& dst, const Tensor& src) = 0;

    // Fill tensor with value
    virtual void fill(Tensor& dst, float value) = 0;

    // Concatenate tensors along dimension
    virtual void concat(Tensor& out, const std::vector<Tensor>& inputs, int dim) = 0;

    // Synchronize (wait for all operations to complete)
    virtual void sync() = 0;
};

/**
 * Create a backend by type
 *
 * @param type  Backend type (ODI_BACKEND_AUTO selects best available)
 * @param num_threads  Number of threads for CPU backend (0 = auto)
 * @return  Backend instance, or nullptr if not available
 */
std::unique_ptr<Backend> create_backend(odi_backend_type_t type, int num_threads = 0);

/**
 * Get list of available backends
 */
std::vector<odi_backend_type_t> available_backends();

/**
 * Check if a backend is available
 */
bool is_backend_available(odi_backend_type_t type);

} // namespace odi

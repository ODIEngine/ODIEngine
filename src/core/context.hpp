#pragma once

#include "../tensor/tensor.hpp"
#include <odi/odi_types.h>

#include <vector>
#include <memory>

namespace odi {

// Forward declarations
class Model;
class Backend;

/**
 * Context - Inference context with KV cache
 *
 * Each context maintains its own KV cache and can generate independently.
 */
class Context {
public:
    Context(Model* model, const odi_context_config_t& config);
    ~Context();

    // Prevent copying
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Model access
    Model* model() { return model_; }
    const Model* model() const { return model_; }

    // Configuration
    const odi_context_config_t& config() const { return config_; }
    size_t max_context_length() const { return config_.context_length; }
    size_t batch_size() const { return config_.batch_size; }

    // Current state
    int current_position() const { return current_pos_; }
    int num_tokens_evaluated() const { return tokens_evaluated_; }

    // Reset context (clear KV cache)
    void reset();

    // Evaluate tokens (add to context)
    odi_error_t eval(const int32_t* tokens, int num_tokens);

    // Get logits for sampling (after eval)
    const float* get_logits() const;

    // Get embeddings (if supported)
    const float* get_embeddings() const;

    // KV cache management
    bool has_kv_cache() const { return !kv_cache_k_.empty(); }
    void clear_kv_cache();

    // Internal: forward pass through model
    odi_error_t forward(const int32_t* tokens, int num_tokens, bool compute_logits);

private:
    Model* model_;
    odi_context_config_t config_;

    int current_pos_ = 0;
    int tokens_evaluated_ = 0;

    // Output tensors
    Tensor logits_;
    Tensor embeddings_;

    // KV cache: [layer][batch, heads, seq, head_dim]
    std::vector<Tensor> kv_cache_k_;
    std::vector<Tensor> kv_cache_v_;

    // Working buffers
    Tensor hidden_state_;
    Tensor attention_out_;
    Tensor ffn_out_;

    void allocate_kv_cache();
    void allocate_working_buffers();

    // Layer forward passes
    void attention_layer(int layer, const Tensor& input, Tensor& output, int pos);
    void ffn_layer(int layer, const Tensor& input, Tensor& output);
};

} // namespace odi

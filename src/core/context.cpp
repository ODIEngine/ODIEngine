#include "context.hpp"
#include "model.hpp"
#include "engine.hpp"
#include "../compute/backend.hpp"

#include <cstring>

namespace odi {

Context::Context(Model* model, const odi_context_config_t& config)
    : model_(model), config_(config) {

    // Validate context length
    if (config_.context_length == 0) {
        config_.context_length = static_cast<size_t>(model->context_length());
    }

    if (config_.context_length > static_cast<size_t>(model->context_length())) {
        config_.context_length = static_cast<size_t>(model->context_length());
    }

    // Allocate working memory
    allocate_kv_cache();
    allocate_working_buffers();
}

Context::~Context() = default;

void Context::reset() {
    current_pos_ = 0;
    tokens_evaluated_ = 0;
    clear_kv_cache();
}

void Context::clear_kv_cache() {
    for (auto& k : kv_cache_k_) {
        k.zero();
    }
    for (auto& v : kv_cache_v_) {
        v.zero();
    }
}

void Context::allocate_kv_cache() {
    int64_t num_layers = model_->num_layers();
    int64_t num_kv_heads = model_->num_kv_heads();
    int64_t head_dim = model_->head_dim();
    int64_t max_seq = static_cast<int64_t>(config_.context_length);
    int64_t batch = static_cast<int64_t>(config_.batch_size);

    kv_cache_k_.resize(num_layers);
    kv_cache_v_.resize(num_layers);

    for (int64_t i = 0; i < num_layers; ++i) {
        kv_cache_k_[i] = zeros({batch, num_kv_heads, max_seq, head_dim});
        kv_cache_v_[i] = zeros({batch, num_kv_heads, max_seq, head_dim});
    }
}

void Context::allocate_working_buffers() {
    int64_t embed_dim = model_->embedding_dim();
    int64_t vocab_size = model_->vocab_size();
    int64_t max_seq = static_cast<int64_t>(config_.context_length);
    int64_t batch = static_cast<int64_t>(config_.batch_size);

    hidden_state_ = zeros({batch, max_seq, embed_dim});
    attention_out_ = zeros({batch, max_seq, embed_dim});
    ffn_out_ = zeros({batch, max_seq, embed_dim});
    logits_ = zeros({batch, max_seq, vocab_size});
}

odi_error_t Context::eval(const int32_t* tokens, int num_tokens) {
    return forward(tokens, num_tokens, true);
}

const float* Context::get_logits() const {
    return logits_.data_ptr<float>();
}

const float* Context::get_embeddings() const {
    return hidden_state_.data_ptr<float>();
}

odi_error_t Context::forward(const int32_t* tokens, int num_tokens, bool compute_logits) {
    if (num_tokens <= 0) {
        return ODI_SUCCESS;
    }

    // Check context overflow
    if (current_pos_ + num_tokens > static_cast<int>(config_.context_length)) {
        odi_set_error(ODI_ERROR_CONTEXT_TOO_LONG, "Context length exceeded");
        return ODI_ERROR_CONTEXT_TOO_LONG;
    }

    Backend* backend = model_->engine()->backend();

    // Get embeddings
    Tensor token_embeds = model_->get_tensor("token_embd.weight");
    if (!token_embeds.is_valid()) {
        odi_set_error(ODI_ERROR_TENSOR_NOT_FOUND, "Token embedding not found");
        return ODI_ERROR_TENSOR_NOT_FOUND;
    }

    // Lookup token embeddings
    int64_t embed_dim = model_->embedding_dim();
    Tensor input_embeds = zeros({1, static_cast<int64_t>(num_tokens), embed_dim});

    backend->embedding(input_embeds, token_embeds, tokens, num_tokens);

    // Copy to working buffer at current position
    float* hidden_ptr = hidden_state_.data_ptr<float>() + current_pos_ * embed_dim;
    std::memcpy(hidden_ptr, input_embeds.data(), num_tokens * embed_dim * sizeof(float));

    // Process through layers
    int64_t num_layers = model_->num_layers();

    for (int64_t layer = 0; layer < num_layers; ++layer) {
        // Create view for current sequence
        Tensor layer_input = hidden_state_.slice(1, current_pos_, current_pos_ + num_tokens);

        // Self-attention with KV cache
        attention_layer(static_cast<int>(layer), layer_input, attention_out_, current_pos_);

        // Add residual
        backend->add(layer_input, layer_input, attention_out_.slice(1, 0, num_tokens));

        // FFN
        ffn_layer(static_cast<int>(layer), layer_input, ffn_out_);

        // Add residual
        backend->add(layer_input, layer_input, ffn_out_.slice(1, 0, num_tokens));
    }

    // Final norm
    Tensor norm_weight = model_->get_tensor("output_norm.weight");
    if (norm_weight.is_valid()) {
        Tensor final_hidden = hidden_state_.slice(1, current_pos_, current_pos_ + num_tokens);
        backend->rms_norm(final_hidden, final_hidden, norm_weight, model_->rms_norm_eps());
    }

    // Compute logits if requested
    if (compute_logits) {
        Tensor output_weight = model_->get_tensor("output.weight");
        if (!output_weight.is_valid()) {
            // Some models tie embeddings
            output_weight = token_embeds;
        }

        Tensor final_hidden = hidden_state_.slice(1, current_pos_ + num_tokens - 1, current_pos_ + num_tokens);
        Tensor logits_view = logits_.slice(1, 0, 1);

        backend->matmul(logits_view, final_hidden, output_weight);
    }

    current_pos_ += num_tokens;
    tokens_evaluated_ += num_tokens;

    return ODI_SUCCESS;
}

void Context::attention_layer(int layer, const Tensor& input, Tensor& output, int pos) {
    Backend* backend = model_->engine()->backend();

    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Load weights
    Tensor attn_norm = model_->get_tensor(prefix + "attn_norm.weight");
    Tensor wq = model_->get_tensor(prefix + "attn_q.weight");
    Tensor wk = model_->get_tensor(prefix + "attn_k.weight");
    Tensor wv = model_->get_tensor(prefix + "attn_v.weight");
    Tensor wo = model_->get_tensor(prefix + "attn_output.weight");

    int64_t seq_len = input.shape(1);
    int64_t embed_dim = model_->embedding_dim();
    int64_t num_heads = model_->num_heads();
    int64_t num_kv_heads = model_->num_kv_heads();
    int64_t head_dim = model_->head_dim();

    // Pre-attention norm
    Tensor normed = zeros(input.shape_vec());
    backend->rms_norm(normed, input, attn_norm, model_->rms_norm_eps());

    // Project to Q, K, V
    Tensor q = zeros({1, seq_len, embed_dim});
    Tensor k = zeros({1, seq_len, num_kv_heads * head_dim});
    Tensor v = zeros({1, seq_len, num_kv_heads * head_dim});

    backend->matmul(q, normed, wq);
    backend->matmul(k, normed, wk);
    backend->matmul(v, normed, wv);

    // Reshape for multi-head attention
    q = q.reshape({1, seq_len, num_heads, head_dim}).transpose(1, 2);
    k = k.reshape({1, seq_len, num_kv_heads, head_dim}).transpose(1, 2);
    v = v.reshape({1, seq_len, num_kv_heads, head_dim}).transpose(1, 2);

    // Apply RoPE
    backend->rope(q, k, pos, model_->rope_theta());

    // Update KV cache
    Tensor& cache_k = kv_cache_k_[layer];
    Tensor& cache_v = kv_cache_v_[layer];

    // Copy new K, V to cache
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            float* cache_k_ptr = cache_k.data_ptr<float>() + h * config_.context_length * head_dim + (pos + i) * head_dim;
            float* cache_v_ptr = cache_v.data_ptr<float>() + h * config_.context_length * head_dim + (pos + i) * head_dim;
            const float* k_ptr = k.data_ptr<float>() + h * seq_len * head_dim + i * head_dim;
            const float* v_ptr = v.data_ptr<float>() + h * seq_len * head_dim + i * head_dim;
            std::memcpy(cache_k_ptr, k_ptr, head_dim * sizeof(float));
            std::memcpy(cache_v_ptr, v_ptr, head_dim * sizeof(float));
        }
    }

    // Attention with cached K, V
    int kv_len = pos + static_cast<int>(seq_len);
    Tensor cached_k = cache_k.slice(2, 0, kv_len);
    Tensor cached_v = cache_v.slice(2, 0, kv_len);

    // Handle GQA (grouped query attention)
    if (num_heads != num_kv_heads) {
        // Repeat K, V for each head group
        int groups = static_cast<int>(num_heads / num_kv_heads);
        // For simplicity, expand in place (inefficient but correct)
        Tensor expanded_k = zeros({1, num_heads, static_cast<int64_t>(kv_len), head_dim});
        Tensor expanded_v = zeros({1, num_heads, static_cast<int64_t>(kv_len), head_dim});

        for (int64_t h = 0; h < num_heads; ++h) {
            int64_t kv_h = h / groups;
            std::memcpy(expanded_k.data_ptr<float>() + h * kv_len * head_dim,
                       cached_k.data_ptr<float>() + kv_h * kv_len * head_dim,
                       kv_len * head_dim * sizeof(float));
            std::memcpy(expanded_v.data_ptr<float>() + h * kv_len * head_dim,
                       cached_v.data_ptr<float>() + kv_h * kv_len * head_dim,
                       kv_len * head_dim * sizeof(float));
        }
        cached_k = expanded_k;
        cached_v = expanded_v;
    }

    // Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor attn_out = zeros({1, num_heads, seq_len, head_dim});
    backend->attention(attn_out, q, cached_k, cached_v, scale, true);

    // Reshape and project output
    attn_out = attn_out.transpose(1, 2).reshape({1, seq_len, embed_dim});
    backend->matmul(output.slice(1, 0, seq_len), attn_out, wo);
}

void Context::ffn_layer(int layer, const Tensor& input, Tensor& output) {
    Backend* backend = model_->engine()->backend();

    std::string prefix = "blk." + std::to_string(layer) + ".";

    // Load weights
    Tensor ffn_norm = model_->get_tensor(prefix + "ffn_norm.weight");
    Tensor w_gate = model_->get_tensor(prefix + "ffn_gate.weight");
    Tensor w_up = model_->get_tensor(prefix + "ffn_up.weight");
    Tensor w_down = model_->get_tensor(prefix + "ffn_down.weight");

    int64_t seq_len = input.shape(1);
    int64_t ffn_dim = model_->ffn_dim();

    // Pre-FFN norm
    Tensor normed = zeros(input.shape_vec());
    backend->rms_norm(normed, input, ffn_norm, model_->rms_norm_eps());

    // SwiGLU: gate * silu(up)
    Tensor gate_out = zeros({1, seq_len, ffn_dim});
    Tensor up_out = zeros({1, seq_len, ffn_dim});

    backend->matmul(gate_out, normed, w_gate);
    backend->matmul(up_out, normed, w_up);

    backend->silu(gate_out, gate_out);
    backend->mul(gate_out, gate_out, up_out);

    // Down projection
    backend->matmul(output.slice(1, 0, seq_len), gate_out, w_down);
}

} // namespace odi

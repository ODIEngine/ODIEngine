#pragma once

#include "../format/gguf_parser.hpp"
#include "../tensor/tensor.hpp"
#include <odi/odi_types.h>

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace odi {

// Forward declarations
class Engine;
class Context;

/**
 * Model - Loaded GGUF model
 *
 * Contains model weights, configuration, and tokenizer.
 */
class Model {
public:
    Model(Engine* engine, const std::string& path, const odi_model_config_t& config,
          odi_progress_callback_t progress_callback = nullptr, void* user_data = nullptr);
    ~Model();

    // Prevent copying
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Model info
    const std::string& path() const { return path_; }
    const GGUFModelInfo& info() const { return gguf_->model_info(); }
    odi_arch_type_t arch() const { return info().get_arch_type(); }
    const std::string& name() const { return info().name; }

    // Architecture parameters
    int64_t vocab_size() const { return info().vocab_size; }
    int64_t context_length() const { return info().context_length; }
    int64_t embedding_dim() const { return info().embedding_length; }
    int64_t num_layers() const { return info().block_count; }
    int64_t num_heads() const { return info().attention_head_count; }
    int64_t num_kv_heads() const { return info().attention_head_count_kv; }
    int64_t head_dim() const { return embedding_dim() / num_heads(); }
    int64_t ffn_dim() const { return info().feed_forward_length; }
    float rms_norm_eps() const { return info().attention_layer_norm_rms_epsilon; }
    float rope_theta() const { return info().rope_freq_base; }

    // Special tokens
    int32_t bos_token_id() const { return info().bos_token_id; }
    int32_t eos_token_id() const { return info().eos_token_id; }
    int32_t pad_token_id() const { return info().pad_token_id; }
    int32_t unk_token_id() const { return info().unk_token_id; }

    // Tokenization
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = true) const;
    std::string detokenize(const std::vector<int32_t>& tokens) const;
    std::string token_to_string(int32_t token_id) const;

    // Tensor access
    bool has_tensor(const std::string& name) const;
    Tensor get_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    // Engine access
    Engine* engine() { return engine_; }
    const Engine* engine() const { return engine_; }

    // Get model info as JSON
    std::string get_info_json() const;

private:
    Engine* engine_;
    std::string path_;
    odi_model_config_t config_;

    std::unique_ptr<GGUFFile> gguf_;

    // Vocabulary for tokenization
    std::unordered_map<std::string, int32_t> vocab_to_id_;
    std::vector<std::string> id_to_vocab_;
    std::vector<float> vocab_scores_;

    // BPE merges
    std::unordered_map<std::pair<std::string, std::string>, int, struct PairHash> merges_;

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };

    void load_vocab();
    void load_merges();

    // BPE tokenization
    std::vector<std::string> bpe_tokenize(const std::string& text) const;
};

} // namespace odi

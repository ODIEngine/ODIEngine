#include "model.hpp"
#include "engine.hpp"

#include <sstream>
#include <algorithm>
#include <regex>
#include <codecvt>
#include <locale>

namespace odi {

Model::Model(Engine* engine, const std::string& path, const odi_model_config_t& config,
             odi_progress_callback_t progress_callback, void* user_data)
    : engine_(engine), path_(path), config_(config) {

    // Report progress
    if (progress_callback) {
        progress_callback(0.0f, user_data);
    }

    // Open GGUF file
    gguf_ = GGUFFile::open(path, config.use_mmap);
    if (!gguf_) {
        throw std::runtime_error("Failed to load model: " + path);
    }

    if (progress_callback) {
        progress_callback(0.5f, user_data);
    }

    // Load vocabulary and merges
    load_vocab();
    load_merges();

    if (progress_callback) {
        progress_callback(1.0f, user_data);
    }
}

Model::~Model() = default;

void Model::load_vocab() {
    const auto& tokens = gguf_->vocab_tokens();
    const auto& scores = gguf_->vocab_scores();

    id_to_vocab_ = tokens;
    vocab_scores_ = scores;

    for (size_t i = 0; i < tokens.size(); ++i) {
        vocab_to_id_[tokens[i]] = static_cast<int32_t>(i);
    }
}

void Model::load_merges() {
    const auto& merges_str = gguf_->vocab_merges();

    for (size_t i = 0; i < merges_str.size(); ++i) {
        const std::string& merge = merges_str[i];
        size_t space_pos = merge.find(' ');
        if (space_pos != std::string::npos) {
            std::string first = merge.substr(0, space_pos);
            std::string second = merge.substr(space_pos + 1);
            merges_[{first, second}] = static_cast<int>(i);
        }
    }
}

std::vector<int32_t> Model::tokenize(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;

    if (add_bos && bos_token_id() >= 0) {
        tokens.push_back(bos_token_id());
    }

    // Simple word-level tokenization with BPE
    std::vector<std::string> words = bpe_tokenize(text);

    for (const auto& word : words) {
        auto it = vocab_to_id_.find(word);
        if (it != vocab_to_id_.end()) {
            tokens.push_back(it->second);
        } else if (unk_token_id() >= 0) {
            tokens.push_back(unk_token_id());
        }
    }

    return tokens;
}

std::string Model::detokenize(const std::vector<int32_t>& tokens) const {
    std::string result;

    for (int32_t token : tokens) {
        if (token >= 0 && token < static_cast<int32_t>(id_to_vocab_.size())) {
            std::string piece = id_to_vocab_[token];

            // Handle special tokens
            if (token == bos_token_id() || token == eos_token_id() ||
                token == pad_token_id()) {
                continue;
            }

            // Handle SentencePiece-style space encoding
            if (piece.size() >= 3 && piece[0] == '\xe2' && piece[1] == '\x96' && piece[2] == '\x81') {
                result += ' ';
                piece = piece.substr(3);
            }

            // Replace special characters
            std::string decoded;
            for (size_t i = 0; i < piece.size(); ++i) {
                if (piece[i] == '\xc4' && i + 1 < piece.size() && piece[i+1] == '\xa0') {
                    decoded += ' ';
                    i++;
                } else if (piece[i] == '\xc4' && i + 1 < piece.size() && piece[i+1] == '\x82') {
                    decoded += '\n';
                    i++;
                } else {
                    decoded += piece[i];
                }
            }

            result += decoded;
        }
    }

    return result;
}

std::string Model::token_to_string(int32_t token_id) const {
    if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_vocab_.size())) {
        return id_to_vocab_[token_id];
    }
    return "";
}

std::vector<std::string> Model::bpe_tokenize(const std::string& text) const {
    std::vector<std::string> tokens;

    // Split text into words (simple whitespace split)
    std::string current_word;
    for (char c : text) {
        if (std::isspace(c)) {
            if (!current_word.empty()) {
                // Prepend space marker for SentencePiece compatibility
                std::string marked = "\xe2\x96\x81" + current_word;

                // Apply BPE merges
                std::vector<std::string> subwords;
                subwords.push_back(marked);

                // Simple BPE: try to find vocab matches
                auto it = vocab_to_id_.find(marked);
                if (it != vocab_to_id_.end()) {
                    tokens.push_back(marked);
                } else {
                    // Fall back to character-level
                    for (char ch : current_word) {
                        std::string s(1, ch);
                        if (vocab_to_id_.count(s)) {
                            tokens.push_back(s);
                        } else if (unk_token_id() >= 0) {
                            tokens.push_back(id_to_vocab_[unk_token_id()]);
                        }
                    }
                }

                current_word.clear();
            }
        } else {
            current_word += c;
        }
    }

    // Handle last word
    if (!current_word.empty()) {
        std::string marked = "\xe2\x96\x81" + current_word;
        auto it = vocab_to_id_.find(marked);
        if (it != vocab_to_id_.end()) {
            tokens.push_back(marked);
        } else {
            for (char ch : current_word) {
                std::string s(1, ch);
                if (vocab_to_id_.count(s)) {
                    tokens.push_back(s);
                }
            }
        }
    }

    return tokens;
}

bool Model::has_tensor(const std::string& name) const {
    return gguf_->has_tensor(name);
}

Tensor Model::get_tensor(const std::string& name) const {
    return gguf_->load_tensor(name);
}

std::vector<std::string> Model::tensor_names() const {
    return gguf_->tensor_names();
}

std::string Model::get_info_json() const {
    std::ostringstream ss;
    const auto& info = this->info();

    ss << "{\n";
    ss << "  \"path\": \"" << path_ << "\",\n";
    ss << "  \"architecture\": \"" << info.architecture << "\",\n";
    ss << "  \"name\": \"" << info.name << "\",\n";
    ss << "  \"context_length\": " << info.context_length << ",\n";
    ss << "  \"embedding_dim\": " << info.embedding_length << ",\n";
    ss << "  \"num_layers\": " << info.block_count << ",\n";
    ss << "  \"num_heads\": " << info.attention_head_count << ",\n";
    ss << "  \"num_kv_heads\": " << info.attention_head_count_kv << ",\n";
    ss << "  \"vocab_size\": " << info.vocab_size << ",\n";
    ss << "  \"rope_theta\": " << info.rope_freq_base << ",\n";
    ss << "  \"rms_norm_eps\": " << info.attention_layer_norm_rms_epsilon << "\n";
    ss << "}";

    return ss.str();
}

} // namespace odi

#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace odi {

// Forward declaration
class Model;

/**
 * Tokenizer - Text tokenization utilities
 *
 * Wraps model's vocabulary for encoding/decoding text.
 */
class Tokenizer {
public:
    explicit Tokenizer(const Model* model);
    ~Tokenizer();

    // Encode text to tokens
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const;

    // Decode tokens to text
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Decode a single token
    std::string decode_token(int32_t token) const;

    // Get vocabulary size
    int vocab_size() const;

    // Get special token IDs
    int32_t bos_id() const;
    int32_t eos_id() const;
    int32_t pad_id() const;

private:
    const Model* model_;
};

} // namespace odi

#include "tokenizer.hpp"
#include "model.hpp"

namespace odi {

Tokenizer::Tokenizer(const Model* model) : model_(model) {}

Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    return model_->tokenize(text, add_bos);
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    return model_->detokenize(tokens);
}

std::string Tokenizer::decode_token(int32_t token) const {
    return model_->token_to_string(token);
}

int Tokenizer::vocab_size() const {
    return static_cast<int>(model_->vocab_size());
}

int32_t Tokenizer::bos_id() const {
    return model_->bos_token_id();
}

int32_t Tokenizer::eos_id() const {
    return model_->eos_token_id();
}

int32_t Tokenizer::pad_id() const {
    return model_->pad_token_id();
}

} // namespace odi

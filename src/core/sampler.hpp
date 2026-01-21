#pragma once

#include <odi/odi_types.h>
#include <cstdint>
#include <random>
#include <vector>

namespace odi {

/**
 * Sampler - Token sampling strategies
 *
 * Implements various sampling methods: greedy, temperature, top-k, top-p, mirostat.
 */
class Sampler {
public:
    explicit Sampler(const odi_sampler_config_t& config);
    ~Sampler();

    // Reset sampler state
    void reset();

    // Sample next token from logits
    // logits: [vocab_size] array of log probabilities
    int32_t sample(const float* logits, int vocab_size);

    // Get/set configuration
    const odi_sampler_config_t& config() const { return config_; }
    void set_config(const odi_sampler_config_t& config);

    // Set random seed
    void set_seed(uint64_t seed);

private:
    odi_sampler_config_t config_;
    std::mt19937_64 rng_;

    // Mirostat state
    float mirostat_mu_ = 0.0f;

    // Recent tokens for repetition penalty
    std::vector<int32_t> recent_tokens_;

    // Internal sampling methods
    int32_t sample_greedy(const float* logits, int vocab_size);
    int32_t sample_temperature(float* logits, int vocab_size);
    int32_t sample_top_k(float* logits, int vocab_size, int k);
    int32_t sample_top_p(float* logits, int vocab_size, float p);
    int32_t sample_mirostat_v2(float* logits, int vocab_size);

    // Apply repetition penalty
    void apply_repetition_penalty(float* logits, int vocab_size);

    // Softmax normalization
    void softmax(float* logits, int vocab_size);
};

} // namespace odi

#include "sampler.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>

namespace odi {

Sampler::Sampler(const odi_sampler_config_t& config) : config_(config) {
    // Initialize RNG
    if (config.seed == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        set_seed(now.time_since_epoch().count());
    } else {
        set_seed(config.seed);
    }

    mirostat_mu_ = 2.0f * config.mirostat_tau;
}

Sampler::~Sampler() = default;

void Sampler::reset() {
    recent_tokens_.clear();
    mirostat_mu_ = 2.0f * config_.mirostat_tau;
}

void Sampler::set_config(const odi_sampler_config_t& config) {
    config_ = config;
    if (config.seed != 0) {
        set_seed(config.seed);
    }
}

void Sampler::set_seed(uint64_t seed) {
    rng_.seed(seed);
}

int32_t Sampler::sample(const float* logits, int vocab_size) {
    // Make a copy for modification
    std::vector<float> logits_copy(logits, logits + vocab_size);

    // Apply repetition penalty
    if (config_.repeat_penalty != 1.0f && !recent_tokens_.empty()) {
        apply_repetition_penalty(logits_copy.data(), vocab_size);
    }

    int32_t token;

    // Select sampling method
    if (config_.mirostat > 0) {
        token = sample_mirostat_v2(logits_copy.data(), vocab_size);
    } else if (config_.temperature <= 0.0f) {
        token = sample_greedy(logits_copy.data(), vocab_size);
    } else {
        // Apply temperature
        for (int i = 0; i < vocab_size; ++i) {
            logits_copy[i] /= config_.temperature;
        }

        // Apply top-k if enabled
        if (config_.top_k > 0) {
            token = sample_top_k(logits_copy.data(), vocab_size, config_.top_k);
        }
        // Apply top-p if enabled
        else if (config_.top_p < 1.0f) {
            token = sample_top_p(logits_copy.data(), vocab_size, config_.top_p);
        } else {
            token = sample_temperature(logits_copy.data(), vocab_size);
        }
    }

    // Track recent tokens for repetition penalty
    recent_tokens_.push_back(token);
    if (static_cast<int>(recent_tokens_.size()) > config_.repeat_last_n) {
        recent_tokens_.erase(recent_tokens_.begin());
    }

    return token;
}

int32_t Sampler::sample_greedy(const float* logits, int vocab_size) {
    return static_cast<int32_t>(std::max_element(logits, logits + vocab_size) - logits);
}

int32_t Sampler::sample_temperature(float* logits, int vocab_size) {
    softmax(logits, vocab_size);

    // Sample from probability distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);

    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += logits[i];
        if (r <= cumsum) {
            return static_cast<int32_t>(i);
        }
    }

    return static_cast<int32_t>(vocab_size - 1);
}

int32_t Sampler::sample_top_k(float* logits, int vocab_size, int k) {
    // Find top-k indices
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                     [logits](int a, int b) { return logits[a] > logits[b]; });

    // Zero out non-top-k
    for (int i = k; i < vocab_size; ++i) {
        logits[indices[i]] = -std::numeric_limits<float>::infinity();
    }

    return sample_temperature(logits, vocab_size);
}

int32_t Sampler::sample_top_p(float* logits, int vocab_size, float p) {
    // Softmax first
    softmax(logits, vocab_size);

    // Sort by probability
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
             [logits](int a, int b) { return logits[a] > logits[b]; });

    // Find nucleus
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += logits[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }

    // Zero out non-nucleus
    for (int i = cutoff; i < vocab_size; ++i) {
        logits[indices[i]] = 0.0f;
    }

    // Renormalize and sample
    float sum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        sum += logits[indices[i]];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng_);

    cumsum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        cumsum += logits[indices[i]];
        if (r <= cumsum) {
            return static_cast<int32_t>(indices[i]);
        }
    }

    return static_cast<int32_t>(indices[cutoff - 1]);
}

int32_t Sampler::sample_mirostat_v2(float* logits, int vocab_size) {
    // Mirostat v2 sampling
    // Based on https://arxiv.org/abs/2007.14966

    softmax(logits, vocab_size);

    // Sort by probability
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
             [logits](int a, int b) { return logits[a] > logits[b]; });

    // Calculate surprise values
    float mu = mirostat_mu_;
    float tau = config_.mirostat_tau;
    float eta = config_.mirostat_eta;

    // Find cutoff based on target surprise
    float s_sum = 0.0f;
    int k = 0;
    for (int i = 0; i < vocab_size; ++i) {
        float p = logits[indices[i]];
        if (p <= 0) continue;

        float s = -std::log2(p);
        s_sum += s * p;
        k = i + 1;

        if (s > mu) break;
    }

    // Truncate and renormalize
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += logits[indices[i]];
    }

    // Sample
    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng_);

    float cumsum = 0.0f;
    int32_t token = indices[k - 1];
    for (int i = 0; i < k; ++i) {
        cumsum += logits[indices[i]];
        if (r <= cumsum) {
            token = static_cast<int32_t>(indices[i]);
            break;
        }
    }

    // Update mu
    float surprise = -std::log2(logits[token]);
    mirostat_mu_ = mu - eta * (surprise - tau);

    return token;
}

void Sampler::apply_repetition_penalty(float* logits, int vocab_size) {
    for (int32_t token : recent_tokens_) {
        if (token >= 0 && token < vocab_size) {
            if (logits[token] > 0) {
                logits[token] /= config_.repeat_penalty;
            } else {
                logits[token] *= config_.repeat_penalty;
            }
        }
    }
}

void Sampler::softmax(float* logits, int vocab_size) {
    // Find max for numerical stability
    float max_val = *std::max_element(logits, logits + vocab_size);

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_sum;
    }
}

} // namespace odi

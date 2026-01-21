/**
 * ODI Engine - C ABI Wrapper
 *
 * This file implements the public C API defined in odi_engine.h
 * by wrapping the internal C++ implementation.
 */

#include <odi/odi_engine.h>
#include <odi/odi_types.h>
#include <odi/odi_error.h>

#include "core/engine.hpp"
#include "core/model.hpp"
#include "core/context.hpp"
#include "core/sampler.hpp"
#include "core/tokenizer.hpp"

#include <cstring>
#include <cstdlib>
#include <mutex>

// Thread-local error state
static thread_local odi_error_t g_last_error = ODI_SUCCESS;
static thread_local char g_last_error_msg[256] = {0};

extern "C" {

// ============================================================================
// Error handling
// ============================================================================

const char* odi_error_string(odi_error_t error) {
    switch (error) {
        case ODI_SUCCESS: return "Success";
        case ODI_ERROR_UNKNOWN: return "Unknown error";
        case ODI_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case ODI_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case ODI_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case ODI_ERROR_INVALID_STATE: return "Invalid state";
        case ODI_ERROR_OPERATION_FAILED: return "Operation failed";
        case ODI_ERROR_FILE_NOT_FOUND: return "File not found";
        case ODI_ERROR_FILE_READ: return "File read error";
        case ODI_ERROR_FILE_WRITE: return "File write error";
        case ODI_ERROR_FILE_INVALID: return "Invalid file";
        case ODI_ERROR_MMAP_FAILED: return "Memory map failed";
        case ODI_ERROR_MODEL_INVALID: return "Invalid model";
        case ODI_ERROR_MODEL_UNSUPPORTED_ARCH: return "Unsupported architecture";
        case ODI_ERROR_MODEL_UNSUPPORTED_DTYPE: return "Unsupported data type";
        case ODI_ERROR_MODEL_CORRUPTED: return "Corrupted model";
        case ODI_ERROR_MODEL_VERSION: return "Unsupported model version";
        case ODI_ERROR_TENSOR_NOT_FOUND: return "Tensor not found";
        case ODI_ERROR_BACKEND_UNAVAILABLE: return "Backend unavailable";
        case ODI_ERROR_BACKEND_INIT_FAILED: return "Backend initialization failed";
        case ODI_ERROR_BACKEND_NOT_SUPPORTED: return "Backend not supported";
        case ODI_ERROR_GPU_OUT_OF_MEMORY: return "GPU out of memory";
        case ODI_ERROR_CONTEXT_TOO_LONG: return "Context too long";
        case ODI_ERROR_TOKENIZATION_FAILED: return "Tokenization failed";
        case ODI_ERROR_INFERENCE_FAILED: return "Inference failed";
        case ODI_ERROR_CANCELLED: return "Operation cancelled";
        default: return "Unknown error code";
    }
}

odi_error_t odi_get_last_error(void) {
    return g_last_error;
}

const char* odi_get_last_error_message(void) {
    return g_last_error_msg;
}

void odi_set_error(odi_error_t error, const char* message) {
    g_last_error = error;
    if (message) {
        strncpy(g_last_error_msg, message, sizeof(g_last_error_msg) - 1);
        g_last_error_msg[sizeof(g_last_error_msg) - 1] = '\0';
    } else {
        g_last_error_msg[0] = '\0';
    }
}

void odi_clear_error(void) {
    g_last_error = ODI_SUCCESS;
    g_last_error_msg[0] = '\0';
}

// ============================================================================
// Version
// ============================================================================

void odi_get_version(int* major, int* minor, int* patch) {
    if (major) *major = ODI_VERSION_MAJOR;
    if (minor) *minor = ODI_VERSION_MINOR;
    if (patch) *patch = ODI_VERSION_PATCH;
}

static char g_version_string[32] = {0};

const char* odi_get_version_string(void) {
    if (g_version_string[0] == '\0') {
        snprintf(g_version_string, sizeof(g_version_string),
                "%d.%d.%d", ODI_VERSION_MAJOR, ODI_VERSION_MINOR, ODI_VERSION_PATCH);
    }
    return g_version_string;
}

// ============================================================================
// Utility functions
// ============================================================================

size_t odi_dtype_size(odi_dtype_t dtype) {
    return odi::dtype_size(dtype);
}

const char* odi_dtype_name(odi_dtype_t dtype) {
    switch (dtype) {
        case ODI_DTYPE_F32: return "F32";
        case ODI_DTYPE_F16: return "F16";
        case ODI_DTYPE_BF16: return "BF16";
        case ODI_DTYPE_I32: return "I32";
        case ODI_DTYPE_I16: return "I16";
        case ODI_DTYPE_I8: return "I8";
        case ODI_DTYPE_Q8_0: return "Q8_0";
        case ODI_DTYPE_Q4_0: return "Q4_0";
        case ODI_DTYPE_Q4_1: return "Q4_1";
        case ODI_DTYPE_Q5_0: return "Q5_0";
        case ODI_DTYPE_Q5_1: return "Q5_1";
        case ODI_DTYPE_Q4_K: return "Q4_K";
        case ODI_DTYPE_Q5_K: return "Q5_K";
        case ODI_DTYPE_Q6_K: return "Q6_K";
        case ODI_DTYPE_Q8_K: return "Q8_K";
        default: return "Unknown";
    }
}

const char* odi_backend_name(odi_backend_type_t backend) {
    switch (backend) {
        case ODI_BACKEND_AUTO: return "Auto";
        case ODI_BACKEND_CPU: return "CPU";
        case ODI_BACKEND_METAL: return "Metal";
        case ODI_BACKEND_VULKAN: return "Vulkan";
        case ODI_BACKEND_CUDA: return "CUDA";
        default: return "Unknown";
    }
}

const char* odi_arch_name(odi_arch_type_t arch) {
    switch (arch) {
        case ODI_ARCH_UNKNOWN: return "Unknown";
        case ODI_ARCH_LLAMA: return "LLaMA";
        case ODI_ARCH_MISTRAL: return "Mistral";
        case ODI_ARCH_GEMMA: return "Gemma";
        case ODI_ARCH_PHI: return "Phi";
        case ODI_ARCH_QWEN: return "Qwen";
        case ODI_ARCH_GPT2: return "GPT-2";
        default: return "Unknown";
    }
}

void odi_free(void* ptr) {
    free(ptr);
}

// ============================================================================
// Engine
// ============================================================================

odi_engine_t* odi_engine_create(const odi_engine_config_t* config) {
    odi_clear_error();

    odi_engine_config_t cfg = config ? *config : odi_engine_config_default();

    try {
        auto* engine = new odi::Engine(cfg);
        return reinterpret_cast<odi_engine_t*>(engine);
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_OPERATION_FAILED, e.what());
        return nullptr;
    }
}

void odi_engine_destroy(odi_engine_t* engine) {
    if (engine) {
        delete reinterpret_cast<odi::Engine*>(engine);
    }
}

odi_backend_type_t odi_engine_get_backend(odi_engine_t* engine) {
    if (!engine) return ODI_BACKEND_CPU;
    return reinterpret_cast<odi::Engine*>(engine)->backend_type();
}

char* odi_engine_get_system_info(odi_engine_t* engine) {
    if (!engine) return nullptr;

    try {
        std::string info = reinterpret_cast<odi::Engine*>(engine)->system_info();
        char* result = static_cast<char*>(malloc(info.size() + 1));
        if (result) {
            strcpy(result, info.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

// ============================================================================
// Model
// ============================================================================

odi_model_t* odi_model_load(odi_engine_t* engine, const char* path,
                           const odi_model_config_t* config) {
    return odi_model_load_with_progress(engine, path, config, nullptr, nullptr);
}

odi_model_t* odi_model_load_with_progress(odi_engine_t* engine, const char* path,
                                          const odi_model_config_t* config,
                                          odi_progress_callback_t callback,
                                          void* user_data) {
    odi_clear_error();

    if (!engine || !path) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid engine or path");
        return nullptr;
    }

    odi_model_config_t cfg = config ? *config : odi_model_config_default();

    try {
        auto* eng = reinterpret_cast<odi::Engine*>(engine);
        auto* model = eng->load_model_with_progress(path, cfg, callback, user_data);
        return reinterpret_cast<odi_model_t*>(model);
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_OPERATION_FAILED, e.what());
        return nullptr;
    }
}

void odi_model_unload(odi_model_t* model) {
    if (model) {
        auto* m = reinterpret_cast<odi::Model*>(model);
        m->engine()->unload_model(m);
    }
}

odi_arch_type_t odi_model_get_arch(odi_model_t* model) {
    if (!model) return ODI_ARCH_UNKNOWN;
    return reinterpret_cast<odi::Model*>(model)->arch();
}

const char* odi_model_get_name(odi_model_t* model) {
    if (!model) return "";
    return reinterpret_cast<odi::Model*>(model)->name().c_str();
}

char* odi_model_get_info(odi_model_t* model) {
    if (!model) return nullptr;

    try {
        std::string info = reinterpret_cast<odi::Model*>(model)->get_info_json();
        char* result = static_cast<char*>(malloc(info.size() + 1));
        if (result) {
            strcpy(result, info.c_str());
        }
        return result;
    } catch (...) {
        return nullptr;
    }
}

int odi_model_get_vocab_size(odi_model_t* model) {
    if (!model) return 0;
    return static_cast<int>(reinterpret_cast<odi::Model*>(model)->vocab_size());
}

int odi_model_get_context_length(odi_model_t* model) {
    if (!model) return 0;
    return static_cast<int>(reinterpret_cast<odi::Model*>(model)->context_length());
}

int odi_model_get_embedding_dim(odi_model_t* model) {
    if (!model) return 0;
    return static_cast<int>(reinterpret_cast<odi::Model*>(model)->embedding_dim());
}

// ============================================================================
// Context
// ============================================================================

odi_context_t* odi_context_create(odi_model_t* model, const odi_context_config_t* config) {
    odi_clear_error();

    if (!model) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid model");
        return nullptr;
    }

    odi_context_config_t cfg = config ? *config : odi_context_config_default();

    try {
        auto* m = reinterpret_cast<odi::Model*>(model);
        auto* ctx = new odi::Context(m, cfg);
        return reinterpret_cast<odi_context_t*>(ctx);
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_OPERATION_FAILED, e.what());
        return nullptr;
    }
}

void odi_context_destroy(odi_context_t* ctx) {
    if (ctx) {
        delete reinterpret_cast<odi::Context*>(ctx);
    }
}

void odi_context_reset(odi_context_t* ctx) {
    if (ctx) {
        reinterpret_cast<odi::Context*>(ctx)->reset();
    }
}

int odi_context_get_length(odi_context_t* ctx) {
    if (!ctx) return 0;
    return reinterpret_cast<odi::Context*>(ctx)->current_position();
}

// ============================================================================
// Tokenization
// ============================================================================

int odi_tokenize(odi_model_t* model, const char* text, int32_t* tokens,
                 int max_tokens, bool add_bos) {
    if (!model || !text || !tokens) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return -1;
    }

    try {
        auto* m = reinterpret_cast<odi::Model*>(model);
        auto result = m->tokenize(text, add_bos);

        int count = std::min(static_cast<int>(result.size()), max_tokens);
        for (int i = 0; i < count; ++i) {
            tokens[i] = result[i];
        }
        return count;
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_TOKENIZATION_FAILED, e.what());
        return -1;
    }
}

int odi_detokenize(odi_model_t* model, const int32_t* tokens, int num_tokens,
                   char* text, size_t max_len) {
    if (!model || !tokens || !text || max_len == 0) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return -1;
    }

    try {
        auto* m = reinterpret_cast<odi::Model*>(model);
        std::vector<int32_t> token_vec(tokens, tokens + num_tokens);
        std::string result = m->detokenize(token_vec);

        size_t len = std::min(result.size(), max_len - 1);
        strncpy(text, result.c_str(), len);
        text[len] = '\0';
        return static_cast<int>(len);
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_TOKENIZATION_FAILED, e.what());
        return -1;
    }
}

const char* odi_token_to_string(odi_model_t* model, int32_t token_id) {
    if (!model) return nullptr;

    try {
        static thread_local std::string result;
        auto* m = reinterpret_cast<odi::Model*>(model);
        result = m->token_to_string(token_id);
        return result.c_str();
    } catch (...) {
        return nullptr;
    }
}

// ============================================================================
// Generation
// ============================================================================

odi_error_t odi_generate(odi_context_t* ctx, const char* prompt, char* output,
                         size_t max_len, const odi_sampler_config_t* sampler) {
    return odi_generate_ex(ctx, prompt, output, max_len, sampler, nullptr);
}

odi_error_t odi_generate_ex(odi_context_t* ctx, const char* prompt, char* output,
                            size_t max_len, const odi_sampler_config_t* sampler,
                            odi_generation_result_t* result) {
    odi_clear_error();

    if (!ctx || !prompt || !output || max_len == 0) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return ODI_ERROR_INVALID_ARGUMENT;
    }

    auto* context = reinterpret_cast<odi::Context*>(ctx);
    auto* model = context->model();

    odi_sampler_config_t cfg = sampler ? *sampler : odi_sampler_config_default();
    odi::Sampler samp(cfg);

    try {
        // Tokenize prompt
        auto tokens = model->tokenize(prompt, true);

        // Evaluate prompt
        odi_error_t err = context->eval(tokens.data(), static_cast<int>(tokens.size()));
        if (err != ODI_SUCCESS) return err;

        int prompt_tokens = static_cast<int>(tokens.size());

        // Generate tokens
        std::vector<int32_t> generated;
        int max_gen = static_cast<int>(max_len / 4);  // Rough estimate

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < max_gen; ++i) {
            // Sample next token
            const float* logits = context->get_logits();
            int vocab_size = static_cast<int>(model->vocab_size());
            int32_t token = samp.sample(logits, vocab_size);

            // Check for EOS
            if (token == model->eos_token_id()) {
                break;
            }

            generated.push_back(token);

            // Evaluate new token
            err = context->eval(&token, 1);
            if (err != ODI_SUCCESS) break;
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        // Decode output
        std::string text = model->detokenize(generated);
        size_t len = std::min(text.size(), max_len - 1);
        strncpy(output, text.c_str(), len);
        output[len] = '\0';

        // Fill result if requested
        if (result) {
            float duration_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
            result->tokens_generated = static_cast<int>(generated.size());
            result->tokens_evaluated = prompt_tokens;
            result->generation_time_ms = duration_ms;
            result->tokens_per_second = (duration_ms > 0) ? (generated.size() * 1000.0f / duration_ms) : 0;
            result->stop_reason = (generated.empty() || generated.back() == model->eos_token_id())
                                 ? ODI_STOP_EOS : ODI_STOP_MAX_TOKENS;
        }

        return ODI_SUCCESS;

    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_INFERENCE_FAILED, e.what());
        return ODI_ERROR_INFERENCE_FAILED;
    }
}

odi_error_t odi_generate_tokens(odi_context_t* ctx, const char* prompt, char* output,
                                 size_t max_len, int max_tokens,
                                 const odi_sampler_config_t* sampler) {
    // Similar to odi_generate_ex but with explicit token limit
    odi_clear_error();

    if (!ctx || !prompt || !output || max_len == 0) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return ODI_ERROR_INVALID_ARGUMENT;
    }

    auto* context = reinterpret_cast<odi::Context*>(ctx);
    auto* model = context->model();

    odi_sampler_config_t cfg = sampler ? *sampler : odi_sampler_config_default();
    odi::Sampler samp(cfg);

    try {
        auto tokens = model->tokenize(prompt, true);
        odi_error_t err = context->eval(tokens.data(), static_cast<int>(tokens.size()));
        if (err != ODI_SUCCESS) return err;

        std::vector<int32_t> generated;

        for (int i = 0; i < max_tokens; ++i) {
            const float* logits = context->get_logits();
            int vocab_size = static_cast<int>(model->vocab_size());
            int32_t token = samp.sample(logits, vocab_size);

            if (token == model->eos_token_id()) break;

            generated.push_back(token);
            err = context->eval(&token, 1);
            if (err != ODI_SUCCESS) break;
        }

        std::string text = model->detokenize(generated);
        size_t len = std::min(text.size(), max_len - 1);
        strncpy(output, text.c_str(), len);
        output[len] = '\0';

        return ODI_SUCCESS;

    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_INFERENCE_FAILED, e.what());
        return ODI_ERROR_INFERENCE_FAILED;
    }
}

// ============================================================================
// Streaming Generation
// ============================================================================

odi_error_t odi_generate_stream(odi_context_t* ctx, const char* prompt,
                                odi_token_callback_t callback, void* user_data,
                                const odi_sampler_config_t* sampler) {
    return odi_generate_stream_ex(ctx, prompt, callback, user_data, sampler, nullptr);
}

odi_error_t odi_generate_stream_ex(odi_context_t* ctx, const char* prompt,
                                   odi_token_callback_t callback, void* user_data,
                                   const odi_sampler_config_t* sampler,
                                   odi_generation_result_t* result) {
    odi_clear_error();

    if (!ctx || !prompt || !callback) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return ODI_ERROR_INVALID_ARGUMENT;
    }

    auto* context = reinterpret_cast<odi::Context*>(ctx);
    auto* model = context->model();

    odi_sampler_config_t cfg = sampler ? *sampler : odi_sampler_config_default();
    odi::Sampler samp(cfg);

    try {
        auto tokens = model->tokenize(prompt, true);
        odi_error_t err = context->eval(tokens.data(), static_cast<int>(tokens.size()));
        if (err != ODI_SUCCESS) return err;

        int prompt_tokens = static_cast<int>(tokens.size());
        int generated_count = 0;
        int max_gen = static_cast<int>(context->max_context_length()) - prompt_tokens;
        int stop_reason = ODI_STOP_NONE;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < max_gen; ++i) {
            const float* logits = context->get_logits();
            int vocab_size = static_cast<int>(model->vocab_size());
            int32_t token = samp.sample(logits, vocab_size);

            if (token == model->eos_token_id()) {
                stop_reason = ODI_STOP_EOS;
                break;
            }

            generated_count++;

            // Decode and send to callback
            std::string token_str = model->token_to_string(token);
            if (!callback(token_str.c_str(), token_str.size(), user_data)) {
                stop_reason = ODI_STOP_USER;
                break;
            }

            err = context->eval(&token, 1);
            if (err != ODI_SUCCESS) {
                stop_reason = ODI_STOP_ERROR;
                break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        if (result) {
            float duration_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
            result->tokens_generated = generated_count;
            result->tokens_evaluated = prompt_tokens;
            result->generation_time_ms = duration_ms;
            result->tokens_per_second = (duration_ms > 0) ? (generated_count * 1000.0f / duration_ms) : 0;
            result->stop_reason = stop_reason;
        }

        return ODI_SUCCESS;

    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_INFERENCE_FAILED, e.what());
        return ODI_ERROR_INFERENCE_FAILED;
    }
}

// ============================================================================
// Low-level API
// ============================================================================

odi_error_t odi_eval(odi_context_t* ctx, const int32_t* tokens, int num_tokens) {
    if (!ctx || !tokens || num_tokens <= 0) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid argument");
        return ODI_ERROR_INVALID_ARGUMENT;
    }

    return reinterpret_cast<odi::Context*>(ctx)->eval(tokens, num_tokens);
}

int32_t odi_sample(odi_context_t* ctx, const odi_sampler_config_t* sampler) {
    if (!ctx) {
        odi_set_error(ODI_ERROR_INVALID_ARGUMENT, "Invalid context");
        return -1;
    }

    auto* context = reinterpret_cast<odi::Context*>(ctx);
    auto* model = context->model();

    odi_sampler_config_t cfg = sampler ? *sampler : odi_sampler_config_default();
    odi::Sampler samp(cfg);

    const float* logits = context->get_logits();
    int vocab_size = static_cast<int>(model->vocab_size());

    return samp.sample(logits, vocab_size);
}

const float* odi_get_logits(odi_context_t* ctx) {
    if (!ctx) return nullptr;
    return reinterpret_cast<odi::Context*>(ctx)->get_logits();
}

const float* odi_get_embeddings(odi_context_t* ctx) {
    if (!ctx) return nullptr;
    return reinterpret_cast<odi::Context*>(ctx)->get_embeddings();
}

} // extern "C"

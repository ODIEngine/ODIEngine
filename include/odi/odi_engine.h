#ifndef ODI_ENGINE_H
#define ODI_ENGINE_H

#include "odi_types.h"
#include "odi_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ODI Engine - On-Device Inference Engine
 *
 * A lightweight inference engine for running LLMs locally on consumer
 * hardware - laptops, desktops, phones. Optimized for devices without
 * datacenter GPUs. Supports GGUF models with INT4/INT8 quantization.
 *
 * Basic usage:
 *   odi_engine_t* engine = odi_engine_create(NULL);
 *   odi_model_t* model = odi_model_load(engine, "model.gguf", NULL);
 *   odi_context_t* ctx = odi_context_create(model, NULL);
 *
 *   char output[4096];
 *   odi_generate(ctx, "Hello, ", output, sizeof(output), NULL);
 *   printf("%s\n", output);
 *
 *   odi_context_destroy(ctx);
 *   odi_model_unload(model);
 *   odi_engine_destroy(engine);
 */

/* ============================================================================
 * Version
 * ============================================================================ */

/* Get version numbers */
ODI_API void odi_get_version(int* major, int* minor, int* patch);

/* Get version string (e.g., "0.1.0") */
ODI_API const char* odi_get_version_string(void);

/* ============================================================================
 * Engine
 * ============================================================================ */

/*
 * Create an inference engine instance.
 *
 * @param config  Engine configuration, or NULL for defaults
 * @return        Engine handle, or NULL on error
 */
ODI_API odi_engine_t* odi_engine_create(const odi_engine_config_t* config);

/*
 * Destroy an engine instance and free all resources.
 * This also unloads all models and destroys all contexts.
 *
 * @param engine  Engine to destroy (may be NULL)
 */
ODI_API void odi_engine_destroy(odi_engine_t* engine);

/*
 * Get the active backend type for the engine.
 *
 * @param engine  Engine handle
 * @return        Active backend type
 */
ODI_API odi_backend_type_t odi_engine_get_backend(odi_engine_t* engine);

/*
 * Get system information (available backends, CPU features, etc.)
 *
 * @param engine  Engine handle
 * @return        JSON string with system info (caller must free)
 */
ODI_API char* odi_engine_get_system_info(odi_engine_t* engine);

/* ============================================================================
 * Model
 * ============================================================================ */

/*
 * Load a model from a GGUF file.
 *
 * @param engine    Engine handle
 * @param path      Path to the GGUF model file
 * @param config    Model configuration, or NULL for defaults
 * @return          Model handle, or NULL on error
 */
ODI_API odi_model_t* odi_model_load(
    odi_engine_t* engine,
    const char* path,
    const odi_model_config_t* config
);

/*
 * Load a model with progress callback.
 *
 * @param engine    Engine handle
 * @param path      Path to the GGUF model file
 * @param config    Model configuration, or NULL for defaults
 * @param callback  Progress callback function
 * @param user_data User data passed to callback
 * @return          Model handle, or NULL on error
 */
ODI_API odi_model_t* odi_model_load_with_progress(
    odi_engine_t* engine,
    const char* path,
    const odi_model_config_t* config,
    odi_progress_callback_t callback,
    void* user_data
);

/*
 * Unload a model and free its resources.
 *
 * @param model  Model to unload (may be NULL)
 */
ODI_API void odi_model_unload(odi_model_t* model);

/*
 * Get the model architecture type.
 *
 * @param model  Model handle
 * @return       Architecture type
 */
ODI_API odi_arch_type_t odi_model_get_arch(odi_model_t* model);

/*
 * Get the model name from metadata.
 *
 * @param model  Model handle
 * @return       Model name string (do not free)
 */
ODI_API const char* odi_model_get_name(odi_model_t* model);

/*
 * Get model information as JSON string.
 *
 * @param model  Model handle
 * @return       JSON string with model info (caller must free)
 */
ODI_API char* odi_model_get_info(odi_model_t* model);

/*
 * Get the model's vocabulary size.
 *
 * @param model  Model handle
 * @return       Vocabulary size
 */
ODI_API int odi_model_get_vocab_size(odi_model_t* model);

/*
 * Get the model's context length limit.
 *
 * @param model  Model handle
 * @return       Maximum context length
 */
ODI_API int odi_model_get_context_length(odi_model_t* model);

/*
 * Get the model's embedding dimension.
 *
 * @param model  Model handle
 * @return       Embedding dimension
 */
ODI_API int odi_model_get_embedding_dim(odi_model_t* model);

/* ============================================================================
 * Context
 * ============================================================================ */

/*
 * Create an inference context for a model.
 * Each context maintains its own KV cache and can generate independently.
 *
 * @param model   Model handle
 * @param config  Context configuration, or NULL for defaults
 * @return        Context handle, or NULL on error
 */
ODI_API odi_context_t* odi_context_create(
    odi_model_t* model,
    const odi_context_config_t* config
);

/*
 * Destroy an inference context.
 *
 * @param ctx  Context to destroy (may be NULL)
 */
ODI_API void odi_context_destroy(odi_context_t* ctx);

/*
 * Reset the context (clear KV cache).
 *
 * @param ctx  Context handle
 */
ODI_API void odi_context_reset(odi_context_t* ctx);

/*
 * Get the current number of tokens in the context.
 *
 * @param ctx  Context handle
 * @return     Number of tokens in KV cache
 */
ODI_API int odi_context_get_length(odi_context_t* ctx);

/* ============================================================================
 * Tokenization
 * ============================================================================ */

/*
 * Tokenize text into token IDs.
 *
 * @param model      Model handle
 * @param text       Input text
 * @param tokens     Output token array
 * @param max_tokens Maximum number of tokens to output
 * @param add_bos    Add beginning-of-sequence token
 * @return           Number of tokens, or negative error code
 */
ODI_API int odi_tokenize(
    odi_model_t* model,
    const char* text,
    int32_t* tokens,
    int max_tokens,
    bool add_bos
);

/*
 * Detokenize token IDs to text.
 *
 * @param model      Model handle
 * @param tokens     Input token array
 * @param num_tokens Number of tokens
 * @param text       Output text buffer
 * @param max_len    Maximum output length
 * @return           Length of output text, or negative error code
 */
ODI_API int odi_detokenize(
    odi_model_t* model,
    const int32_t* tokens,
    int num_tokens,
    char* text,
    size_t max_len
);

/*
 * Get the token string for a token ID.
 *
 * @param model    Model handle
 * @param token_id Token ID
 * @return         Token string (do not free), or NULL if invalid
 */
ODI_API const char* odi_token_to_string(odi_model_t* model, int32_t token_id);

/* ============================================================================
 * Generation (Synchronous)
 * ============================================================================ */

/*
 * Generate text from a prompt (blocking).
 *
 * @param ctx        Context handle
 * @param prompt     Input prompt text
 * @param output     Output buffer for generated text
 * @param max_len    Maximum output length
 * @param sampler    Sampling parameters, or NULL for defaults
 * @return           Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_generate(
    odi_context_t* ctx,
    const char* prompt,
    char* output,
    size_t max_len,
    const odi_sampler_config_t* sampler
);

/*
 * Generate text with detailed result information.
 *
 * @param ctx        Context handle
 * @param prompt     Input prompt text
 * @param output     Output buffer for generated text
 * @param max_len    Maximum output length
 * @param sampler    Sampling parameters, or NULL for defaults
 * @param result     Output generation result (may be NULL)
 * @return           Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_generate_ex(
    odi_context_t* ctx,
    const char* prompt,
    char* output,
    size_t max_len,
    const odi_sampler_config_t* sampler,
    odi_generation_result_t* result
);

/*
 * Generate with maximum token count limit.
 *
 * @param ctx         Context handle
 * @param prompt      Input prompt text
 * @param output      Output buffer for generated text
 * @param max_len     Maximum output buffer length
 * @param max_tokens  Maximum tokens to generate
 * @param sampler     Sampling parameters, or NULL for defaults
 * @return            Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_generate_tokens(
    odi_context_t* ctx,
    const char* prompt,
    char* output,
    size_t max_len,
    int max_tokens,
    const odi_sampler_config_t* sampler
);

/* ============================================================================
 * Generation (Streaming/Async)
 * ============================================================================ */

/*
 * Generate text with streaming callback.
 * The callback is called for each generated token.
 * Return false from the callback to stop generation.
 *
 * @param ctx        Context handle
 * @param prompt     Input prompt text
 * @param callback   Token callback function
 * @param user_data  User data passed to callback
 * @param sampler    Sampling parameters, or NULL for defaults
 * @return           Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_generate_stream(
    odi_context_t* ctx,
    const char* prompt,
    odi_token_callback_t callback,
    void* user_data,
    const odi_sampler_config_t* sampler
);

/*
 * Generate streaming with detailed result.
 *
 * @param ctx        Context handle
 * @param prompt     Input prompt text
 * @param callback   Token callback function
 * @param user_data  User data passed to callback
 * @param sampler    Sampling parameters, or NULL for defaults
 * @param result     Output generation result (may be NULL)
 * @return           Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_generate_stream_ex(
    odi_context_t* ctx,
    const char* prompt,
    odi_token_callback_t callback,
    void* user_data,
    const odi_sampler_config_t* sampler,
    odi_generation_result_t* result
);

/* ============================================================================
 * Low-level API
 * ============================================================================ */

/*
 * Evaluate tokens (add to context without generating).
 * Useful for processing prompts in chunks.
 *
 * @param ctx        Context handle
 * @param tokens     Token array to evaluate
 * @param num_tokens Number of tokens
 * @return           Error code (ODI_SUCCESS on success)
 */
ODI_API odi_error_t odi_eval(
    odi_context_t* ctx,
    const int32_t* tokens,
    int num_tokens
);

/*
 * Sample the next token based on current context.
 *
 * @param ctx      Context handle
 * @param sampler  Sampling parameters, or NULL for defaults
 * @return         Sampled token ID, or negative error code
 */
ODI_API int32_t odi_sample(
    odi_context_t* ctx,
    const odi_sampler_config_t* sampler
);

/*
 * Get logits for the last token.
 *
 * @param ctx   Context handle
 * @return      Pointer to logits array (vocab_size floats), or NULL on error
 */
ODI_API const float* odi_get_logits(odi_context_t* ctx);

/*
 * Get embeddings for the evaluated tokens.
 *
 * @param ctx   Context handle
 * @return      Pointer to embeddings array, or NULL if not available
 */
ODI_API const float* odi_get_embeddings(odi_context_t* ctx);

/* ============================================================================
 * Utility
 * ============================================================================ */

/*
 * Free memory allocated by ODI functions.
 *
 * @param ptr  Pointer to free (may be NULL)
 */
ODI_API void odi_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* ODI_ENGINE_H */

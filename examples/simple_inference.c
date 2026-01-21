/**
 * ODI Engine - Simple Inference Example
 *
 * This example demonstrates basic usage of the ODI Engine C API
 * to load a GGUF model and generate text.
 *
 * Usage: simple_inference <model.gguf> "Your prompt here"
 */

#include <odi/odi_engine.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_progress(float progress, void* user_data) {
    (void)user_data;
    printf("\rLoading model: %.0f%%", progress * 100);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> \"prompt\"\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];

    // Print version
    int major, minor, patch;
    odi_get_version(&major, &minor, &patch);
    printf("ODI Engine v%d.%d.%d\n", major, minor, patch);

    // Create engine
    printf("Creating engine...\n");
    odi_engine_config_t engine_config = odi_engine_config_default();
    engine_config.verbose = true;

    odi_engine_t* engine = odi_engine_create(&engine_config);
    if (!engine) {
        fprintf(stderr, "Failed to create engine: %s\n", odi_get_last_error_message());
        return 1;
    }

    // Print system info
    char* sys_info = odi_engine_get_system_info(engine);
    if (sys_info) {
        printf("System info: %s\n", sys_info);
        odi_free(sys_info);
    }

    // Load model
    printf("Loading model: %s\n", model_path);
    odi_model_config_t model_config = odi_model_config_default();

    odi_model_t* model = odi_model_load_with_progress(engine, model_path, &model_config,
                                                       print_progress, NULL);
    printf("\n");

    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", odi_get_last_error_message());
        odi_engine_destroy(engine);
        return 1;
    }

    // Print model info
    printf("Model: %s\n", odi_model_get_name(model));
    printf("Architecture: %s\n", odi_arch_name(odi_model_get_arch(model)));
    printf("Vocab size: %d\n", odi_model_get_vocab_size(model));
    printf("Context length: %d\n", odi_model_get_context_length(model));
    printf("Embedding dim: %d\n", odi_model_get_embedding_dim(model));

    // Create context
    printf("\nCreating inference context...\n");
    odi_context_config_t ctx_config = odi_context_config_default();
    ctx_config.context_length = 2048;

    odi_context_t* ctx = odi_context_create(model, &ctx_config);
    if (!ctx) {
        fprintf(stderr, "Failed to create context: %s\n", odi_get_last_error_message());
        odi_model_unload(model);
        odi_engine_destroy(engine);
        return 1;
    }

    // Generate text
    printf("\nPrompt: %s\n", prompt);
    printf("Generating...\n\n");

    odi_sampler_config_t sampler = odi_sampler_config_default();
    sampler.temperature = 0.7f;
    sampler.top_p = 0.9f;

    char output[4096] = {0};
    odi_generation_result_t result;

    odi_error_t err = odi_generate_ex(ctx, prompt, output, sizeof(output), &sampler, &result);

    if (err != ODI_SUCCESS) {
        fprintf(stderr, "Generation failed: %s\n", odi_error_string(err));
    } else {
        printf("Output: %s\n", output);
        printf("\n---\n");
        printf("Tokens generated: %d\n", result.tokens_generated);
        printf("Tokens evaluated: %d\n", result.tokens_evaluated);
        printf("Generation time: %.2f ms\n", result.generation_time_ms);
        printf("Speed: %.2f tokens/sec\n", result.tokens_per_second);
    }

    // Cleanup
    odi_context_destroy(ctx);
    odi_model_unload(model);
    odi_engine_destroy(engine);

    printf("\nDone.\n");
    return 0;
}

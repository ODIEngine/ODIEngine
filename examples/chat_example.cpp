/**
 * ODI Engine - Interactive Chat Example
 *
 * This example demonstrates streaming generation with the ODI Engine.
 * It creates an interactive chat session with a loaded model.
 *
 * Usage: chat_example <model.gguf>
 */

#include <odi/odi_engine.h>
#include <iostream>
#include <string>
#include <vector>

// Callback for streaming tokens
bool stream_callback(const char* token, size_t token_len, void* user_data) {
    (void)token_len;
    (void)user_data;
    std::cout << token << std::flush;
    return true;  // Continue generation
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <model.gguf>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Commands:" << std::endl;
    std::cerr << "  /quit     - Exit the chat" << std::endl;
    std::cerr << "  /reset    - Clear conversation history" << std::endl;
    std::cerr << "  /info     - Show model information" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    // Print version
    std::cout << "ODI Engine v" << odi_get_version_string() << std::endl;
    std::cout << "Interactive Chat" << std::endl;
    std::cout << "=================" << std::endl;

    // Create engine
    odi_engine_config_t engine_config = odi_engine_config_default();
    odi_engine_t* engine = odi_engine_create(&engine_config);
    if (!engine) {
        std::cerr << "Failed to create engine: " << odi_get_last_error_message() << std::endl;
        return 1;
    }

    // Load model with progress
    std::cout << "Loading model: " << model_path << std::endl;

    auto progress_cb = [](float progress, void*) {
        std::cout << "\rLoading: " << static_cast<int>(progress * 100) << "%" << std::flush;
    };

    odi_model_config_t model_config = odi_model_config_default();
    odi_model_t* model = odi_model_load_with_progress(engine, model_path, &model_config,
                                                       progress_cb, nullptr);
    std::cout << std::endl;

    if (!model) {
        std::cerr << "Failed to load model: " << odi_get_last_error_message() << std::endl;
        odi_engine_destroy(engine);
        return 1;
    }

    std::cout << "Model: " << odi_model_get_name(model) << std::endl;
    std::cout << "Architecture: " << odi_arch_name(odi_model_get_arch(model)) << std::endl;
    std::cout << std::endl;

    // Create context
    odi_context_config_t ctx_config = odi_context_config_default();
    ctx_config.context_length = 4096;

    odi_context_t* ctx = odi_context_create(model, &ctx_config);
    if (!ctx) {
        std::cerr << "Failed to create context: " << odi_get_last_error_message() << std::endl;
        odi_model_unload(model);
        odi_engine_destroy(engine);
        return 1;
    }

    // Sampler configuration
    odi_sampler_config_t sampler = odi_sampler_config_default();
    sampler.temperature = 0.7f;
    sampler.top_p = 0.9f;
    sampler.repeat_penalty = 1.1f;

    // Chat loop
    std::cout << "Type your message (or /quit to exit, /reset to clear, /info for model info)" << std::endl;
    std::cout << std::endl;

    std::string conversation;
    std::string line;

    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, line);

        if (line.empty()) continue;

        // Handle commands
        if (line == "/quit" || line == "/exit") {
            break;
        } else if (line == "/reset") {
            odi_context_reset(ctx);
            conversation.clear();
            std::cout << "[Context cleared]" << std::endl;
            continue;
        } else if (line == "/info") {
            char* info = odi_model_get_info(model);
            if (info) {
                std::cout << info << std::endl;
                odi_free(info);
            }
            continue;
        }

        // Build prompt (simple chat format)
        std::string prompt;
        if (conversation.empty()) {
            prompt = "User: " + line + "\nAssistant:";
        } else {
            conversation += "\nUser: " + line + "\nAssistant:";
            prompt = conversation;
        }

        std::cout << "Assistant: ";

        // Stream generation
        odi_generation_result_t result;
        odi_error_t err = odi_generate_stream_ex(ctx, prompt.c_str(), stream_callback,
                                                  nullptr, &sampler, &result);

        std::cout << std::endl;

        if (err != ODI_SUCCESS) {
            std::cerr << "[Error: " << odi_error_string(err) << "]" << std::endl;
        } else {
            // Update conversation history (simplified - would need to capture output)
            std::cout << "[" << result.tokens_generated << " tokens, "
                     << result.tokens_per_second << " tok/s]" << std::endl;
        }

        std::cout << std::endl;
    }

    // Cleanup
    std::cout << "Cleaning up..." << std::endl;
    odi_context_destroy(ctx);
    odi_model_unload(model);
    odi_engine_destroy(engine);

    std::cout << "Goodbye!" << std::endl;
    return 0;
}

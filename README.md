# ODI Engine

**On-Device Inference Engine** — Run LLMs locally on your hardware.

A lightweight inference engine for consumer devices (laptops, desktops, phones) without datacenter GPUs. Load GGUF models and generate text with a simple C API.

## Features

- **GGUF Support** — Native parser for GGUF models (Llama, Mistral, Gemma, Phi, Qwen)
- **Quantization** — INT4/INT8 inference (Q4_0, Q4_K, Q8_0, etc.)
- **SIMD Optimized** — AVX2/AVX-512 on x86, NEON on ARM
- **Multi-Backend** — CPU now, Metal/Vulkan/CUDA planned
- **Streaming** — Token-by-token generation with callbacks
- **C API** — Simple `extern "C"` interface, easy to bind to any language

## Building

```bash
git clone https://github.com/user/ODIEngine.git
cd ODIEngine
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options

| Option               | Default | Description                  |
| -------------------- | ------- | ---------------------------- |
| `ODI_BUILD_TESTS`    | ON      | Build unit tests             |
| `ODI_BUILD_EXAMPLES` | ON      | Build example programs       |
| `ODI_ENABLE_METAL`   | Auto    | Enable Metal backend (macOS) |
| `ODI_ENABLE_VULKAN`  | OFF     | Enable Vulkan backend        |
| `ODI_ENABLE_CUDA`    | OFF     | Enable CUDA backend          |

## Quick Start

### C API

```c
#include <odi/odi_engine.h>

int main() {
    // Create engine
    odi_engine_t* engine = odi_engine_create(NULL);

    // Load model
    odi_model_t* model = odi_model_load(engine, "llama-7b-q4.gguf", NULL);

    // Create inference context
    odi_context_t* ctx = odi_context_create(model, NULL);

    // Generate text
    char output[4096];
    odi_generate(ctx, "The meaning of life is", output, sizeof(output), NULL);
    printf("%s\n", output);

    // Cleanup
    odi_context_destroy(ctx);
    odi_model_unload(model);
    odi_engine_destroy(engine);

    return 0;
}
```

### Streaming Generation

```c
bool on_token(const char* token, size_t len, void* user_data) {
    printf("%s", token);
    fflush(stdout);
    return true;  // continue generating
}

odi_generate_stream(ctx, "Once upon a time", on_token, NULL, NULL);
```

### Sampling Configuration

```c
odi_sampler_config_t sampler = odi_sampler_config_default();
sampler.temperature = 0.7f;
sampler.top_p = 0.9f;
sampler.top_k = 40;
sampler.repeat_penalty = 1.1f;

odi_generate(ctx, prompt, output, sizeof(output), &sampler);
```

## API Overview

### Engine Lifecycle

```c
odi_engine_t* odi_engine_create(const odi_engine_config_t* config);
void odi_engine_destroy(odi_engine_t* engine);
```

### Model Loading

```c
odi_model_t* odi_model_load(odi_engine_t* engine, const char* path, const odi_model_config_t* config);
void odi_model_unload(odi_model_t* model);

// Model info
const char* odi_model_get_name(odi_model_t* model);
int odi_model_get_vocab_size(odi_model_t* model);
int odi_model_get_context_length(odi_model_t* model);
```

### Inference Context

```c
odi_context_t* odi_context_create(odi_model_t* model, const odi_context_config_t* config);
void odi_context_destroy(odi_context_t* ctx);
void odi_context_reset(odi_context_t* ctx);  // Clear KV cache
```

### Generation

```c
// Blocking
odi_error_t odi_generate(odi_context_t* ctx, const char* prompt,
                         char* output, size_t max_len,
                         const odi_sampler_config_t* sampler);

// Streaming
odi_error_t odi_generate_stream(odi_context_t* ctx, const char* prompt,
                                odi_token_callback_t callback, void* user_data,
                                const odi_sampler_config_t* sampler);
```

### Tokenization

```c
int odi_tokenize(odi_model_t* model, const char* text, int32_t* tokens, int max_tokens, bool add_bos);
int odi_detokenize(odi_model_t* model, const int32_t* tokens, int num_tokens, char* text, size_t max_len);
```

## Supported Models

Any GGUF model should work. Tested architectures:

| Architecture              | Status |
| ------------------------- | ------ |
| LLaMA / LLaMA 2 / LLaMA 3 | ✅     |
| Mistral / Mixtral         | ✅     |
| Gemma / Gemma 2           | ✅     |
| Phi-2 / Phi-3             | ✅     |
| Qwen / Qwen2              | ✅     |

## Quantization Formats

| Format | Bits | Description                 |
| ------ | ---- | --------------------------- |
| Q4_0   | 4    | Basic 4-bit quantization    |
| Q4_K   | 4    | K-quant, better quality     |
| Q5_K   | 5    | K-quant, good balance       |
| Q6_K   | 6    | K-quant, near-FP16 quality  |
| Q8_0   | 8    | 8-bit, minimal quality loss |

## Project Structure

```
ODIEngine/
├── include/odi/          # Public C API headers
│   ├── odi_engine.h      # Main API
│   ├── odi_types.h       # Types and configs
│   └── odi_error.h       # Error codes
├── src/
│   ├── core/             # Engine, Model, Context
│   ├── format/           # GGUF parser
│   ├── tensor/           # Tensor operations
│   └── compute/          # Backend implementations
│       └── cpu/          # CPU with SIMD
├── examples/             # Usage examples
└── tests/                # Unit tests
```

## Roadmap

- [x] GGUF parser
- [x] CPU backend with SIMD (AVX2/NEON)
- [x] INT4/INT8 quantized inference
- [x] Streaming generation
- [x] Sampling strategies (temperature, top-k, top-p, mirostat)
- [ ] Metal backend (Apple Silicon)
- [ ] Vulkan backend (cross-platform GPU)
- [ ] CUDA backend (NVIDIA)
- [ ] Flash attention
- [ ] Speculative decoding
- [ ] Continuous batching

## License

[Apache 2.0](LICENSE)

## Acknowledgments

Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) and the broader open-source LLM community.

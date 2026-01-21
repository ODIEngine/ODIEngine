#ifndef ODI_TYPES_H
#define ODI_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version info */
#define ODI_VERSION_MAJOR 0
#define ODI_VERSION_MINOR 1
#define ODI_VERSION_PATCH 0

/* Export macros */
#ifdef _WIN32
    #ifdef ODI_BUILD_SHARED
        #define ODI_API __declspec(dllexport)
    #else
        #define ODI_API __declspec(dllimport)
    #endif
#else
    #define ODI_API __attribute__((visibility("default")))
#endif

/* Maximum dimensions for tensors */
#define ODI_MAX_DIMS 8

/* Data types */
typedef enum odi_dtype {
    ODI_DTYPE_F32 = 0,      /* 32-bit float */
    ODI_DTYPE_F16 = 1,      /* 16-bit float (IEEE 754) */
    ODI_DTYPE_BF16 = 2,     /* Brain float 16 */
    ODI_DTYPE_I32 = 3,      /* 32-bit signed integer */
    ODI_DTYPE_I16 = 4,      /* 16-bit signed integer */
    ODI_DTYPE_I8 = 5,       /* 8-bit signed integer */
    ODI_DTYPE_Q8_0 = 6,     /* Quantized 8-bit (block size 32) */
    ODI_DTYPE_Q4_0 = 7,     /* Quantized 4-bit (block size 32) */
    ODI_DTYPE_Q4_1 = 8,     /* Quantized 4-bit with min (block size 32) */
    ODI_DTYPE_Q5_0 = 9,     /* Quantized 5-bit (block size 32) */
    ODI_DTYPE_Q5_1 = 10,    /* Quantized 5-bit with min (block size 32) */
    ODI_DTYPE_Q4_K = 11,    /* K-quant 4-bit */
    ODI_DTYPE_Q5_K = 12,    /* K-quant 5-bit */
    ODI_DTYPE_Q6_K = 13,    /* K-quant 6-bit */
    ODI_DTYPE_Q8_K = 14,    /* K-quant 8-bit */
    ODI_DTYPE_COUNT
} odi_dtype_t;

/* Compute backend types */
typedef enum odi_backend_type {
    ODI_BACKEND_AUTO = 0,   /* Auto-select best available */
    ODI_BACKEND_CPU = 1,    /* CPU with SIMD */
    ODI_BACKEND_METAL = 2,  /* Apple Metal */
    ODI_BACKEND_VULKAN = 3, /* Vulkan */
    ODI_BACKEND_CUDA = 4,   /* NVIDIA CUDA */
    ODI_BACKEND_COUNT
} odi_backend_type_t;

/* Model architecture types */
typedef enum odi_arch_type {
    ODI_ARCH_UNKNOWN = 0,
    ODI_ARCH_LLAMA = 1,
    ODI_ARCH_MISTRAL = 2,
    ODI_ARCH_GEMMA = 3,
    ODI_ARCH_PHI = 4,
    ODI_ARCH_QWEN = 5,
    ODI_ARCH_GPT2 = 6,
    ODI_ARCH_COUNT
} odi_arch_type_t;

/* Opaque types */
typedef struct odi_engine odi_engine_t;
typedef struct odi_model odi_model_t;
typedef struct odi_context odi_context_t;
typedef struct odi_tensor odi_tensor_t;

/* Engine configuration */
typedef struct odi_engine_config {
    odi_backend_type_t backend;     /* Preferred backend */
    int num_threads;                /* Number of threads (0 = auto) */
    size_t memory_limit;            /* Max memory usage in bytes (0 = unlimited) */
    bool use_mmap;                  /* Use memory-mapped files */
    bool verbose;                   /* Enable verbose logging */
} odi_engine_config_t;

/* Model loading options */
typedef struct odi_model_config {
    bool use_mmap;                  /* Memory-map the model file */
    int gpu_layers;                 /* Number of layers to offload to GPU (-1 = all) */
    odi_dtype_t compute_dtype;      /* Compute precision (F32, F16) */
} odi_model_config_t;

/* Inference context configuration */
typedef struct odi_context_config {
    size_t context_length;          /* Maximum context length */
    size_t batch_size;              /* Batch size for parallel inference */
    bool flash_attention;           /* Use flash attention if available */
} odi_context_config_t;

/* Sampling parameters */
typedef struct odi_sampler_config {
    float temperature;              /* Sampling temperature (default: 0.8) */
    int top_k;                      /* Top-k sampling (default: 40, 0 = disabled) */
    float top_p;                    /* Top-p/nucleus sampling (default: 0.95) */
    float repeat_penalty;           /* Repetition penalty (default: 1.1) */
    int repeat_last_n;              /* Tokens to consider for repeat penalty */
    uint64_t seed;                  /* Random seed (0 = random) */
    /* Mirostat sampling */
    int mirostat;                   /* Mirostat mode (0=disabled, 1=v1, 2=v2) */
    float mirostat_tau;             /* Target entropy */
    float mirostat_eta;             /* Learning rate */
} odi_sampler_config_t;

/* Generation result */
typedef struct odi_generation_result {
    int tokens_generated;           /* Number of tokens generated */
    int tokens_evaluated;           /* Number of prompt tokens processed */
    float generation_time_ms;       /* Time for generation in ms */
    float tokens_per_second;        /* Generation speed */
    int stop_reason;                /* Why generation stopped */
} odi_generation_result_t;

/* Stop reasons */
typedef enum odi_stop_reason {
    ODI_STOP_NONE = 0,
    ODI_STOP_EOS = 1,               /* End of sequence token */
    ODI_STOP_MAX_TOKENS = 2,        /* Reached max tokens */
    ODI_STOP_USER = 3,              /* User requested stop */
    ODI_STOP_ERROR = 4              /* Error occurred */
} odi_stop_reason_t;

/* Token callback for streaming */
typedef bool (*odi_token_callback_t)(const char* token, size_t token_len, void* user_data);

/* Progress callback for model loading */
typedef void (*odi_progress_callback_t)(float progress, void* user_data);

/* Helper functions for default configs */
static inline odi_engine_config_t odi_engine_config_default(void) {
    odi_engine_config_t config = {
        .backend = ODI_BACKEND_AUTO,
        .num_threads = 0,
        .memory_limit = 0,
        .use_mmap = true,
        .verbose = false
    };
    return config;
}

static inline odi_model_config_t odi_model_config_default(void) {
    odi_model_config_t config = {
        .use_mmap = true,
        .gpu_layers = 0,
        .compute_dtype = ODI_DTYPE_F32
    };
    return config;
}

static inline odi_context_config_t odi_context_config_default(void) {
    odi_context_config_t config = {
        .context_length = 2048,
        .batch_size = 1,
        .flash_attention = true
    };
    return config;
}

static inline odi_sampler_config_t odi_sampler_config_default(void) {
    odi_sampler_config_t config = {
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.95f,
        .repeat_penalty = 1.1f,
        .repeat_last_n = 64,
        .seed = 0,
        .mirostat = 0,
        .mirostat_tau = 5.0f,
        .mirostat_eta = 0.1f
    };
    return config;
}

/* Utility functions */
ODI_API size_t odi_dtype_size(odi_dtype_t dtype);
ODI_API const char* odi_dtype_name(odi_dtype_t dtype);
ODI_API const char* odi_backend_name(odi_backend_type_t backend);
ODI_API const char* odi_arch_name(odi_arch_type_t arch);

#ifdef __cplusplus
}
#endif

#endif /* ODI_TYPES_H */

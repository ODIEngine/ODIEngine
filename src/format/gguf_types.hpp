#pragma once

#include <cstdint>
#include <cstddef>

namespace odi {

// GGUF magic number: "GGUF" in little-endian
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"

// Supported GGUF versions
constexpr uint32_t GGUF_VERSION_MIN = 2;
constexpr uint32_t GGUF_VERSION_MAX = 3;

// GGUF metadata value types
enum class GGUFMetadataType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// GGUF tensor types (matching GGML types)
enum class GGUFTensorType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 and Q4_3 removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    BF16    = 29,
    COUNT
};

// GGUF file header (version 2/3)
struct GGUFHeader {
    uint32_t magic;             // GGUF_MAGIC
    uint32_t version;           // Format version (2 or 3)
    uint64_t tensor_count;      // Number of tensors
    uint64_t metadata_kv_count; // Number of metadata key-value pairs
};

// String in GGUF format
struct GGUFString {
    uint64_t length;
    // char data[length] follows (not null-terminated in file)
};

// Note: GGUFTensorInfo is defined in gguf_parser.hpp with full implementation

// Common GGUF metadata keys
namespace GGUFKeys {
    // General
    constexpr const char* GENERAL_ARCHITECTURE = "general.architecture";
    constexpr const char* GENERAL_NAME = "general.name";
    constexpr const char* GENERAL_AUTHOR = "general.author";
    constexpr const char* GENERAL_QUANTIZATION_VERSION = "general.quantization_version";
    constexpr const char* GENERAL_FILE_TYPE = "general.file_type";

    // LLM architecture
    constexpr const char* CONTEXT_LENGTH = ".context_length";
    constexpr const char* EMBEDDING_LENGTH = ".embedding_length";
    constexpr const char* BLOCK_COUNT = ".block_count";
    constexpr const char* FEED_FORWARD_LENGTH = ".feed_forward_length";
    constexpr const char* ATTENTION_HEAD_COUNT = ".attention.head_count";
    constexpr const char* ATTENTION_HEAD_COUNT_KV = ".attention.head_count_kv";
    constexpr const char* ATTENTION_LAYER_NORM_RMS_EPSILON = ".attention.layer_norm_rms_epsilon";
    constexpr const char* ROPE_FREQ_BASE = ".rope.freq_base";
    constexpr const char* ROPE_DIMENSION_COUNT = ".rope.dimension_count";

    // Tokenizer
    constexpr const char* TOKENIZER_MODEL = "tokenizer.ggml.model";
    constexpr const char* TOKENIZER_TOKENS = "tokenizer.ggml.tokens";
    constexpr const char* TOKENIZER_TOKEN_TYPE = "tokenizer.ggml.token_type";
    constexpr const char* TOKENIZER_SCORES = "tokenizer.ggml.scores";
    constexpr const char* TOKENIZER_MERGES = "tokenizer.ggml.merges";
    constexpr const char* TOKENIZER_BOS_ID = "tokenizer.ggml.bos_token_id";
    constexpr const char* TOKENIZER_EOS_ID = "tokenizer.ggml.eos_token_id";
    constexpr const char* TOKENIZER_PAD_ID = "tokenizer.ggml.padding_token_id";
    constexpr const char* TOKENIZER_UNK_ID = "tokenizer.ggml.unknown_token_id";
}

// Quantized block structures (matching GGML)
#pragma pack(push, 1)

// Q4_0: 4-bit quantization, 32 weights per block
struct BlockQ4_0 {
    uint16_t d;          // delta (fp16)
    uint8_t qs[16];      // 32 x 4-bit weights (packed)
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 size mismatch");

// Q4_1: 4-bit quantization with min, 32 weights per block
struct BlockQ4_1 {
    uint16_t d;          // delta (fp16)
    uint16_t m;          // min (fp16)
    uint8_t qs[16];      // 32 x 4-bit weights (packed)
};
static_assert(sizeof(BlockQ4_1) == 20, "BlockQ4_1 size mismatch");

// Q5_0: 5-bit quantization, 32 weights per block
struct BlockQ5_0 {
    uint16_t d;          // delta (fp16)
    uint8_t qh[4];       // 32 x 5th bit (packed)
    uint8_t qs[16];      // 32 x low 4-bit weights (packed)
};
static_assert(sizeof(BlockQ5_0) == 22, "BlockQ5_0 size mismatch");

// Q5_1: 5-bit quantization with min, 32 weights per block
struct BlockQ5_1 {
    uint16_t d;          // delta (fp16)
    uint16_t m;          // min (fp16)
    uint8_t qh[4];       // 32 x 5th bit (packed)
    uint8_t qs[16];      // 32 x low 4-bit weights (packed)
};
static_assert(sizeof(BlockQ5_1) == 24, "BlockQ5_1 size mismatch");

// Q8_0: 8-bit quantization, 32 weights per block
struct BlockQ8_0 {
    uint16_t d;          // delta (fp16)
    int8_t qs[32];       // 32 x 8-bit weights
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 size mismatch");

// Q8_1: 8-bit quantization with sum, 32 weights per block
struct BlockQ8_1 {
    float d;             // delta (fp32)
    float s;             // sum of all weights
    int8_t qs[32];       // 32 x 8-bit weights
};
static_assert(sizeof(BlockQ8_1) == 40, "BlockQ8_1 size mismatch");

#pragma pack(pop)

// Get bytes per block for a tensor type
inline size_t gguf_tensor_type_size(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::F32:     return 4;
        case GGUFTensorType::F16:     return 2;
        case GGUFTensorType::BF16:    return 2;
        case GGUFTensorType::I8:      return 1;
        case GGUFTensorType::I16:     return 2;
        case GGUFTensorType::I32:     return 4;
        case GGUFTensorType::I64:     return 8;
        case GGUFTensorType::F64:     return 8;
        case GGUFTensorType::Q4_0:    return sizeof(BlockQ4_0);
        case GGUFTensorType::Q4_1:    return sizeof(BlockQ4_1);
        case GGUFTensorType::Q5_0:    return sizeof(BlockQ5_0);
        case GGUFTensorType::Q5_1:    return sizeof(BlockQ5_1);
        case GGUFTensorType::Q8_0:    return sizeof(BlockQ8_0);
        case GGUFTensorType::Q8_1:    return sizeof(BlockQ8_1);
        // K-quants have variable sizes
        case GGUFTensorType::Q2_K:    return 256/16 * 2 + 256/4 + 2 + 2;  // ~84
        case GGUFTensorType::Q3_K:    return 256/8 * 4 + 256/4 + 12;      // ~110
        case GGUFTensorType::Q4_K:    return 2 + 2 + 12 + 256/2;          // ~144
        case GGUFTensorType::Q5_K:    return 2 + 2 + 12 + 256/8 + 256/2;  // ~176
        case GGUFTensorType::Q6_K:    return 256/2 + 256/4 + 256/16 + 2;  // ~210
        case GGUFTensorType::Q8_K:    return 4 + 256 + 16*2;              // ~292
        default:                      return 0;
    }
}

// Get elements per block for a tensor type
inline int gguf_tensor_type_block_size(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::F32:
        case GGUFTensorType::F16:
        case GGUFTensorType::BF16:
        case GGUFTensorType::I8:
        case GGUFTensorType::I16:
        case GGUFTensorType::I32:
        case GGUFTensorType::I64:
        case GGUFTensorType::F64:
            return 1;
        case GGUFTensorType::Q4_0:
        case GGUFTensorType::Q4_1:
        case GGUFTensorType::Q5_0:
        case GGUFTensorType::Q5_1:
        case GGUFTensorType::Q8_0:
        case GGUFTensorType::Q8_1:
            return 32;
        case GGUFTensorType::Q2_K:
        case GGUFTensorType::Q3_K:
        case GGUFTensorType::Q4_K:
        case GGUFTensorType::Q5_K:
        case GGUFTensorType::Q6_K:
        case GGUFTensorType::Q8_K:
            return 256;
        default:
            return 1;
    }
}

// Get name of tensor type
inline const char* gguf_tensor_type_name(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::F32:     return "F32";
        case GGUFTensorType::F16:     return "F16";
        case GGUFTensorType::BF16:    return "BF16";
        case GGUFTensorType::I8:      return "I8";
        case GGUFTensorType::I16:     return "I16";
        case GGUFTensorType::I32:     return "I32";
        case GGUFTensorType::I64:     return "I64";
        case GGUFTensorType::F64:     return "F64";
        case GGUFTensorType::Q4_0:    return "Q4_0";
        case GGUFTensorType::Q4_1:    return "Q4_1";
        case GGUFTensorType::Q5_0:    return "Q5_0";
        case GGUFTensorType::Q5_1:    return "Q5_1";
        case GGUFTensorType::Q8_0:    return "Q8_0";
        case GGUFTensorType::Q8_1:    return "Q8_1";
        case GGUFTensorType::Q2_K:    return "Q2_K";
        case GGUFTensorType::Q3_K:    return "Q3_K";
        case GGUFTensorType::Q4_K:    return "Q4_K";
        case GGUFTensorType::Q5_K:    return "Q5_K";
        case GGUFTensorType::Q6_K:    return "Q6_K";
        case GGUFTensorType::Q8_K:    return "Q8_K";
        default:                      return "UNKNOWN";
    }
}

} // namespace odi

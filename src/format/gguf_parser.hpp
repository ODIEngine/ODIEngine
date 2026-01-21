#pragma once

#include "gguf_types.hpp"
#include "../tensor/tensor.hpp"
#include <odi/odi_types.h>
#include <odi/odi_error.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>
#include <functional>

namespace odi {

// Forward declarations
class GGUFFile;

// Metadata value variant type
using GGUFValue = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    uint64_t,
    int64_t,
    float,
    double,
    bool,
    std::string,
    std::vector<uint8_t>,
    std::vector<int8_t>,
    std::vector<uint16_t>,
    std::vector<int16_t>,
    std::vector<uint32_t>,
    std::vector<int32_t>,
    std::vector<uint64_t>,
    std::vector<int64_t>,
    std::vector<float>,
    std::vector<double>,
    std::vector<bool>,
    std::vector<std::string>
>;

// Parsed tensor information
struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    GGUFTensorType type;
    uint64_t offset;        // Offset from start of tensor data section
    size_t size_bytes;      // Total size in bytes

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// Model architecture info extracted from metadata
struct GGUFModelInfo {
    std::string architecture;   // e.g., "llama", "mistral", "gemma"
    std::string name;           // Model name
    std::string author;         // Model author

    // Architecture parameters
    int64_t context_length = 0;
    int64_t embedding_length = 0;
    int64_t block_count = 0;    // Number of transformer layers
    int64_t feed_forward_length = 0;
    int64_t attention_head_count = 0;
    int64_t attention_head_count_kv = 0;
    float attention_layer_norm_rms_epsilon = 1e-5f;
    float rope_freq_base = 10000.0f;
    int64_t rope_dimension_count = 0;

    // Vocabulary
    int64_t vocab_size = 0;
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    int32_t pad_token_id = -1;
    int32_t unk_token_id = -1;

    // Get odi_arch_type_t from architecture string
    odi_arch_type_t get_arch_type() const;
};

/**
 * GGUF File Parser
 *
 * Parses GGUF files and provides access to metadata and tensors.
 * Supports both memory-mapped and loaded modes.
 */
class GGUFFile {
public:
    // Open a GGUF file
    // If use_mmap is true, the file is memory-mapped (faster, less memory for large files)
    // If use_mmap is false, tensors are loaded into memory when requested
    static std::unique_ptr<GGUFFile> open(const std::string& path, bool use_mmap = true);

    ~GGUFFile();

    // Prevent copying
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // Move support
    GGUFFile(GGUFFile&&) noexcept;
    GGUFFile& operator=(GGUFFile&&) noexcept;

    // File info
    const std::string& path() const { return path_; }
    uint32_t version() const { return version_; }
    bool is_mmap() const { return mmap_data_ != nullptr; }

    // Metadata access
    size_t metadata_count() const { return metadata_.size(); }
    bool has_metadata(const std::string& key) const;
    const GGUFValue* get_metadata(const std::string& key) const;

    // Typed metadata getters (return default if not found or wrong type)
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    int64_t get_int(const std::string& key, int64_t default_val = 0) const;
    uint64_t get_uint(const std::string& key, uint64_t default_val = 0) const;
    float get_float(const std::string& key, float default_val = 0.0f) const;
    bool get_bool(const std::string& key, bool default_val = false) const;
    std::vector<std::string> get_string_array(const std::string& key) const;
    std::vector<float> get_float_array(const std::string& key) const;
    std::vector<int32_t> get_int_array(const std::string& key) const;

    // Get all metadata keys
    std::vector<std::string> metadata_keys() const;

    // Tensor access
    size_t tensor_count() const { return tensors_.size(); }
    bool has_tensor(const std::string& name) const;
    const GGUFTensorInfo* get_tensor_info(const std::string& name) const;

    // Get all tensor names
    std::vector<std::string> tensor_names() const;

    // Load a tensor by name
    // Returns empty tensor if not found
    Tensor load_tensor(const std::string& name) const;

    // Load a tensor as a specific dtype (converts if necessary)
    Tensor load_tensor_as(const std::string& name, odi_dtype_t dtype) const;

    // Get raw pointer to tensor data (only works with mmap)
    const void* tensor_data_ptr(const std::string& name) const;

    // Get model info
    const GGUFModelInfo& model_info() const { return model_info_; }

    // Get vocabulary tokens (if available)
    const std::vector<std::string>& vocab_tokens() const { return vocab_tokens_; }

    // Get token scores (if available)
    const std::vector<float>& vocab_scores() const { return vocab_scores_; }

    // Get BPE merges (if available)
    const std::vector<std::string>& vocab_merges() const { return vocab_merges_; }

    // Print file summary
    std::string summary() const;

private:
    GGUFFile() = default;

    // Parse the file
    odi_error_t parse();

    // Parse metadata section
    odi_error_t parse_metadata();

    // Parse tensor info section
    odi_error_t parse_tensors();

    // Extract model info from metadata
    void extract_model_info();

    // Read helpers
    template<typename T>
    T read();

    std::string read_string();
    GGUFValue read_value(GGUFMetadataType type);

    // Convert GGUF tensor type to odi dtype
    static odi_dtype_t gguf_to_odi_dtype(GGUFTensorType type);

    std::string path_;
    uint32_t version_ = 0;

    // File handle or mmap
    void* mmap_data_ = nullptr;
    size_t mmap_size_ = 0;
    int fd_ = -1;

    // Current read position (for parsing)
    const uint8_t* read_ptr_ = nullptr;
    const uint8_t* end_ptr_ = nullptr;

    // Tensor data section start
    const uint8_t* tensor_data_start_ = nullptr;

    // Parsed data
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::unordered_map<std::string, GGUFTensorInfo> tensors_;
    std::vector<std::string> tensor_names_;  // Preserve order

    // Extracted info
    GGUFModelInfo model_info_;
    std::vector<std::string> vocab_tokens_;
    std::vector<float> vocab_scores_;
    std::vector<std::string> vocab_merges_;
};

} // namespace odi

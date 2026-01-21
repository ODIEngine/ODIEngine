#include "gguf_parser.hpp"

#include <cstring>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace odi {

// ============================================================================
// GGUFModelInfo
// ============================================================================

odi_arch_type_t GGUFModelInfo::get_arch_type() const {
    if (architecture == "llama") return ODI_ARCH_LLAMA;
    if (architecture == "mistral") return ODI_ARCH_MISTRAL;
    if (architecture == "gemma") return ODI_ARCH_GEMMA;
    if (architecture == "phi" || architecture == "phi2" || architecture == "phi3") return ODI_ARCH_PHI;
    if (architecture == "qwen" || architecture == "qwen2") return ODI_ARCH_QWEN;
    if (architecture == "gpt2") return ODI_ARCH_GPT2;
    return ODI_ARCH_UNKNOWN;
}

// ============================================================================
// GGUFFile
// ============================================================================

std::unique_ptr<GGUFFile> GGUFFile::open(const std::string& path, bool use_mmap) {
    auto file = std::unique_ptr<GGUFFile>(new GGUFFile());
    file->path_ = path;

#ifdef _WIN32
    // Windows implementation
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        odi_set_error(ODI_ERROR_FILE_NOT_FOUND, "Failed to open file");
        return nullptr;
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(hFile, &fileSize)) {
        CloseHandle(hFile);
        odi_set_error(ODI_ERROR_FILE_READ, "Failed to get file size");
        return nullptr;
    }

    file->mmap_size_ = static_cast<size_t>(fileSize.QuadPart);

    if (use_mmap) {
        HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (hMapping == nullptr) {
            CloseHandle(hFile);
            odi_set_error(ODI_ERROR_MMAP_FAILED, "Failed to create file mapping");
            return nullptr;
        }

        file->mmap_data_ = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMapping);
        CloseHandle(hFile);

        if (file->mmap_data_ == nullptr) {
            odi_set_error(ODI_ERROR_MMAP_FAILED, "Failed to map file");
            return nullptr;
        }
    } else {
        // Load entire file into memory
        file->mmap_data_ = malloc(file->mmap_size_);
        if (file->mmap_data_ == nullptr) {
            CloseHandle(hFile);
            odi_set_error(ODI_ERROR_OUT_OF_MEMORY, "Failed to allocate memory");
            return nullptr;
        }

        DWORD bytesRead;
        if (!ReadFile(hFile, file->mmap_data_, static_cast<DWORD>(file->mmap_size_), &bytesRead, nullptr)) {
            free(file->mmap_data_);
            CloseHandle(hFile);
            odi_set_error(ODI_ERROR_FILE_READ, "Failed to read file");
            return nullptr;
        }
        CloseHandle(hFile);
    }
#else
    // Unix implementation
    file->fd_ = ::open(path.c_str(), O_RDONLY);
    if (file->fd_ < 0) {
        odi_set_error(ODI_ERROR_FILE_NOT_FOUND, "Failed to open file");
        return nullptr;
    }

    struct stat st;
    if (fstat(file->fd_, &st) < 0) {
        ::close(file->fd_);
        odi_set_error(ODI_ERROR_FILE_READ, "Failed to get file size");
        return nullptr;
    }

    file->mmap_size_ = static_cast<size_t>(st.st_size);

    if (use_mmap) {
        file->mmap_data_ = mmap(nullptr, file->mmap_size_, PROT_READ, MAP_PRIVATE, file->fd_, 0);
        if (file->mmap_data_ == MAP_FAILED) {
            file->mmap_data_ = nullptr;
            ::close(file->fd_);
            odi_set_error(ODI_ERROR_MMAP_FAILED, "Failed to mmap file");
            return nullptr;
        }
        // Advise sequential access
        madvise(file->mmap_data_, file->mmap_size_, MADV_SEQUENTIAL);
    } else {
        // Load entire file into memory
        file->mmap_data_ = malloc(file->mmap_size_);
        if (file->mmap_data_ == nullptr) {
            ::close(file->fd_);
            odi_set_error(ODI_ERROR_OUT_OF_MEMORY, "Failed to allocate memory");
            return nullptr;
        }

        size_t total_read = 0;
        while (total_read < file->mmap_size_) {
            ssize_t n = ::read(file->fd_, static_cast<char*>(file->mmap_data_) + total_read,
                              file->mmap_size_ - total_read);
            if (n <= 0) {
                free(file->mmap_data_);
                ::close(file->fd_);
                odi_set_error(ODI_ERROR_FILE_READ, "Failed to read file");
                return nullptr;
            }
            total_read += n;
        }
        ::close(file->fd_);
        file->fd_ = -1;
    }
#endif

    // Set up read pointers
    file->read_ptr_ = static_cast<const uint8_t*>(file->mmap_data_);
    file->end_ptr_ = file->read_ptr_ + file->mmap_size_;

    // Parse the file
    odi_error_t err = file->parse();
    if (err != ODI_SUCCESS) {
        return nullptr;
    }

    return file;
}

GGUFFile::~GGUFFile() {
    if (mmap_data_) {
#ifdef _WIN32
        if (fd_ == -1) {
            // Memory was allocated, not mapped
            free(mmap_data_);
        } else {
            UnmapViewOfFile(mmap_data_);
        }
#else
        if (fd_ != -1) {
            munmap(mmap_data_, mmap_size_);
            ::close(fd_);
        } else {
            free(mmap_data_);
        }
#endif
    }
}

GGUFFile::GGUFFile(GGUFFile&& other) noexcept
    : path_(std::move(other.path_))
    , version_(other.version_)
    , mmap_data_(other.mmap_data_)
    , mmap_size_(other.mmap_size_)
    , fd_(other.fd_)
    , read_ptr_(other.read_ptr_)
    , end_ptr_(other.end_ptr_)
    , tensor_data_start_(other.tensor_data_start_)
    , metadata_(std::move(other.metadata_))
    , tensors_(std::move(other.tensors_))
    , tensor_names_(std::move(other.tensor_names_))
    , model_info_(std::move(other.model_info_))
    , vocab_tokens_(std::move(other.vocab_tokens_))
    , vocab_scores_(std::move(other.vocab_scores_))
    , vocab_merges_(std::move(other.vocab_merges_)) {
    other.mmap_data_ = nullptr;
    other.fd_ = -1;
}

GGUFFile& GGUFFile::operator=(GGUFFile&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        if (mmap_data_) {
#ifdef _WIN32
            if (fd_ == -1) {
                free(mmap_data_);
            } else {
                UnmapViewOfFile(mmap_data_);
            }
#else
            if (fd_ != -1) {
                munmap(mmap_data_, mmap_size_);
                ::close(fd_);
            } else {
                free(mmap_data_);
            }
#endif
        }

        path_ = std::move(other.path_);
        version_ = other.version_;
        mmap_data_ = other.mmap_data_;
        mmap_size_ = other.mmap_size_;
        fd_ = other.fd_;
        read_ptr_ = other.read_ptr_;
        end_ptr_ = other.end_ptr_;
        tensor_data_start_ = other.tensor_data_start_;
        metadata_ = std::move(other.metadata_);
        tensors_ = std::move(other.tensors_);
        tensor_names_ = std::move(other.tensor_names_);
        model_info_ = std::move(other.model_info_);
        vocab_tokens_ = std::move(other.vocab_tokens_);
        vocab_scores_ = std::move(other.vocab_scores_);
        vocab_merges_ = std::move(other.vocab_merges_);

        other.mmap_data_ = nullptr;
        other.fd_ = -1;
    }
    return *this;
}

template<typename T>
T GGUFFile::read() {
    if (read_ptr_ + sizeof(T) > end_ptr_) {
        throw std::runtime_error("Unexpected end of file");
    }
    T value;
    std::memcpy(&value, read_ptr_, sizeof(T));
    read_ptr_ += sizeof(T);
    return value;
}

std::string GGUFFile::read_string() {
    uint64_t len = read<uint64_t>();
    if (read_ptr_ + len > end_ptr_) {
        throw std::runtime_error("Unexpected end of file reading string");
    }
    std::string str(reinterpret_cast<const char*>(read_ptr_), len);
    read_ptr_ += len;
    return str;
}

GGUFValue GGUFFile::read_value(GGUFMetadataType type) {
    switch (type) {
        case GGUFMetadataType::UINT8:   return read<uint8_t>();
        case GGUFMetadataType::INT8:    return read<int8_t>();
        case GGUFMetadataType::UINT16:  return read<uint16_t>();
        case GGUFMetadataType::INT16:   return read<int16_t>();
        case GGUFMetadataType::UINT32:  return read<uint32_t>();
        case GGUFMetadataType::INT32:   return read<int32_t>();
        case GGUFMetadataType::UINT64:  return read<uint64_t>();
        case GGUFMetadataType::INT64:   return read<int64_t>();
        case GGUFMetadataType::FLOAT32: return read<float>();
        case GGUFMetadataType::FLOAT64: return read<double>();
        case GGUFMetadataType::BOOL:    return static_cast<bool>(read<uint8_t>());
        case GGUFMetadataType::STRING:  return read_string();

        case GGUFMetadataType::ARRAY: {
            GGUFMetadataType elem_type = static_cast<GGUFMetadataType>(read<uint32_t>());
            uint64_t count = read<uint64_t>();

            switch (elem_type) {
                case GGUFMetadataType::UINT8: {
                    std::vector<uint8_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<uint8_t>();
                    return arr;
                }
                case GGUFMetadataType::INT8: {
                    std::vector<int8_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<int8_t>();
                    return arr;
                }
                case GGUFMetadataType::UINT16: {
                    std::vector<uint16_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<uint16_t>();
                    return arr;
                }
                case GGUFMetadataType::INT16: {
                    std::vector<int16_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<int16_t>();
                    return arr;
                }
                case GGUFMetadataType::UINT32: {
                    std::vector<uint32_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<uint32_t>();
                    return arr;
                }
                case GGUFMetadataType::INT32: {
                    std::vector<int32_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<int32_t>();
                    return arr;
                }
                case GGUFMetadataType::UINT64: {
                    std::vector<uint64_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<uint64_t>();
                    return arr;
                }
                case GGUFMetadataType::INT64: {
                    std::vector<int64_t> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<int64_t>();
                    return arr;
                }
                case GGUFMetadataType::FLOAT32: {
                    std::vector<float> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<float>();
                    return arr;
                }
                case GGUFMetadataType::FLOAT64: {
                    std::vector<double> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read<double>();
                    return arr;
                }
                case GGUFMetadataType::BOOL: {
                    std::vector<bool> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = static_cast<bool>(read<uint8_t>());
                    return arr;
                }
                case GGUFMetadataType::STRING: {
                    std::vector<std::string> arr(count);
                    for (uint64_t i = 0; i < count; ++i) arr[i] = read_string();
                    return arr;
                }
                default:
                    throw std::runtime_error("Unsupported array element type");
            }
        }

        default:
            throw std::runtime_error("Unsupported metadata type");
    }
}

odi_error_t GGUFFile::parse() {
    try {
        // Read header
        GGUFHeader header;
        header.magic = read<uint32_t>();

        if (header.magic != GGUF_MAGIC) {
            odi_set_error(ODI_ERROR_MODEL_INVALID, "Invalid GGUF magic number");
            return ODI_ERROR_MODEL_INVALID;
        }

        header.version = read<uint32_t>();
        version_ = header.version;

        if (version_ < GGUF_VERSION_MIN || version_ > GGUF_VERSION_MAX) {
            odi_set_error(ODI_ERROR_MODEL_VERSION, "Unsupported GGUF version");
            return ODI_ERROR_MODEL_VERSION;
        }

        header.tensor_count = read<uint64_t>();
        header.metadata_kv_count = read<uint64_t>();

        // Parse metadata
        odi_error_t err = parse_metadata();
        if (err != ODI_SUCCESS) return err;

        // Parse tensor info
        err = parse_tensors();
        if (err != ODI_SUCCESS) return err;

        // Extract model info
        extract_model_info();

        return ODI_SUCCESS;
    } catch (const std::exception& e) {
        odi_set_error(ODI_ERROR_MODEL_CORRUPTED, e.what());
        return ODI_ERROR_MODEL_CORRUPTED;
    }
}

odi_error_t GGUFFile::parse_metadata() {
    // Re-read header to get counts
    const uint8_t* saved_ptr = read_ptr_;
    read_ptr_ = static_cast<const uint8_t*>(mmap_data_) + 8;  // Skip magic and version
    uint64_t tensor_count = read<uint64_t>();
    uint64_t metadata_count = read<uint64_t>();

    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key = read_string();
        GGUFMetadataType type = static_cast<GGUFMetadataType>(read<uint32_t>());
        GGUFValue value = read_value(type);
        metadata_[key] = std::move(value);
    }

    return ODI_SUCCESS;
}

odi_error_t GGUFFile::parse_tensors() {
    // Re-read header to get tensor count
    const uint8_t* header_ptr = static_cast<const uint8_t*>(mmap_data_) + 8;
    uint64_t tensor_count;
    std::memcpy(&tensor_count, header_ptr, sizeof(uint64_t));

    // read_ptr_ should now be at tensor info section
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info;
        info.name = read_string();

        uint32_t n_dims = read<uint32_t>();
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.shape[d] = static_cast<int64_t>(read<uint64_t>());
        }

        info.type = static_cast<GGUFTensorType>(read<uint32_t>());
        info.offset = read<uint64_t>();

        // Calculate size
        int64_t numel = info.numel();
        int block_size = gguf_tensor_type_block_size(info.type);
        int64_t num_blocks = (numel + block_size - 1) / block_size;
        info.size_bytes = num_blocks * gguf_tensor_type_size(info.type);

        tensor_names_.push_back(info.name);
        tensors_[info.name] = std::move(info);
    }

    // Align to 32 bytes for tensor data
    size_t current_offset = read_ptr_ - static_cast<const uint8_t*>(mmap_data_);
    size_t alignment = 32;
    size_t aligned_offset = (current_offset + alignment - 1) & ~(alignment - 1);
    tensor_data_start_ = static_cast<const uint8_t*>(mmap_data_) + aligned_offset;

    return ODI_SUCCESS;
}

void GGUFFile::extract_model_info() {
    // Architecture
    model_info_.architecture = get_string(GGUFKeys::GENERAL_ARCHITECTURE);
    model_info_.name = get_string(GGUFKeys::GENERAL_NAME);
    model_info_.author = get_string(GGUFKeys::GENERAL_AUTHOR);

    // Build architecture-specific keys
    std::string arch = model_info_.architecture;
    if (!arch.empty()) {
        model_info_.context_length = get_int(arch + GGUFKeys::CONTEXT_LENGTH);
        model_info_.embedding_length = get_int(arch + GGUFKeys::EMBEDDING_LENGTH);
        model_info_.block_count = get_int(arch + GGUFKeys::BLOCK_COUNT);
        model_info_.feed_forward_length = get_int(arch + GGUFKeys::FEED_FORWARD_LENGTH);
        model_info_.attention_head_count = get_int(arch + GGUFKeys::ATTENTION_HEAD_COUNT);
        model_info_.attention_head_count_kv = get_int(arch + GGUFKeys::ATTENTION_HEAD_COUNT_KV);
        model_info_.attention_layer_norm_rms_epsilon = get_float(arch + GGUFKeys::ATTENTION_LAYER_NORM_RMS_EPSILON, 1e-5f);
        model_info_.rope_freq_base = get_float(arch + GGUFKeys::ROPE_FREQ_BASE, 10000.0f);
        model_info_.rope_dimension_count = get_int(arch + GGUFKeys::ROPE_DIMENSION_COUNT);
    }

    // Tokenizer
    vocab_tokens_ = get_string_array(GGUFKeys::TOKENIZER_TOKENS);
    model_info_.vocab_size = static_cast<int64_t>(vocab_tokens_.size());
    vocab_scores_ = get_float_array(GGUFKeys::TOKENIZER_SCORES);
    vocab_merges_ = get_string_array(GGUFKeys::TOKENIZER_MERGES);

    model_info_.bos_token_id = static_cast<int32_t>(get_int(GGUFKeys::TOKENIZER_BOS_ID, -1));
    model_info_.eos_token_id = static_cast<int32_t>(get_int(GGUFKeys::TOKENIZER_EOS_ID, -1));
    model_info_.pad_token_id = static_cast<int32_t>(get_int(GGUFKeys::TOKENIZER_PAD_ID, -1));
    model_info_.unk_token_id = static_cast<int32_t>(get_int(GGUFKeys::TOKENIZER_UNK_ID, -1));
}

bool GGUFFile::has_metadata(const std::string& key) const {
    return metadata_.find(key) != metadata_.end();
}

const GGUFValue* GGUFFile::get_metadata(const std::string& key) const {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? &it->second : nullptr;
}

std::string GGUFFile::get_string(const std::string& key, const std::string& default_val) const {
    auto* val = get_metadata(key);
    if (!val) return default_val;
    if (auto* s = std::get_if<std::string>(val)) return *s;
    return default_val;
}

int64_t GGUFFile::get_int(const std::string& key, int64_t default_val) const {
    auto* val = get_metadata(key);
    if (!val) return default_val;

    // Try various integer types
    if (auto* v = std::get_if<int64_t>(val)) return *v;
    if (auto* v = std::get_if<uint64_t>(val)) return static_cast<int64_t>(*v);
    if (auto* v = std::get_if<int32_t>(val)) return *v;
    if (auto* v = std::get_if<uint32_t>(val)) return *v;
    if (auto* v = std::get_if<int16_t>(val)) return *v;
    if (auto* v = std::get_if<uint16_t>(val)) return *v;
    if (auto* v = std::get_if<int8_t>(val)) return *v;
    if (auto* v = std::get_if<uint8_t>(val)) return *v;

    return default_val;
}

uint64_t GGUFFile::get_uint(const std::string& key, uint64_t default_val) const {
    auto* val = get_metadata(key);
    if (!val) return default_val;

    if (auto* v = std::get_if<uint64_t>(val)) return *v;
    if (auto* v = std::get_if<int64_t>(val)) return static_cast<uint64_t>(*v);
    if (auto* v = std::get_if<uint32_t>(val)) return *v;
    if (auto* v = std::get_if<int32_t>(val)) return static_cast<uint64_t>(*v);
    if (auto* v = std::get_if<uint16_t>(val)) return *v;
    if (auto* v = std::get_if<int16_t>(val)) return static_cast<uint64_t>(*v);
    if (auto* v = std::get_if<uint8_t>(val)) return *v;
    if (auto* v = std::get_if<int8_t>(val)) return static_cast<uint64_t>(*v);

    return default_val;
}

float GGUFFile::get_float(const std::string& key, float default_val) const {
    auto* val = get_metadata(key);
    if (!val) return default_val;

    if (auto* v = std::get_if<float>(val)) return *v;
    if (auto* v = std::get_if<double>(val)) return static_cast<float>(*v);

    return default_val;
}

bool GGUFFile::get_bool(const std::string& key, bool default_val) const {
    auto* val = get_metadata(key);
    if (!val) return default_val;

    if (auto* v = std::get_if<bool>(val)) return *v;
    if (auto* v = std::get_if<uint8_t>(val)) return *v != 0;

    return default_val;
}

std::vector<std::string> GGUFFile::get_string_array(const std::string& key) const {
    auto* val = get_metadata(key);
    if (!val) return {};
    if (auto* arr = std::get_if<std::vector<std::string>>(val)) return *arr;
    return {};
}

std::vector<float> GGUFFile::get_float_array(const std::string& key) const {
    auto* val = get_metadata(key);
    if (!val) return {};
    if (auto* arr = std::get_if<std::vector<float>>(val)) return *arr;
    return {};
}

std::vector<int32_t> GGUFFile::get_int_array(const std::string& key) const {
    auto* val = get_metadata(key);
    if (!val) return {};
    if (auto* arr = std::get_if<std::vector<int32_t>>(val)) return *arr;
    return {};
}

std::vector<std::string> GGUFFile::metadata_keys() const {
    std::vector<std::string> keys;
    keys.reserve(metadata_.size());
    for (const auto& kv : metadata_) {
        keys.push_back(kv.first);
    }
    return keys;
}

bool GGUFFile::has_tensor(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const GGUFTensorInfo* GGUFFile::get_tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    return it != tensors_.end() ? &it->second : nullptr;
}

std::vector<std::string> GGUFFile::tensor_names() const {
    return tensor_names_;
}

odi_dtype_t GGUFFile::gguf_to_odi_dtype(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::F32:  return ODI_DTYPE_F32;
        case GGUFTensorType::F16:  return ODI_DTYPE_F16;
        case GGUFTensorType::BF16: return ODI_DTYPE_BF16;
        case GGUFTensorType::I8:   return ODI_DTYPE_I8;
        case GGUFTensorType::I16:  return ODI_DTYPE_I16;
        case GGUFTensorType::I32:  return ODI_DTYPE_I32;
        case GGUFTensorType::Q4_0: return ODI_DTYPE_Q4_0;
        case GGUFTensorType::Q4_1: return ODI_DTYPE_Q4_1;
        case GGUFTensorType::Q5_0: return ODI_DTYPE_Q5_0;
        case GGUFTensorType::Q5_1: return ODI_DTYPE_Q5_1;
        case GGUFTensorType::Q8_0: return ODI_DTYPE_Q8_0;
        case GGUFTensorType::Q4_K: return ODI_DTYPE_Q4_K;
        case GGUFTensorType::Q5_K: return ODI_DTYPE_Q5_K;
        case GGUFTensorType::Q6_K: return ODI_DTYPE_Q6_K;
        case GGUFTensorType::Q8_K: return ODI_DTYPE_Q8_K;
        default:                   return ODI_DTYPE_F32;
    }
}

Tensor GGUFFile::load_tensor(const std::string& name) const {
    const GGUFTensorInfo* info = get_tensor_info(name);
    if (!info) {
        return Tensor();  // Empty tensor
    }

    odi_dtype_t dtype = gguf_to_odi_dtype(info->type);
    const void* data_ptr = tensor_data_start_ + info->offset;

    // Create tensor with data (view if mmap, copy otherwise)
    Tensor tensor(const_cast<void*>(data_ptr), info->shape, dtype);
    tensor.set_name(name);

    return tensor;
}

Tensor GGUFFile::load_tensor_as(const std::string& name, odi_dtype_t dtype) const {
    Tensor tensor = load_tensor(name);
    if (!tensor.is_valid()) return tensor;

    if (tensor.dtype() == dtype) {
        return tensor;
    }

    return tensor.to(dtype);
}

const void* GGUFFile::tensor_data_ptr(const std::string& name) const {
    const GGUFTensorInfo* info = get_tensor_info(name);
    if (!info || !tensor_data_start_) {
        return nullptr;
    }
    return tensor_data_start_ + info->offset;
}

std::string GGUFFile::summary() const {
    std::ostringstream ss;

    ss << "GGUF File: " << path_ << "\n";
    ss << "Version: " << version_ << "\n";
    ss << "Metadata entries: " << metadata_.size() << "\n";
    ss << "Tensors: " << tensors_.size() << "\n";
    ss << "\n";

    ss << "Model Info:\n";
    ss << "  Architecture: " << model_info_.architecture << "\n";
    ss << "  Name: " << model_info_.name << "\n";
    ss << "  Context length: " << model_info_.context_length << "\n";
    ss << "  Embedding dim: " << model_info_.embedding_length << "\n";
    ss << "  Layers: " << model_info_.block_count << "\n";
    ss << "  Attention heads: " << model_info_.attention_head_count << "\n";
    ss << "  KV heads: " << model_info_.attention_head_count_kv << "\n";
    ss << "  Vocab size: " << model_info_.vocab_size << "\n";
    ss << "\n";

    ss << "Tensors:\n";
    size_t total_bytes = 0;
    for (const auto& name : tensor_names_) {
        const auto& info = tensors_.at(name);
        ss << "  " << name << ": [";
        for (size_t i = 0; i < info.shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << info.shape[i];
        }
        ss << "] " << gguf_tensor_type_name(info.type);
        ss << " (" << (info.size_bytes / 1024.0 / 1024.0) << " MB)\n";
        total_bytes += info.size_bytes;
    }
    ss << "\nTotal tensor size: " << (total_bytes / 1024.0 / 1024.0) << " MB\n";

    return ss.str();
}

} // namespace odi

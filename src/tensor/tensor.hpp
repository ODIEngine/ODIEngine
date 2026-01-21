#pragma once

#include <odi/odi_types.h>
#include <cstdint>
#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace odi {

// Forward declaration
class Tensor;

// Aligned memory allocation
void* aligned_alloc(size_t size, size_t alignment = 64);
void aligned_free(void* ptr);

// Get size of data type in bytes
size_t dtype_size(odi_dtype_t dtype);

// Get size of quantized block in bytes
size_t dtype_block_size(odi_dtype_t dtype);

// Get number of elements per quantized block
int dtype_block_elements(odi_dtype_t dtype);

// Check if dtype is quantized
bool dtype_is_quantized(odi_dtype_t dtype);

/**
 * Tensor - Multi-dimensional array with support for quantized types
 *
 * Supports both owned memory and views (references to external memory).
 * Memory is aligned for SIMD operations.
 */
class Tensor {
public:
    // Maximum number of dimensions
    static constexpr int MAX_DIMS = ODI_MAX_DIMS;

    // Default constructor (empty tensor)
    Tensor();

    // Create tensor with shape and dtype (allocates memory)
    Tensor(const std::vector<int64_t>& shape, odi_dtype_t dtype);

    // Create tensor view over existing memory (no ownership)
    Tensor(void* data, const std::vector<int64_t>& shape, odi_dtype_t dtype);

    // Create tensor view with custom strides
    Tensor(void* data, const std::vector<int64_t>& shape,
           const std::vector<int64_t>& strides, odi_dtype_t dtype);

    // Copy constructor (creates a deep copy for owned tensors, shallow for views)
    Tensor(const Tensor& other);

    // Move constructor
    Tensor(Tensor&& other) noexcept;

    // Copy assignment
    Tensor& operator=(const Tensor& other);

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor();

    // Check if tensor is valid (has data)
    bool is_valid() const { return data_ != nullptr; }
    operator bool() const { return is_valid(); }

    // Check if tensor owns its memory
    bool owns_memory() const { return owned_; }

    // Check if tensor is a view
    bool is_view() const { return !owned_; }

    // Check if tensor is contiguous in memory
    bool is_contiguous() const;

    // Get data pointer
    void* data() { return data_; }
    const void* data() const { return data_; }

    // Get typed data pointer
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }

    // Get data type
    odi_dtype_t dtype() const { return dtype_; }

    // Get number of dimensions
    int ndim() const { return ndim_; }

    // Get shape
    const int64_t* shape() const { return shape_.data(); }
    int64_t shape(int dim) const;
    std::vector<int64_t> shape_vec() const;

    // Get strides (in bytes)
    const int64_t* strides() const { return strides_.data(); }
    int64_t stride(int dim) const;
    std::vector<int64_t> strides_vec() const;

    // Get total number of elements
    int64_t numel() const { return numel_; }

    // Get total size in bytes
    size_t nbytes() const;

    // Get size of a single element in bytes
    size_t element_size() const { return dtype_size(dtype_); }

    // Get name (optional, for debugging)
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // Element access (for debugging, not optimized)
    float get_f32(const std::vector<int64_t>& indices) const;
    void set_f32(const std::vector<int64_t>& indices, float value);

    // Fill tensor with a value
    void fill(float value);

    // Fill with zeros
    void zero();

    // Clone tensor (deep copy)
    Tensor clone() const;

    // Make contiguous copy if not already contiguous
    Tensor contiguous() const;

    // Reshape (returns view if possible, copy otherwise)
    Tensor reshape(const std::vector<int64_t>& new_shape) const;

    // View with different shape (must have same total elements)
    Tensor view(const std::vector<int64_t>& new_shape) const;

    // Transpose dimensions
    Tensor transpose(int dim0, int dim1) const;

    // Permute dimensions
    Tensor permute(const std::vector<int>& dims) const;

    // Slice along a dimension
    Tensor slice(int dim, int64_t start, int64_t end) const;

    // Select a single index along a dimension (reduces ndim by 1)
    Tensor select(int dim, int64_t index) const;

    // Squeeze (remove dimensions of size 1)
    Tensor squeeze(int dim = -1) const;

    // Unsqueeze (add dimension of size 1)
    Tensor unsqueeze(int dim) const;

    // Convert to different dtype (allocates new memory)
    Tensor to(odi_dtype_t new_dtype) const;

    // Print tensor info
    std::string info() const;

    // Print tensor values (for debugging small tensors)
    std::string to_string(int max_elements = 100) const;

private:
    void* data_ = nullptr;
    odi_dtype_t dtype_ = ODI_DTYPE_F32;
    int ndim_ = 0;
    int64_t numel_ = 0;
    bool owned_ = false;

    std::array<int64_t, MAX_DIMS> shape_ = {};
    std::array<int64_t, MAX_DIMS> strides_ = {};

    std::string name_;

    // Compute strides from shape for contiguous layout
    void compute_strides();

    // Compute byte offset for indices
    size_t offset(const std::vector<int64_t>& indices) const;
};

// Factory functions
Tensor zeros(const std::vector<int64_t>& shape, odi_dtype_t dtype = ODI_DTYPE_F32);
Tensor ones(const std::vector<int64_t>& shape, odi_dtype_t dtype = ODI_DTYPE_F32);
Tensor empty(const std::vector<int64_t>& shape, odi_dtype_t dtype = ODI_DTYPE_F32);
Tensor from_data(const float* data, const std::vector<int64_t>& shape);

} // namespace odi

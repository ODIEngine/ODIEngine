#include "tensor.hpp"
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace odi {

// Aligned memory allocation
void* aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
    if (ptr == nullptr) return;

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Get size of data type
size_t dtype_size(odi_dtype_t dtype) {
    switch (dtype) {
        case ODI_DTYPE_F32:  return 4;
        case ODI_DTYPE_F16:  return 2;
        case ODI_DTYPE_BF16: return 2;
        case ODI_DTYPE_I32:  return 4;
        case ODI_DTYPE_I16:  return 2;
        case ODI_DTYPE_I8:   return 1;
        // Quantized types return size per block
        case ODI_DTYPE_Q8_0: return 34;  // 32 bytes + 2 bytes scale (fp16)
        case ODI_DTYPE_Q4_0: return 18;  // 16 bytes (32 x 4bit) + 2 bytes scale
        case ODI_DTYPE_Q4_1: return 20;  // 16 bytes + 2 scale + 2 min
        case ODI_DTYPE_Q5_0: return 22;  // 16 bytes + 4 high bits + 2 scale
        case ODI_DTYPE_Q5_1: return 24;  // + min
        case ODI_DTYPE_Q4_K: return 144; // K-quant block
        case ODI_DTYPE_Q5_K: return 176;
        case ODI_DTYPE_Q6_K: return 210;
        case ODI_DTYPE_Q8_K: return 292;
        default: return 0;
    }
}

// Get size of quantized block
size_t dtype_block_size(odi_dtype_t dtype) {
    return dtype_size(dtype);
}

// Get number of elements per quantized block
int dtype_block_elements(odi_dtype_t dtype) {
    switch (dtype) {
        case ODI_DTYPE_Q8_0:
        case ODI_DTYPE_Q4_0:
        case ODI_DTYPE_Q4_1:
        case ODI_DTYPE_Q5_0:
        case ODI_DTYPE_Q5_1:
            return 32;
        case ODI_DTYPE_Q4_K:
        case ODI_DTYPE_Q5_K:
        case ODI_DTYPE_Q6_K:
        case ODI_DTYPE_Q8_K:
            return 256;
        default:
            return 1;
    }
}

// Check if dtype is quantized
bool dtype_is_quantized(odi_dtype_t dtype) {
    return dtype >= ODI_DTYPE_Q8_0;
}

// ============================================================================
// Tensor implementation
// ============================================================================

Tensor::Tensor() = default;

Tensor::Tensor(const std::vector<int64_t>& shape, odi_dtype_t dtype)
    : dtype_(dtype), ndim_(static_cast<int>(shape.size())), owned_(true) {

    if (ndim_ > MAX_DIMS) {
        throw std::runtime_error("Too many dimensions (max: " + std::to_string(MAX_DIMS) + ")");
    }

    numel_ = 1;
    for (int i = 0; i < ndim_; ++i) {
        shape_[i] = shape[i];
        numel_ *= shape[i];
    }

    compute_strides();

    size_t bytes = nbytes();
    if (bytes > 0) {
        data_ = aligned_alloc(bytes, 64);
        if (data_ == nullptr) {
            throw std::runtime_error("Failed to allocate tensor memory");
        }
    }
}

Tensor::Tensor(void* data, const std::vector<int64_t>& shape, odi_dtype_t dtype)
    : data_(data), dtype_(dtype), ndim_(static_cast<int>(shape.size())), owned_(false) {

    if (ndim_ > MAX_DIMS) {
        throw std::runtime_error("Too many dimensions");
    }

    numel_ = 1;
    for (int i = 0; i < ndim_; ++i) {
        shape_[i] = shape[i];
        numel_ *= shape[i];
    }

    compute_strides();
}

Tensor::Tensor(void* data, const std::vector<int64_t>& shape,
               const std::vector<int64_t>& strides, odi_dtype_t dtype)
    : data_(data), dtype_(dtype), ndim_(static_cast<int>(shape.size())), owned_(false) {

    if (ndim_ > MAX_DIMS) {
        throw std::runtime_error("Too many dimensions");
    }

    numel_ = 1;
    for (int i = 0; i < ndim_; ++i) {
        shape_[i] = shape[i];
        strides_[i] = strides[i];
        numel_ *= shape[i];
    }
}

Tensor::Tensor(const Tensor& other)
    : dtype_(other.dtype_), ndim_(other.ndim_), numel_(other.numel_),
      shape_(other.shape_), strides_(other.strides_), name_(other.name_) {

    if (other.owned_ && other.data_) {
        // Deep copy for owned tensors
        size_t bytes = other.nbytes();
        data_ = aligned_alloc(bytes, 64);
        if (data_ == nullptr) {
            throw std::runtime_error("Failed to allocate tensor memory");
        }
        std::memcpy(data_, other.data_, bytes);
        owned_ = true;
    } else {
        // Shallow copy for views
        data_ = other.data_;
        owned_ = false;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), dtype_(other.dtype_), ndim_(other.ndim_),
      numel_(other.numel_), owned_(other.owned_), shape_(other.shape_),
      strides_(other.strides_), name_(std::move(other.name_)) {
    other.data_ = nullptr;
    other.owned_ = false;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        if (owned_ && data_) {
            aligned_free(data_);
        }

        dtype_ = other.dtype_;
        ndim_ = other.ndim_;
        numel_ = other.numel_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        name_ = other.name_;

        if (other.owned_ && other.data_) {
            size_t bytes = other.nbytes();
            data_ = aligned_alloc(bytes, 64);
            if (data_ == nullptr) {
                throw std::runtime_error("Failed to allocate tensor memory");
            }
            std::memcpy(data_, other.data_, bytes);
            owned_ = true;
        } else {
            data_ = other.data_;
            owned_ = false;
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owned_ && data_) {
            aligned_free(data_);
        }

        data_ = other.data_;
        dtype_ = other.dtype_;
        ndim_ = other.ndim_;
        numel_ = other.numel_;
        owned_ = other.owned_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        name_ = std::move(other.name_);

        other.data_ = nullptr;
        other.owned_ = false;
    }
    return *this;
}

Tensor::~Tensor() {
    if (owned_ && data_) {
        aligned_free(data_);
    }
}

bool Tensor::is_contiguous() const {
    if (!data_ || ndim_ == 0) return true;

    int64_t expected_stride = dtype_size(dtype_);
    if (dtype_is_quantized(dtype_)) {
        // For quantized types, check block-level contiguity
        expected_stride = dtype_block_size(dtype_);
    }

    for (int i = ndim_ - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

int64_t Tensor::shape(int dim) const {
    if (dim < 0) dim += ndim_;
    if (dim < 0 || dim >= ndim_) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

std::vector<int64_t> Tensor::shape_vec() const {
    return std::vector<int64_t>(shape_.begin(), shape_.begin() + ndim_);
}

int64_t Tensor::stride(int dim) const {
    if (dim < 0) dim += ndim_;
    if (dim < 0 || dim >= ndim_) {
        throw std::out_of_range("Dimension out of range");
    }
    return strides_[dim];
}

std::vector<int64_t> Tensor::strides_vec() const {
    return std::vector<int64_t>(strides_.begin(), strides_.begin() + ndim_);
}

size_t Tensor::nbytes() const {
    if (dtype_is_quantized(dtype_)) {
        // For quantized types, calculate based on blocks
        int64_t num_blocks = (numel_ + dtype_block_elements(dtype_) - 1) / dtype_block_elements(dtype_);
        return num_blocks * dtype_block_size(dtype_);
    }
    return numel_ * dtype_size(dtype_);
}

void Tensor::compute_strides() {
    if (ndim_ == 0) return;

    size_t elem_size = dtype_size(dtype_);
    if (dtype_is_quantized(dtype_)) {
        elem_size = dtype_block_size(dtype_);
    }

    strides_[ndim_ - 1] = elem_size;
    for (int i = ndim_ - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

size_t Tensor::offset(const std::vector<int64_t>& indices) const {
    if (static_cast<int>(indices.size()) != ndim_) {
        throw std::runtime_error("Index dimension mismatch");
    }

    size_t off = 0;
    for (int i = 0; i < ndim_; ++i) {
        off += indices[i] * strides_[i];
    }
    return off;
}

float Tensor::get_f32(const std::vector<int64_t>& indices) const {
    if (!data_) throw std::runtime_error("Tensor has no data");

    size_t off = offset(indices);
    const uint8_t* ptr = static_cast<const uint8_t*>(data_) + off;

    switch (dtype_) {
        case ODI_DTYPE_F32:
            return *reinterpret_cast<const float*>(ptr);
        case ODI_DTYPE_F16: {
            // Simple FP16 -> FP32 conversion
            uint16_t h = *reinterpret_cast<const uint16_t*>(ptr);
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF);

            if (exp == 0) {
                if (mant == 0) {
                    uint32_t f = sign;
                    return *reinterpret_cast<float*>(&f);
                }
                // Denormal
                exp = 1;
                while ((mant & 0x400) == 0) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x3FF;
            } else if (exp == 31) {
                exp = 255;
            } else {
                exp = exp + 127 - 15;
            }

            uint32_t f = sign | (exp << 23) | (mant << 13);
            return *reinterpret_cast<float*>(&f);
        }
        case ODI_DTYPE_I32:
            return static_cast<float>(*reinterpret_cast<const int32_t*>(ptr));
        case ODI_DTYPE_I16:
            return static_cast<float>(*reinterpret_cast<const int16_t*>(ptr));
        case ODI_DTYPE_I8:
            return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
        default:
            throw std::runtime_error("get_f32 not implemented for this dtype");
    }
}

void Tensor::set_f32(const std::vector<int64_t>& indices, float value) {
    if (!data_) throw std::runtime_error("Tensor has no data");

    size_t off = offset(indices);
    uint8_t* ptr = static_cast<uint8_t*>(data_) + off;

    switch (dtype_) {
        case ODI_DTYPE_F32:
            *reinterpret_cast<float*>(ptr) = value;
            break;
        case ODI_DTYPE_I32:
            *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(value);
            break;
        case ODI_DTYPE_I16:
            *reinterpret_cast<int16_t*>(ptr) = static_cast<int16_t>(value);
            break;
        case ODI_DTYPE_I8:
            *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(value);
            break;
        default:
            throw std::runtime_error("set_f32 not implemented for this dtype");
    }
}

void Tensor::fill(float value) {
    if (!data_) return;

    if (dtype_ == ODI_DTYPE_F32) {
        float* ptr = static_cast<float*>(data_);
        for (int64_t i = 0; i < numel_; ++i) {
            ptr[i] = value;
        }
    } else {
        // Generic fill through set_f32
        std::vector<int64_t> indices(ndim_, 0);
        for (int64_t i = 0; i < numel_; ++i) {
            set_f32(indices, value);
            // Increment indices
            for (int d = ndim_ - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
}

void Tensor::zero() {
    if (data_ && is_contiguous()) {
        std::memset(data_, 0, nbytes());
    } else {
        fill(0.0f);
    }
}

Tensor Tensor::clone() const {
    Tensor result(shape_vec(), dtype_);
    result.name_ = name_;

    if (is_contiguous()) {
        std::memcpy(result.data_, data_, nbytes());
    } else {
        // Copy element by element for non-contiguous tensors
        std::vector<int64_t> indices(ndim_, 0);
        for (int64_t i = 0; i < numel_; ++i) {
            result.set_f32(indices, get_f32(indices));
            for (int d = ndim_ - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    return result;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    return clone();
}

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // Calculate total elements in new shape
    int64_t new_numel = 1;
    int neg_idx = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx != -1) {
                throw std::runtime_error("Only one dimension can be -1");
            }
            neg_idx = static_cast<int>(i);
        } else {
            new_numel *= new_shape[i];
        }
    }

    std::vector<int64_t> actual_shape = new_shape;
    if (neg_idx != -1) {
        actual_shape[neg_idx] = numel_ / new_numel;
        new_numel *= actual_shape[neg_idx];
    }

    if (new_numel != numel_) {
        throw std::runtime_error("Cannot reshape: element count mismatch");
    }

    if (is_contiguous()) {
        return view(actual_shape);
    } else {
        return clone().view(actual_shape);
    }
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("Cannot create view of non-contiguous tensor");
    }

    int64_t new_numel = 1;
    for (auto s : new_shape) {
        new_numel *= s;
    }

    if (new_numel != numel_) {
        throw std::runtime_error("View shape must have same number of elements");
    }

    Tensor result(data_, new_shape, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += ndim_;
    if (dim1 < 0) dim1 += ndim_;

    if (dim0 < 0 || dim0 >= ndim_ || dim1 < 0 || dim1 >= ndim_) {
        throw std::out_of_range("Transpose dimension out of range");
    }

    std::vector<int> perm(ndim_);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim0], perm[dim1]);

    return permute(perm);
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (static_cast<int>(dims.size()) != ndim_) {
        throw std::runtime_error("Permute dimensions must match tensor dimensions");
    }

    std::vector<int64_t> new_shape(ndim_);
    std::vector<int64_t> new_strides(ndim_);

    for (int i = 0; i < ndim_; ++i) {
        int d = dims[i];
        if (d < 0) d += ndim_;
        new_shape[i] = shape_[d];
        new_strides[i] = strides_[d];
    }

    Tensor result(data_, new_shape, new_strides, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    if (dim < 0) dim += ndim_;
    if (dim < 0 || dim >= ndim_) {
        throw std::out_of_range("Slice dimension out of range");
    }

    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];

    if (start < 0 || start >= shape_[dim] || end <= start || end > shape_[dim]) {
        throw std::out_of_range("Slice indices out of range");
    }

    std::vector<int64_t> new_shape = shape_vec();
    new_shape[dim] = end - start;

    std::vector<int64_t> new_strides = strides_vec();

    uint8_t* new_data = static_cast<uint8_t*>(data_) + start * strides_[dim];

    Tensor result(new_data, new_shape, new_strides, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::select(int dim, int64_t index) const {
    if (dim < 0) dim += ndim_;
    if (dim < 0 || dim >= ndim_) {
        throw std::out_of_range("Select dimension out of range");
    }

    if (index < 0) index += shape_[dim];
    if (index < 0 || index >= shape_[dim]) {
        throw std::out_of_range("Select index out of range");
    }

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    for (int i = 0; i < ndim_; ++i) {
        if (i != dim) {
            new_shape.push_back(shape_[i]);
            new_strides.push_back(strides_[i]);
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(dtype_size(dtype_));
    }

    uint8_t* new_data = static_cast<uint8_t*>(data_) + index * strides_[dim];

    Tensor result(new_data, new_shape, new_strides, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::squeeze(int dim) const {
    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    for (int i = 0; i < ndim_; ++i) {
        if (dim == -1) {
            if (shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        } else {
            int d = dim < 0 ? dim + ndim_ : dim;
            if (i != d || shape_[i] != 1) {
                new_shape.push_back(shape_[i]);
                new_strides.push_back(strides_[i]);
            }
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(dtype_size(dtype_));
    }

    Tensor result(data_, new_shape, new_strides, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim_ + 1;
    if (dim < 0 || dim > ndim_) {
        throw std::out_of_range("Unsqueeze dimension out of range");
    }

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    for (int i = 0; i < ndim_; ++i) {
        if (i == dim) {
            new_shape.push_back(1);
            new_strides.push_back(strides_[i] * shape_[i]);
        }
        new_shape.push_back(shape_[i]);
        new_strides.push_back(strides_[i]);
    }

    if (dim == ndim_) {
        new_shape.push_back(1);
        new_strides.push_back(dtype_size(dtype_));
    }

    Tensor result(data_, new_shape, new_strides, dtype_);
    result.name_ = name_;
    return result;
}

Tensor Tensor::to(odi_dtype_t new_dtype) const {
    if (new_dtype == dtype_) {
        return clone();
    }

    if (dtype_is_quantized(dtype_) || dtype_is_quantized(new_dtype)) {
        throw std::runtime_error("Quantized dtype conversion not yet implemented");
    }

    Tensor result(shape_vec(), new_dtype);
    result.name_ = name_;

    // Convert through float
    std::vector<int64_t> indices(ndim_, 0);
    for (int64_t i = 0; i < numel_; ++i) {
        result.set_f32(indices, get_f32(indices));
        for (int d = ndim_ - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }

    return result;
}

std::string Tensor::info() const {
    std::ostringstream ss;
    ss << "Tensor(";

    if (!name_.empty()) {
        ss << "name=" << name_ << ", ";
    }

    ss << "shape=[";
    for (int i = 0; i < ndim_; ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << "], dtype=" << odi_dtype_name(dtype_);

    if (!is_contiguous()) {
        ss << ", non-contiguous";
    }

    ss << ", " << (owned_ ? "owned" : "view");
    ss << ")";

    return ss.str();
}

std::string Tensor::to_string(int max_elements) const {
    std::ostringstream ss;
    ss << info() << "\n";

    if (numel_ == 0) {
        ss << "[]";
        return ss.str();
    }

    ss << std::fixed << std::setprecision(4);

    std::vector<int64_t> indices(ndim_, 0);
    int count = 0;
    bool truncated = false;

    ss << "[";
    for (int64_t i = 0; i < numel_ && count < max_elements; ++i, ++count) {
        if (i > 0) ss << ", ";
        ss << get_f32(indices);

        // Increment indices
        for (int d = ndim_ - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }

    if (count < numel_) {
        ss << ", ... (" << (numel_ - count) << " more)";
        truncated = true;
    }
    ss << "]";

    return ss.str();
}

// Factory functions
Tensor zeros(const std::vector<int64_t>& shape, odi_dtype_t dtype) {
    Tensor t(shape, dtype);
    t.zero();
    return t;
}

Tensor ones(const std::vector<int64_t>& shape, odi_dtype_t dtype) {
    Tensor t(shape, dtype);
    t.fill(1.0f);
    return t;
}

Tensor empty(const std::vector<int64_t>& shape, odi_dtype_t dtype) {
    return Tensor(shape, dtype);
}

Tensor from_data(const float* data, const std::vector<int64_t>& shape) {
    Tensor t(shape, ODI_DTYPE_F32);
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    std::memcpy(t.data(), data, numel * sizeof(float));
    return t;
}

} // namespace odi

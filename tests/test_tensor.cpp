/**
 * ODI Engine - Tensor Unit Tests
 */

#include "tensor/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace odi;

void test_tensor_creation() {
    std::cout << "Testing tensor creation... ";

    // Test empty tensor
    Tensor empty;
    assert(!empty.is_valid());

    // Test 1D tensor
    Tensor t1({10}, ODI_DTYPE_F32);
    assert(t1.is_valid());
    assert(t1.ndim() == 1);
    assert(t1.shape(0) == 10);
    assert(t1.numel() == 10);
    assert(t1.owns_memory());
    assert(t1.is_contiguous());

    // Test 2D tensor
    Tensor t2({3, 4}, ODI_DTYPE_F32);
    assert(t2.ndim() == 2);
    assert(t2.shape(0) == 3);
    assert(t2.shape(1) == 4);
    assert(t2.numel() == 12);

    // Test 3D tensor
    Tensor t3({2, 3, 4}, ODI_DTYPE_F32);
    assert(t3.ndim() == 3);
    assert(t3.numel() == 24);

    std::cout << "PASSED" << std::endl;
}

void test_tensor_fill_and_access() {
    std::cout << "Testing tensor fill and access... ";

    Tensor t({3, 4}, ODI_DTYPE_F32);

    // Fill with value
    t.fill(3.14f);

    // Check all values
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float val = t.get_f32({i, j});
            assert(std::abs(val - 3.14f) < 1e-5f);
        }
    }

    // Set individual values
    t.set_f32({0, 0}, 1.0f);
    t.set_f32({2, 3}, 2.0f);
    assert(std::abs(t.get_f32({0, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(t.get_f32({2, 3}) - 2.0f) < 1e-5f);

    // Test zero
    t.zero();
    assert(std::abs(t.get_f32({1, 1})) < 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_tensor_copy_and_clone() {
    std::cout << "Testing tensor copy and clone... ";

    Tensor t1({2, 3}, ODI_DTYPE_F32);
    t1.fill(5.0f);

    // Clone
    Tensor t2 = t1.clone();
    assert(t2.is_valid());
    assert(t2.owns_memory());
    assert(t2.numel() == t1.numel());
    assert(std::abs(t2.get_f32({1, 2}) - 5.0f) < 1e-5f);

    // Modify original, clone should be unchanged
    t1.set_f32({0, 0}, 100.0f);
    assert(std::abs(t2.get_f32({0, 0}) - 5.0f) < 1e-5f);

    // Move
    Tensor t3 = std::move(t2);
    assert(t3.is_valid());
    assert(!t2.is_valid());

    std::cout << "PASSED" << std::endl;
}

void test_tensor_view_operations() {
    std::cout << "Testing tensor view operations... ";

    Tensor t({4, 3, 2}, ODI_DTYPE_F32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 2; ++k) {
                t.set_f32({i, j, k}, static_cast<float>(i * 6 + j * 2 + k));
            }
        }
    }

    // Reshape
    Tensor reshaped = t.reshape({12, 2});
    assert(reshaped.ndim() == 2);
    assert(reshaped.shape(0) == 12);
    assert(reshaped.shape(1) == 2);
    assert(std::abs(reshaped.get_f32({0, 0}) - t.get_f32({0, 0, 0})) < 1e-5f);

    // Slice
    Tensor sliced = t.slice(0, 1, 3);
    assert(sliced.shape(0) == 2);
    assert(sliced.shape(1) == 3);
    assert(sliced.shape(2) == 2);

    // Select
    Tensor selected = t.select(0, 2);
    assert(selected.ndim() == 2);
    assert(selected.shape(0) == 3);
    assert(selected.shape(1) == 2);

    // Transpose
    Tensor transposed = t.transpose(0, 2);
    assert(transposed.shape(0) == 2);
    assert(transposed.shape(1) == 3);
    assert(transposed.shape(2) == 4);

    std::cout << "PASSED" << std::endl;
}

void test_tensor_factories() {
    std::cout << "Testing tensor factory functions... ";

    // Zeros
    Tensor z = zeros({2, 3});
    assert(std::abs(z.get_f32({0, 0})) < 1e-5f);
    assert(std::abs(z.get_f32({1, 2})) < 1e-5f);

    // Ones
    Tensor o = ones({3, 2});
    assert(std::abs(o.get_f32({0, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(o.get_f32({2, 1}) - 1.0f) < 1e-5f);

    // From data
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor fd = from_data(data, {2, 3});
    assert(std::abs(fd.get_f32({0, 0}) - 1.0f) < 1e-5f);
    assert(std::abs(fd.get_f32({1, 2}) - 6.0f) < 1e-5f);

    std::cout << "PASSED" << std::endl;
}

void test_dtype_utilities() {
    std::cout << "Testing dtype utilities... ";

    assert(dtype_size(ODI_DTYPE_F32) == 4);
    assert(dtype_size(ODI_DTYPE_F16) == 2);
    assert(dtype_size(ODI_DTYPE_I8) == 1);

    assert(!dtype_is_quantized(ODI_DTYPE_F32));
    assert(!dtype_is_quantized(ODI_DTYPE_F16));
    assert(dtype_is_quantized(ODI_DTYPE_Q4_0));
    assert(dtype_is_quantized(ODI_DTYPE_Q8_0));

    assert(dtype_block_elements(ODI_DTYPE_F32) == 1);
    assert(dtype_block_elements(ODI_DTYPE_Q4_0) == 32);
    assert(dtype_block_elements(ODI_DTYPE_Q8_0) == 32);

    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ODI Engine - Tensor Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_tensor_creation();
        test_tensor_fill_and_access();
        test_tensor_copy_and_clone();
        test_tensor_view_operations();
        test_tensor_factories();
        test_dtype_utilities();

        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}

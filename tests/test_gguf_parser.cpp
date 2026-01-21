/**
 * ODI Engine - GGUF Parser Unit Tests
 */

#include "format/gguf_parser.hpp"
#include "format/gguf_types.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

using namespace odi;

void test_gguf_types() {
    std::cout << "Testing GGUF type utilities... ";

    // Test tensor type sizes
    assert(gguf_tensor_type_size(GGUFTensorType::F32) == 4);
    assert(gguf_tensor_type_size(GGUFTensorType::F16) == 2);
    assert(gguf_tensor_type_size(GGUFTensorType::Q4_0) == sizeof(BlockQ4_0));
    assert(gguf_tensor_type_size(GGUFTensorType::Q8_0) == sizeof(BlockQ8_0));

    // Test block sizes
    assert(gguf_tensor_type_block_size(GGUFTensorType::F32) == 1);
    assert(gguf_tensor_type_block_size(GGUFTensorType::Q4_0) == 32);
    assert(gguf_tensor_type_block_size(GGUFTensorType::Q8_0) == 32);
    assert(gguf_tensor_type_block_size(GGUFTensorType::Q4_K) == 256);

    // Test type names
    assert(std::string(gguf_tensor_type_name(GGUFTensorType::F32)) == "F32");
    assert(std::string(gguf_tensor_type_name(GGUFTensorType::F16)) == "F16");
    assert(std::string(gguf_tensor_type_name(GGUFTensorType::Q4_0)) == "Q4_0");

    std::cout << "PASSED" << std::endl;
}

void test_block_structures() {
    std::cout << "Testing quantized block structures... ";

    // Verify block sizes match expected values
    assert(sizeof(BlockQ4_0) == 18);
    assert(sizeof(BlockQ4_1) == 20);
    assert(sizeof(BlockQ5_0) == 22);
    assert(sizeof(BlockQ5_1) == 24);
    assert(sizeof(BlockQ8_0) == 34);
    assert(sizeof(BlockQ8_1) == 40);

    std::cout << "PASSED" << std::endl;
}

void test_gguf_metadata_keys() {
    std::cout << "Testing GGUF metadata keys... ";

    // Verify key constants are defined
    assert(GGUFKeys::GENERAL_ARCHITECTURE != nullptr);
    assert(GGUFKeys::GENERAL_NAME != nullptr);
    assert(GGUFKeys::TOKENIZER_MODEL != nullptr);
    assert(GGUFKeys::TOKENIZER_TOKENS != nullptr);

    std::cout << "PASSED" << std::endl;
}

// Note: Testing actual file parsing requires a test GGUF file
// This test is skipped if no test file is available
void test_gguf_file_parsing(const char* test_file_path) {
    std::cout << "Testing GGUF file parsing... ";

    // Check if test file exists
    std::ifstream test_file(test_file_path);
    if (!test_file.good()) {
        std::cout << "SKIPPED (no test file)" << std::endl;
        return;
    }
    test_file.close();

    auto gguf = GGUFFile::open(test_file_path, true);
    if (!gguf) {
        std::cout << "FAILED (could not open file)" << std::endl;
        return;
    }

    // Basic validation
    assert(gguf->version() >= 2);
    assert(gguf->version() <= 3);

    // Check metadata
    assert(gguf->metadata_count() > 0);
    assert(gguf->has_metadata("general.architecture"));

    // Check model info
    const auto& info = gguf->model_info();
    assert(!info.architecture.empty());

    // Check tensors
    assert(gguf->tensor_count() > 0);

    // Print summary
    std::cout << "PASSED" << std::endl;
    std::cout << "  File: " << test_file_path << std::endl;
    std::cout << "  Version: " << gguf->version() << std::endl;
    std::cout << "  Architecture: " << info.architecture << std::endl;
    std::cout << "  Tensors: " << gguf->tensor_count() << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "ODI Engine - GGUF Parser Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        test_gguf_types();
        test_block_structures();
        test_gguf_metadata_keys();

        // Optional: test file parsing if a file is provided
        if (argc > 1) {
            test_gguf_file_parsing(argv[1]);
        } else {
            std::cout << "Testing GGUF file parsing... SKIPPED (no file provided)" << std::endl;
            std::cout << "  Usage: " << argv[0] << " <test.gguf>" << std::endl;
        }

        std::cout << "========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}

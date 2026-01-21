cmake_minimum_required(VERSION 3.16)
project(ODIEngine VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(ODI_BUILD_TESTS "Build unit tests" ON)
option(ODI_BUILD_EXAMPLES "Build examples" ON)
option(ODI_ENABLE_METAL "Enable Metal backend (macOS/iOS)" OFF)
option(ODI_ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(ODI_ENABLE_CUDA "Enable CUDA backend" OFF)

# Auto-detect Metal on Apple platforms
if(APPLE)
    set(ODI_ENABLE_METAL ON)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)

    # Release optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

    # SIMD detection and flags
    include(CheckCXXCompilerFlag)

    # x86_64 SIMD
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)

        if(COMPILER_SUPPORTS_AVX2)
            add_compile_definitions(ODI_HAS_AVX2)
        endif()
        if(COMPILER_SUPPORTS_AVX512)
            add_compile_definitions(ODI_HAS_AVX512)
        endif()
        if(COMPILER_SUPPORTS_FMA)
            add_compile_definitions(ODI_HAS_FMA)
        endif()
        if(COMPILER_SUPPORTS_F16C)
            add_compile_definitions(ODI_HAS_F16C)
        endif()
    endif()

    # ARM NEON (always available on ARM64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        add_compile_definitions(ODI_HAS_NEON)
        # ARM64 always has NEON
    endif()

elseif(MSVC)
    add_compile_options(/W4)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/O2 /DNDEBUG")

    # MSVC SIMD
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        add_compile_definitions(ODI_HAS_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Collect source files
set(ODI_SOURCES
    src/odi_engine.cpp
    src/tensor/tensor.cpp
    src/format/gguf_parser.cpp
    src/compute/cpu/cpu_backend.cpp
    src/compute/cpu/simd_ops.cpp
    src/core/engine.cpp
    src/core/model.cpp
    src/core/context.cpp
    src/core/tokenizer.cpp
    src/core/sampler.cpp
    src/layers/embedding.cpp
    src/layers/attention.cpp
    src/layers/ffn.cpp
    src/layers/norm.cpp
    src/layers/rope.cpp
)

set(ODI_HEADERS
    include/odi/odi_engine.h
    include/odi/odi_types.h
    include/odi/odi_error.h
)

# Metal backend
if(ODI_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(METALKIT_FRAMEWORK MetalKit)

    if(METAL_FRAMEWORK)
        add_compile_definitions(ODI_HAS_METAL)
        list(APPEND ODI_SOURCES src/compute/metal/metal_backend.mm)
        list(APPEND ODI_LINK_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()

# Vulkan backend
if(ODI_ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        add_compile_definitions(ODI_HAS_VULKAN)
        list(APPEND ODI_SOURCES src/compute/vulkan/vulkan_backend.cpp)
        list(APPEND ODI_LINK_LIBS Vulkan::Vulkan)
    endif()
endif()

# CUDA backend
if(ODI_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_compile_definitions(ODI_HAS_CUDA)
        list(APPEND ODI_SOURCES src/compute/cuda/cuda_backend.cu)
    endif()
endif()

# Threading
find_package(Threads REQUIRED)
list(APPEND ODI_LINK_LIBS Threads::Threads)

# Build static library
add_library(odi_static STATIC ${ODI_SOURCES})
target_link_libraries(odi_static ${ODI_LINK_LIBS})
set_target_properties(odi_static PROPERTIES OUTPUT_NAME odi)

# Build shared library
add_library(odi_shared SHARED ${ODI_SOURCES})
target_link_libraries(odi_shared ${ODI_LINK_LIBS})
set_target_properties(odi_shared PROPERTIES OUTPUT_NAME odi)

# Install targets
install(TARGETS odi_static odi_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/odi DESTINATION include)

# Tests
if(ODI_BUILD_TESTS)
    enable_testing()

    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor odi_static)
    add_test(NAME test_tensor COMMAND test_tensor)

    add_executable(test_gguf_parser tests/test_gguf_parser.cpp)
    target_link_libraries(test_gguf_parser odi_static)
    add_test(NAME test_gguf_parser COMMAND test_gguf_parser)
endif()

# Examples
if(ODI_BUILD_EXAMPLES)
    add_executable(simple_inference examples/simple_inference.c)
    target_link_libraries(simple_inference odi_static)

    add_executable(chat_example examples/chat_example.cpp)
    target_link_libraries(chat_example odi_static)
endif()

# Print configuration
message(STATUS "")
message(STATUS "ODI Engine Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Metal: ${ODI_ENABLE_METAL}")
message(STATUS "  Vulkan: ${ODI_ENABLE_VULKAN}")
message(STATUS "  CUDA: ${ODI_ENABLE_CUDA}")
message(STATUS "  Tests: ${ODI_BUILD_TESTS}")
message(STATUS "  Examples: ${ODI_BUILD_EXAMPLES}")
message(STATUS "")

#pragma once

#include "../compute/backend.hpp"
#include <odi/odi_types.h>
#include <odi/odi_error.h>

#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace odi {

// Forward declarations
class Model;
class Context;

/**
 * Engine - Main entry point for ODI inference
 *
 * Manages backend initialization, model loading, and inference contexts.
 */
class Engine {
public:
    explicit Engine(const odi_engine_config_t& config);
    ~Engine();

    // Prevent copying
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Configuration
    const odi_engine_config_t& config() const { return config_; }

    // Backend access
    Backend* backend() { return backend_.get(); }
    const Backend* backend() const { return backend_.get(); }
    odi_backend_type_t backend_type() const { return backend_ ? backend_->type() : ODI_BACKEND_CPU; }

    // Model management
    Model* load_model(const std::string& path, const odi_model_config_t& config);
    Model* load_model_with_progress(const std::string& path, const odi_model_config_t& config,
                                    odi_progress_callback_t callback, void* user_data);
    void unload_model(Model* model);

    // Get system info
    std::string system_info() const;

    // Thread safety
    int num_threads() const { return config_.num_threads; }

private:
    odi_engine_config_t config_;
    std::unique_ptr<Backend> backend_;

    // Loaded models (for cleanup on destruction)
    std::vector<std::unique_ptr<Model>> models_;
    std::mutex models_mutex_;
};

} // namespace odi

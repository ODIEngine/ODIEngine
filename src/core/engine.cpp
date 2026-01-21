#include "engine.hpp"
#include "model.hpp"
#include "../compute/cpu/cpu_backend.hpp"

#include <sstream>

namespace odi {

Engine::Engine(const odi_engine_config_t& config) : config_(config) {
    // Create backend
    int threads = config.num_threads;
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 4;
    }
    config_.num_threads = threads;

    backend_ = create_backend(config.backend, threads);
    if (!backend_) {
        throw std::runtime_error("Failed to create backend");
    }
}

Engine::~Engine() {
    // Models are automatically cleaned up through unique_ptr
}

Model* Engine::load_model(const std::string& path, const odi_model_config_t& config) {
    return load_model_with_progress(path, config, nullptr, nullptr);
}

Model* Engine::load_model_with_progress(const std::string& path, const odi_model_config_t& config,
                                        odi_progress_callback_t callback, void* user_data) {
    std::lock_guard<std::mutex> lock(models_mutex_);

    auto model = std::make_unique<Model>(this, path, config, callback, user_data);
    Model* ptr = model.get();
    models_.push_back(std::move(model));

    return ptr;
}

void Engine::unload_model(Model* model) {
    std::lock_guard<std::mutex> lock(models_mutex_);

    auto it = std::find_if(models_.begin(), models_.end(),
                          [model](const std::unique_ptr<Model>& m) { return m.get() == model; });

    if (it != models_.end()) {
        models_.erase(it);
    }
}

std::string Engine::system_info() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"version\": \"" << ODI_VERSION_MAJOR << "." << ODI_VERSION_MINOR << "." << ODI_VERSION_PATCH << "\",\n";
    ss << "  \"backend\": \"" << backend_->name() << "\",\n";
    ss << "  \"threads\": " << config_.num_threads << ",\n";
    ss << "  \"device_info\": \"" << backend_->device_info() << "\",\n";
    ss << "  \"available_memory\": " << backend_->available_memory() << "\n";
    ss << "}";
    return ss.str();
}

} // namespace odi

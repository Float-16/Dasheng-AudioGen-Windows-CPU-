#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <fstream>
#include <vector>

#include "gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "backend.h"

namespace dasheng {

class GgufModel {
public:
    explicit GgufModel(const std::string &path) {
        struct gguf_init_params params = {false, &ctx_};
        gguf_ = gguf_init_from_file(path.c_str(), params);
        if (!gguf_) {
            throw std::runtime_error("failed to load GGUF file: " + path);
        }
    }

    ~GgufModel() {
        if (ctx_) ggml_free(ctx_);
        if (gguf_) gguf_free(gguf_);
    }

    GgufModel(const GgufModel &) = delete;
    GgufModel &operator=(const GgufModel &) = delete;

    struct ggml_context *ctx() const { return ctx_; }

    struct ggml_tensor *tensor(const std::string &name) const {
        struct ggml_tensor *t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            throw std::runtime_error("GGUF file is missing tensor: " + name);
        }
        return t;
    }

    bool has_tensor(const std::string &name) const {
        return ggml_get_tensor(ctx_, name.c_str()) != nullptr;
    }

    int32_t kv_i32(const std::string &key) const {
        return gguf_get_val_i32(gguf_, find_key(key));
    }

    float kv_f32(const std::string &key) const {
        return gguf_get_val_f32(gguf_, find_key(key));
    }

    std::string kv_str(const std::string &key) const {
        return gguf_get_val_str(gguf_, find_key(key));
    }

    bool kv_bool(const std::string &key) const {
        return gguf_get_val_bool(gguf_, find_key(key));
    }

private:
    int64_t find_key(const std::string &key) const {
        int64_t id = gguf_find_key(gguf_, key.c_str());
        if (id < 0) {
            throw std::runtime_error("GGUF file is missing metadata key: " + key);
        }
        return id;
    }

    struct gguf_context *gguf_ = nullptr;
    struct ggml_context *ctx_ = nullptr;
};

class GgufModelGPU {
public:
    explicit GgufModelGPU(const std::string &path, ggml_backend_t backend)
        : path_(path), backend_(backend) {

        struct gguf_init_params params = {true, &ctx_};
        gguf_ = gguf_init_from_file(path.c_str(), params);
        if (!gguf_) {
            throw std::runtime_error("failed to load GGUF file: " + path);
        }

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("failed to open GGUF file: " + path);
        }

        file_size_ = file.tellg();
        file.seekg(0, std::ios::beg);

        file_data_.resize(file_size_);
        if (!file.read(reinterpret_cast<char*>(file_data_.data()), file_size_)) {
            throw std::runtime_error("failed to read GGUF file: " + path);
        }

        data_offset_ = gguf_get_data_offset(gguf_);

        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buffer_) {
            throw std::runtime_error("failed to allocate tensors on backend");
        }
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        size_t total_bytes = 0;
        int64_t n_tensors = gguf_get_n_tensors(gguf_);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char* name = gguf_get_tensor_name(gguf_, i);
            size_t offset = gguf_get_tensor_offset(gguf_, i);
            size_t size = gguf_get_tensor_size(gguf_, i);

            struct ggml_tensor* t = ggml_get_tensor(ctx_, name);
            if (t) {
                const void* src = file_data_.data() + data_offset_ + offset;
                ggml_backend_tensor_set(t, src, 0, size);
                total_bytes += size;
            }
        }

        std::fprintf(stderr, "[%s] loaded %.1f MB to %s\n",
                     path.c_str(), total_bytes / (1024.0 * 1024.0),
                     ggml_backend_name(backend));
    }

    ~GgufModelGPU() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
        if (gguf_) gguf_free(gguf_);
    }

    GgufModelGPU(const GgufModelGPU &) = delete;
    GgufModelGPU &operator=(const GgufModelGPU &) = delete;

    struct ggml_context *ctx() const { return ctx_; }

    struct ggml_tensor *tensor(const std::string &name) const {
        struct ggml_tensor *t = ggml_get_tensor(ctx_, name.c_str());
        if (!t) {
            throw std::runtime_error("GGUF file is missing tensor: " + name);
        }
        return t;
    }

    bool has_tensor(const std::string &name) const {
        return ggml_get_tensor(ctx_, name.c_str()) != nullptr;
    }

    int32_t kv_i32(const std::string &key) const {
        return gguf_get_val_i32(gguf_, find_key(key));
    }

    float kv_f32(const std::string &key) const {
        return gguf_get_val_f32(gguf_, find_key(key));
    }

    std::string kv_str(const std::string &key) const {
        return gguf_get_val_str(gguf_, find_key(key));
    }

    bool kv_bool(const std::string &key) const {
        return gguf_get_val_bool(gguf_, find_key(key));
    }

    ggml_backend_t backend() const { return backend_; }

private:
    int64_t find_key(const std::string &key) const {
        int64_t id = gguf_find_key(gguf_, key.c_str());
        if (id < 0) {
            throw std::runtime_error("GGUF file is missing metadata key: " + key);
        }
        return id;
    }

    std::string path_;
    ggml_backend_t backend_;
    struct gguf_context *gguf_ = nullptr;
    struct ggml_context *ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    std::vector<uint8_t> file_data_;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
};

}  // namespace dasheng
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class OnnxStatefulEngine {
public:
    static constexpr int64_t kDefaultFrameSize = 64;
    static constexpr int64_t kParamCount = 4;

    OnnxStatefulEngine();
    ~OnnxStatefulEngine();

    OnnxStatefulEngine(const OnnxStatefulEngine&) = delete;
    OnnxStatefulEngine& operator=(const OnnxStatefulEngine&) = delete;

    bool load(const std::wstring& modelPath, std::string& error);
    bool setFrameSize(int64_t frameSize, std::string& error);
    bool processFrame(
        const float* input,
        const std::array<float, kParamCount>& parameters,
        float* output) noexcept;
    void resetState() noexcept;
    bool isLoaded() const noexcept { return session_ != nullptr; }
    int64_t frameSize() const noexcept { return audioShape_[1]; }

private:
    bool createTensors(std::string& error);

    Ort::Env environment_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_;

    std::vector<float> inputAudio_;
    std::array<float, kParamCount> inputParams_{};
    std::vector<float> inputState_;
    std::vector<float> outputAudio_;
    std::vector<float> outputState_;

    std::array<int64_t, 3> audioShape_{1, kDefaultFrameSize, 1};
    std::array<int64_t, 2> parameterShape_{1, kParamCount};
    std::vector<int64_t> stateShape_;

    std::vector<Ort::Value> inputs_;
    std::vector<Ort::Value> outputs_;
    std::array<const char*, 3> inputNames_{"audio", "params", "state"};
    std::array<const char*, 2> outputNames_{"output", "new_state"};
};

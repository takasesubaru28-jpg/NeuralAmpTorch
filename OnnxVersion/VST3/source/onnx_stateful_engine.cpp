#include "onnx_stateful_engine.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

namespace {
size_t elementCount(const std::vector<int64_t>& shape)
{
    return std::accumulate(
        shape.begin(), shape.end(), size_t{1},
        std::multiplies<size_t>());
}
}

OnnxStatefulEngine::OnnxStatefulEngine()
    : environment_(ORT_LOGGING_LEVEL_WARNING, "NeuralAmpOnnx")
    , memoryInfo_(Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetIntraOpNumThreads(1);
    sessionOptions_.SetInterOpNumThreads(1);
    sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions_.DisableMemPattern();
}

OnnxStatefulEngine::~OnnxStatefulEngine() = default;

bool OnnxStatefulEngine::load(
    const std::wstring& modelPath, std::string& error)
{
    try {
        auto candidate = std::make_unique<Ort::Session>(
            environment_, modelPath.c_str(), sessionOptions_);
        if (candidate->GetInputCount() != 3 ||
            candidate->GetOutputCount() != 2) {
            error = "Expected 3 inputs and 2 outputs";
            return false;
        }
        session_ = std::move(candidate);
        return createTensors(error);
    } catch (const Ort::Exception& exception) {
        error = exception.what();
        session_.reset();
        return false;
    }
}

bool OnnxStatefulEngine::setFrameSize(int64_t frameSize, std::string& error)
{
    if (frameSize != 64 && frameSize != 128 &&
        frameSize != 256 && frameSize != 512) {
        error = "Frame size must be 64, 128, 256, or 512";
        return false;
    }
    if (audioShape_[1] == frameSize)
        return true;
    audioShape_[1] = frameSize;
    return !session_ || createTensors(error);
}

bool OnnxStatefulEngine::createTensors(std::string& error)
{
    try {
        stateShape_ = session_->GetInputTypeInfo(2)
                          .GetTensorTypeAndShapeInfo()
                          .GetShape();
        for (auto& dimension : stateShape_) {
            if (dimension < 1)
                dimension = 1;
        }

        inputAudio_.assign(static_cast<size_t>(audioShape_[1]), 0.0f);
        outputAudio_.assign(static_cast<size_t>(audioShape_[1]), 0.0f);
        inputState_.assign(elementCount(stateShape_), 0.0f);
        outputState_.assign(elementCount(stateShape_), 0.0f);

        inputs_.clear();
        inputs_.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, inputAudio_.data(), inputAudio_.size(),
            audioShape_.data(), audioShape_.size()));
        inputs_.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, inputParams_.data(), inputParams_.size(),
            parameterShape_.data(), parameterShape_.size()));
        inputs_.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, inputState_.data(), inputState_.size(),
            stateShape_.data(), stateShape_.size()));

        outputs_.clear();
        outputs_.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, outputAudio_.data(), outputAudio_.size(),
            audioShape_.data(), audioShape_.size()));
        outputs_.emplace_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, outputState_.data(), outputState_.size(),
            stateShape_.data(), stateShape_.size()));
        return true;
    } catch (const Ort::Exception& exception) {
        error = exception.what();
        session_.reset();
        return false;
    }
}

bool OnnxStatefulEngine::processFrame(
    const float* input,
    const std::array<float, kParamCount>& parameters,
    float* output) noexcept
{
    if (!session_)
        return false;
    const auto frameSize = static_cast<size_t>(audioShape_[1]);
    std::memcpy(inputAudio_.data(), input, sizeof(float) * frameSize);
    inputParams_ = parameters;
    try {
        session_->Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(), inputs_.data(), inputs_.size(),
            outputNames_.data(), outputs_.data(), outputs_.size());
        std::memcpy(output, outputAudio_.data(), sizeof(float) * frameSize);
        std::copy(outputState_.begin(), outputState_.end(), inputState_.begin());
        return true;
    } catch (...) {
        return false;
    }
}

void OnnxStatefulEngine::resetState() noexcept
{
    std::fill(inputState_.begin(), inputState_.end(), 0.0f);
    std::fill(outputState_.begin(), outputState_.end(), 0.0f);
}

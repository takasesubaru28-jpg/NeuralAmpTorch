#include "processor.h"

#include "fuid.h"
#include "parameter.h"
#include "base/source/fstreamer.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/vstspeaker.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

namespace {
constexpr int64_t kFrameSizes[] = {64, 128, 256, 512};
}

#if defined(_WIN32)
#include <windows.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;
#endif

namespace Steinberg::Vst {

NeuralAmpProcessor::NeuralAmpProcessor()
{
    setControllerClass(ControllerUID);
}

NeuralAmpProcessor::~NeuralAmpProcessor() = default;

tresult PLUGIN_API NeuralAmpProcessor::initialize(FUnknown* context)
{
    const auto result = AudioEffect::initialize(context);
    if (result != kResultOk)
        return result;
    addAudioInput(STR16("Mono In"), SpeakerArr::kMono);
    addAudioOutput(STR16("Stereo Out"), SpeakerArr::kStereo);

#if defined(_WIN32)
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(
        reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH);
    pluginDirectory_ = std::filesystem::path(path).parent_path().wstring();
#endif
    engine_ = std::make_unique<OnnxStatefulEngine>();
    applyFrameSize();
    loadSelectedModel();
    return kResultOk;
}

tresult PLUGIN_API NeuralAmpProcessor::setActive(TBool state)
{
    if (state) {
        if (selectedFrameSizeIndex_ != appliedFrameSizeIndex_)
            applyFrameSize();
        if (selectedModel_ != loadedModel_)
            loadSelectedModel();
        resetStreamingState();
    }
    return AudioEffect::setActive(state);
}

tresult PLUGIN_API NeuralAmpProcessor::setupProcessing(ProcessSetup& setup)
{
    smoothingCoefficient_ =
        1.0f - std::exp(-1.0f / static_cast<float>(setup.sampleRate * 0.01));
    resetStreamingState();
    return AudioEffect::setupProcessing(setup);
}

tresult PLUGIN_API NeuralAmpProcessor::setBusArrangements(
    SpeakerArrangement* inputs, int32 numIns,
    SpeakerArrangement* outputs, int32 numOuts)
{
    if (numIns != 1 || numOuts != 1 ||
        inputs[0] != SpeakerArr::kMono ||
        outputs[0] != SpeakerArr::kStereo)
        return kResultFalse;
    return AudioEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
}

tresult PLUGIN_API NeuralAmpProcessor::canProcessSampleSize(int32 size)
{
    return size == kSample32 ? kResultTrue : kResultFalse;
}

uint32 PLUGIN_API NeuralAmpProcessor::getLatencySamples()
{
    return static_cast<uint32>(inputFrame_.empty() ? 64 : inputFrame_.size());
}

tresult PLUGIN_API NeuralAmpProcessor::setState(IBStream* state)
{
    if (!state)
        return kResultFalse;
    IBStreamer streamer(state, kLittleEndian);
    for (auto& value : targetParams_) {
        if (!streamer.readFloat(value))
            return kResultFalse;
    }
    int32 model = 0;
    int32 bypass = 0;
    if (!streamer.readInt32(model) ||
        !streamer.readFloat(targetVolume_) ||
        !streamer.readInt32(bypass))
        return kResultFalse;
    selectedModel_ = std::clamp(static_cast<int>(model), 0, 3);
    int32 frameSize = 0;
    if (streamer.readInt32(frameSize))
        selectedFrameSizeIndex_ = std::clamp(static_cast<int>(frameSize), 0, 3);
    bypass_ = bypass != 0;
    smoothedParams_ = targetParams_;
    smoothedVolume_ = targetVolume_;
    return kResultOk;
}

tresult PLUGIN_API NeuralAmpProcessor::getState(IBStream* state)
{
    if (!state)
        return kResultFalse;
    IBStreamer streamer(state, kLittleEndian);
    for (const auto value : targetParams_)
        streamer.writeFloat(value);
    streamer.writeInt32(selectedModel_);
    streamer.writeFloat(targetVolume_);
    streamer.writeInt32(bypass_ ? 1 : 0);
    streamer.writeInt32(selectedFrameSizeIndex_);
    return kResultOk;
}

void NeuralAmpProcessor::consumeParameters(IParameterChanges* changes)
{
    if (!changes)
        return;
    const int32 count = changes->getParameterCount();
    for (int32 index = 0; index < count; ++index) {
        auto* queue = changes->getParameterData(index);
        if (!queue || queue->getPointCount() == 0)
            continue;
        int32 offset = 0;
        ParamValue value = 0.0;
        if (queue->getPoint(queue->getPointCount() - 1, offset, value) != kResultOk)
            continue;
        const float normalized = static_cast<float>(value);
        switch (queue->getParameterId()) {
        case kBass: targetParams_[0] = normalized; break;
        case kMiddle: targetParams_[1] = normalized; break;
        case kTreble: targetParams_[2] = normalized; break;
        case kGain: targetParams_[3] = normalized; break;
        case kVolume: targetVolume_ = normalized; break;
        case kBypass: bypass_ = normalized >= 0.5f; break;
        case kModel: {
            const int newModel = std::min(3, static_cast<int>(normalized * 4.0f));
            if (newModel != selectedModel_) {
                selectedModel_ = newModel;
            }
            break;
        }
        case kFrameSize:
            selectedFrameSizeIndex_ =
                std::min(3, static_cast<int>(normalized * 4.0f));
            break;
        default: break;
        }
    }
}

float NeuralAmpProcessor::smooth(float current, float target) const noexcept
{
    return current + smoothingCoefficient_ * (target - current);
}

tresult PLUGIN_API NeuralAmpProcessor::process(ProcessData& data)
{
    consumeParameters(data.inputParameterChanges);
    if (selectedFrameSizeIndex_ != appliedFrameSizeIndex_ && writePosition_ == 0)
        applyFrameSize();
    if (data.numSamples == 0)
        return kResultOk;
    if (data.symbolicSampleSize != kSample32)
        return kResultFalse;

    auto* input = data.inputs[0].channelBuffers32[0];
    auto* outputLeft = data.outputs[0].channelBuffers32[0];
    auto* outputRight = data.outputs[0].channelBuffers32[1];

    for (int32 sample = 0; sample < data.numSamples; ++sample) {
        if (bypass_) {
            outputLeft[sample] = input[sample];
            outputRight[sample] = input[sample];
            continue;
        }

        for (size_t parameter = 0; parameter < smoothedParams_.size(); ++parameter)
            smoothedParams_[parameter] =
                smooth(smoothedParams_[parameter], targetParams_[parameter]);
        smoothedVolume_ = smooth(smoothedVolume_, targetVolume_);

        inputFrame_[writePosition_++] = input[sample];
        float processed = outputReady_ ? outputFrame_[readPosition_] : input[sample];
        if (outputReady_)
            readPosition_ = (readPosition_ + 1) % outputFrame_.size();

        if (writePosition_ == inputFrame_.size()) {
            if (engine_ && engine_->processFrame(
                    inputFrame_.data(), smoothedParams_, outputFrame_.data())) {
                outputReady_ = true;
                readPosition_ = 0;
            }
            writePosition_ = 0;
        }

        processed *= smoothedVolume_ * 2.0f;
        outputLeft[sample] = processed;
        outputRight[sample] = processed;
    }
    return kResultOk;
}

bool NeuralAmpProcessor::loadSelectedModel()
{
    static const wchar_t* names[] = {
        L"stateful_gru.onnx",
        L"stateful_lstm.onnx",
        L"stateful_lru.onnx",
        L"stateful_wavenet.onnx",
    };
    if (!engine_)
        return false;
    loadedModel_ = -1;
    applyFrameSize();
    std::string error;
    const auto path =
        (std::filesystem::path(pluginDirectory_) / names[selectedModel_]).wstring();
    if (!engine_->load(path, error))
        return false;
    loadedModel_ = selectedModel_;
    return true;
}

bool NeuralAmpProcessor::applyFrameSize()
{
    if (!engine_)
        return false;
    // The current LRU export has a fixed 64-sample audio dimension.
    const int effectiveIndex = selectedModel_ == 2 ? 0 : selectedFrameSizeIndex_;
    std::string error;
    if (!engine_->setFrameSize(kFrameSizes[effectiveIndex], error))
        return false;
    inputFrame_.assign(static_cast<size_t>(kFrameSizes[effectiveIndex]), 0.0f);
    outputFrame_.assign(static_cast<size_t>(kFrameSizes[effectiveIndex]), 0.0f);
    appliedFrameSizeIndex_ = selectedFrameSizeIndex_;
    writePosition_ = 0;
    readPosition_ = 0;
    outputReady_ = false;
    engine_->resetState();
    return true;
}

void NeuralAmpProcessor::resetStreamingState()
{
    std::fill(inputFrame_.begin(), inputFrame_.end(), 0.0f);
    std::fill(outputFrame_.begin(), outputFrame_.end(), 0.0f);
    writePosition_ = 0;
    readPosition_ = 0;
    outputReady_ = false;
    if (engine_)
        engine_->resetState();
}
}

#pragma once

#include "onnx_stateful_engine.h"
#include "public.sdk/source/vst/vstaudioeffect.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace Steinberg::Vst {

class NeuralAmpProcessor final : public AudioEffect {
public:
    NeuralAmpProcessor();
    ~NeuralAmpProcessor() override;

    static FUnknown* createInstance(void*)
    {
        return static_cast<IAudioProcessor*>(new NeuralAmpProcessor());
    }

    tresult PLUGIN_API initialize(FUnknown* context) override;
    tresult PLUGIN_API setActive(TBool state) override;
    tresult PLUGIN_API setupProcessing(ProcessSetup& setup) override;
    tresult PLUGIN_API setBusArrangements(
        SpeakerArrangement* inputs, int32 numIns,
        SpeakerArrangement* outputs, int32 numOuts) override;
    tresult PLUGIN_API canProcessSampleSize(int32 symbolicSampleSize) override;
    tresult PLUGIN_API process(ProcessData& data) override;
    tresult PLUGIN_API setState(IBStream* state) override;
    tresult PLUGIN_API getState(IBStream* state) override;
    uint32 PLUGIN_API getLatencySamples() override;

private:
    bool loadSelectedModel();
    bool applyFrameSize();
    void resetStreamingState();
    void consumeParameters(IParameterChanges* changes);
    float smooth(float current, float target) const noexcept;

    std::unique_ptr<OnnxStatefulEngine> engine_;
    std::wstring pluginDirectory_;
    int selectedModel_ = 3;
    int loadedModel_ = -1;
    int selectedFrameSizeIndex_ = 0;
    int appliedFrameSizeIndex_ = -1;
    bool bypass_ = false;

    std::array<float, 4> targetParams_{0.5f, 0.5f, 0.5f, 0.5f};
    std::array<float, 4> smoothedParams_{0.5f, 0.5f, 0.5f, 0.5f};
    float targetVolume_ = 0.5f;
    float smoothedVolume_ = 0.5f;
    float smoothingCoefficient_ = 0.001f;

    std::vector<float> inputFrame_;
    std::vector<float> outputFrame_;
    size_t writePosition_ = 0;
    size_t readPosition_ = 0;
    bool outputReady_ = false;
};
}

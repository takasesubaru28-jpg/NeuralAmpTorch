#include "controller.h"

#include "parameter.h"
#include "base/source/fstreamer.h"
#include "public.sdk/source/vst/vstparameters.h"
#include "vstgui/plugin-bindings/vst3editor.h"

#include <algorithm>
#include <cstring>

namespace Steinberg::Vst {
IPlugView* PLUGIN_API NeuralAmpController::createView(const char* name)
{
    if (name && std::strcmp(name, ViewType::kEditor) == 0)
        return new VSTGUI::VST3Editor(this, "view", "design.uidesc");
    return nullptr;
}

tresult PLUGIN_API NeuralAmpController::setParamNormalized(
    ParamID tag, ParamValue value)
{
    const auto previous = getParamNormalized(tag);
    const auto result = EditController::setParamNormalized(tag, value);
    if (result == kResultOk && tag == kFrameSize && value != previous &&
        componentHandler)
        componentHandler->restartComponent(kLatencyChanged);
    return result;
}

tresult PLUGIN_API NeuralAmpController::initialize(FUnknown* context)
{
    const auto result = EditController::initialize(context);
    if (result != kResultOk)
        return result;

    parameters.addParameter(
        new RangeParameter(STR16("Bass"), kBass, nullptr, 0.0, 10.0, 5.0));
    parameters.addParameter(
        new RangeParameter(STR16("Middle"), kMiddle, nullptr, 0.0, 10.0, 5.0));
    parameters.addParameter(
        new RangeParameter(STR16("Treble"), kTreble, nullptr, 0.0, 10.0, 5.0));
    parameters.addParameter(
        new RangeParameter(STR16("Gain"), kGain, nullptr, 0.0, 10.0, 5.0));

    auto* model = new StringListParameter(STR16("Model"), kModel);
    model->appendString(STR16("Stateful GRU"));
    model->appendString(STR16("Stateful LSTM"));
    model->appendString(STR16("Stateful LRU"));
    model->appendString(STR16("Stateful WaveNet"));
    model->getInfo().defaultNormalizedValue = 1.0;
    model->setNormalized(1.0);
    parameters.addParameter(model);

    auto* frameSize = new StringListParameter(STR16("Frame Size"), kFrameSize);
    frameSize->appendString(STR16("64 samples"));
    frameSize->appendString(STR16("128 samples"));
    frameSize->appendString(STR16("256 samples"));
    frameSize->appendString(STR16("512 samples"));
    parameters.addParameter(frameSize);

    parameters.addParameter(
        new RangeParameter(STR16("Volume"), kVolume, STR16("x"), 0.0, 1.0, 0.5));
    parameters.addParameter(
        STR16("Bypass"), nullptr, 1, 0.0,
        ParameterInfo::kCanAutomate | ParameterInfo::kIsBypass, kBypass);
    return kResultOk;
}

tresult PLUGIN_API NeuralAmpController::setComponentState(IBStream* state)
{
    if (!state)
        return kResultFalse;
    IBStreamer streamer(state, kLittleEndian);
    for (ParamID id = kBass; id <= kGain; ++id) {
        float value = 0.0f;
        if (!streamer.readFloat(value))
            return kResultFalse;
        setParamNormalized(id, value);
    }
    int32 model = 0;
    float volume = 0.5f;
    int32 bypass = 0;
    if (!streamer.readInt32(model) ||
        !streamer.readFloat(volume) ||
        !streamer.readInt32(bypass))
        return kResultFalse;
    int32 frameSize = 0;
    setParamNormalized(kModel, static_cast<ParamValue>(model) / 3.0);
    setParamNormalized(kVolume, volume);
    setParamNormalized(kBypass, bypass ? 1.0 : 0.0);
    if (streamer.readInt32(frameSize))
        setParamNormalized(kFrameSize,
            static_cast<ParamValue>(std::clamp(frameSize, 0, 3)) / 3.0);
    return kResultOk;
}
}

#include "controller.h"
#include "fuid.h"
#include "processor.h"

#include "public.sdk/source/main/pluginfactory.h"

#define stringPluginName "NeuralAmp ONNX"

BEGIN_FACTORY_DEF(
    "NeuralAmpTorch",
    "https://example.invalid",
    "mailto:developer@example.invalid")

DEF_CLASS2(
    INLINE_UID_FROM_FUID(Steinberg::Vst::ProcessorUID),
    PClassInfo::kManyInstances,
    kVstAudioEffectClass,
    stringPluginName,
    Vst::kDistributable,
    "Fx|Distortion",
    FULL_VERSION_STR,
    kVstVersionString,
    Steinberg::Vst::NeuralAmpProcessor::createInstance)

DEF_CLASS2(
    INLINE_UID_FROM_FUID(Steinberg::Vst::ControllerUID),
    PClassInfo::kManyInstances,
    kVstComponentControllerClass,
    stringPluginName " Controller",
    0,
    "",
    FULL_VERSION_STR,
    kVstVersionString,
    Steinberg::Vst::NeuralAmpController::createInstance)

END_FACTORY

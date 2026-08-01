#pragma once

#include "public.sdk/source/vst/vsteditcontroller.h"

namespace Steinberg::Vst {
class NeuralAmpController final : public EditController {
public:
    static FUnknown* createInstance(void*)
    {
        return static_cast<IEditController*>(new NeuralAmpController());
    }
    tresult PLUGIN_API initialize(FUnknown* context) override;
    tresult PLUGIN_API setComponentState(IBStream* state) override;
    tresult PLUGIN_API setParamNormalized(ParamID tag, ParamValue value) override;
    IPlugView* PLUGIN_API createView(const char* name) override;
};
}

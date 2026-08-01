#pragma once

#include "pluginterfaces/base/ftypes.h"

namespace Steinberg::Vst {
enum ParameterIds : ParamID {
    kBass = 0,
    kMiddle,
    kTreble,
    kGain,
    kModel,
    kFrameSize,
    kVolume,
    kBypass
};
}

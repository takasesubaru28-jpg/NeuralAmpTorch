// VST3 SDKのインクルードファイル
#include "public.sdk/source/main/pluginfactory.h"

// 自作VSTのヘッダファイルをインクルード
#include "fuid.h"
#include "controller.h"
#include "processor.h"

// 製作者(製作会社)の名前。終端文字「\0」含めて64文字まで。
#define MYVST_VENDOR   "takase.subaru28"

// 製作者(製作会社)のWebサイトのURL。終端文字「\0」含めて256文字まで。
#define MYVST_URL      "none"

// 製作者(製作会社)の連絡先メールアドレス。終端文字「\0」含めて128文字まで。
#define MYVST_EMAIL    "takase.subaru28@gmail.com"

// 自作するVSTの名前。終端文字「\0」含めて64文字まで。
#define MYVST_VSTNAME  "NeuralAmpTorch"

// 自作するVSTのバージョン。終端文字「\0」含めて64文字まで。
#define MYVST_VERSION  "0" 

// 自作するVSTのカテゴリ。終端文字「\0」含めて64文字まで。
#define MYVST_SUBCATEGORIES Vst::PlugType::kFx


// ===================================================================================
// DLLファイルの初期化、終了処理関数
// ===================================================================================
// 基本的に何もする必要はない。(VST SDK 3.7.1より前のバージョンの場合はコメントを外すこと)
//bool InitModule() { return true; }
//bool DeinitModule() { return true; }


// ===================================================================================
// 自作VSTプラグインの生成
// ===================================================================================
BEGIN_FACTORY_DEF(MYVST_VENDOR, MYVST_URL, MYVST_EMAIL)

// Processorクラスの作成を行う
DEF_CLASS2(INLINE_UID_FROM_FUID(Steinberg::Vst::ProcessorUID), // fuid.hの名前
    PClassInfo::kManyInstances,
    kVstAudioEffectClass,
    MYVST_VSTNAME,
    Vst::kDistributable,
    MYVST_SUBCATEGORIES,
    MYVST_VERSION,
    kVstVersionString,
    Steinberg::Vst::MyVSTProcessor::createInstance) // processor.hのクラス名

    // 2. Controllerの登録
DEF_CLASS2(INLINE_UID_FROM_FUID(Steinberg::Vst::ControllerUID), // fuid.hの名前
    PClassInfo::kManyInstances,
    kVstComponentControllerClass,
    MYVST_VSTNAME " Controller",
    0,
    "",
    MYVST_VERSION,
    kVstVersionString,
    Steinberg::Vst::Controller::createInstance)

END_FACTORY
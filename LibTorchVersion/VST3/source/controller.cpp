// 自作VST用のインクルードファイル
#include "parameter.h"
#include "fuid.h"
#include "controller.h"
#include "pluginterfaces/base/ustring.h"

#include "public.sdk/source/vst/vstparameters.h"

#include "vstgui/plugin-bindings/vst3editor.h"
#include "public.sdk/source/vst/vsteditcontroller.h"
#include "pluginterfaces/vst/ivstparameterfunctionname.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/base/ustring.h"
#include <string_view> // これを冒頭に追加
#include <cstring>           // strcmp を使うために必要
#include "vstgui/lib/vstguibase.h"
#include <windows.h> 


namespace Steinberg {
	namespace Vst {
			// クラスを初期化する関数
			tresult PLUGIN_API Controller::initialize(FUnknown* context)
			{
				// まず継承元クラスの初期化
				tresult result = EditController::initialize(context);
				if (result == kResultTrue)
				{
					extern void* gInstance; // 通常、VST3 SDKが内部で保持しているhInstance
					//VSTGUI::init(gInstance);

					//tresult result = EditController::initialize(context);
					//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
					parameters.addParameter(STR16("bypass"), STR16("..."), 1, 0, ParameterInfo::kIsBypass, BYPASS_TAG);
					// 範囲パラメーターを作成
					RangeParameter* treble = new RangeParameter(STR16("treble"), TREBLE, STR16(""), 0.0f, 10.0f, 5.0f);
					treble->setPrecision(2); // 小数第何位まで表示するか
					// 範囲パラメーターをコントローラーに追加
					parameters.addParameter(treble);

					// 範囲パラメーターを作成
					RangeParameter* midle = new RangeParameter(STR16("midle"), MIDDLE, STR16(""), 0.0f, 10.0f, 5.0f);
					midle->setPrecision(2); // 小数第何位まで表示するか
					// 範囲パラメーターをコントローラーに追加
					parameters.addParameter(midle);

					// 範囲パラメーターを作成
					RangeParameter* bass = new RangeParameter(STR16("bass"), BASS, STR16(""), 0.0f, 10.0f, 5.0f);
					bass->setPrecision(2); // 小数第何位まで表示するか
					// 範囲パラメーターをコントローラーに追加
					parameters.addParameter(bass);

					// 範囲パラメーターを作成
					RangeParameter* gain = new RangeParameter(STR16("gain"), GAIN, STR16(""), 0.0f, 10.0f, 5.0f);
					gain->setPrecision(2); // 小数第何位まで表示するか
					// 範囲パラメーターをコントローラーに追加
					parameters.addParameter(gain);

					auto* gan = new StringListParameter(USTRING("GAN"), GAN);
					gan->appendString(USTRING("off"));
					gan->appendString(USTRING("on"));					
					gan->setNormalized(0.0f);
					parameters.addParameter(gan);

					//parameters.addParameter(STR16("GAN"), STR16("..."), 1, 0, ParameterInfo::kCanAutomate, GAN);

					auto* model = new StringListParameter(USTRING("model"), MODEL);
					model->appendString(USTRING("LSTM"));
					model->appendString(USTRING("LSTM_2layer"));
					model->appendString(USTRING("WaveNet"));
					model->appendString(USTRING("WaveNet_LSTM"));
					model->setNormalized(0.0f);
					parameters.addParameter(model);
				
					RangeParameter* volume = new RangeParameter(STR16("volume"), VOLUME, STR16("dB"), 0.0f, 1.0f, 0.5f);
					parameters.addParameter(volume);

				}
				result = kResultTrue;
				return result;
			}

			IPlugView* PLUGIN_API Controller::createView(const char* name)
			{
				// strcmp を使うために <cstring> が必要
				if (strcmp(name, ViewType::kEditor) == 0)
				{
					// "view" は design.uidesc 内のテンプレート名
					// "design.uidesc" は .rc ファイルで定義した名前
					return new VSTGUI::VST3Editor(this, "view", "design.uidesc");
				}
				return nullptr;
			}
	}
}
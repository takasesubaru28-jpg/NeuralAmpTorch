#pragma once
// VST3 SDKのインクルードファイル
#include "public.sdk/source/vst/vsteditcontroller.h"


// 自作VST用のインクルードファイル
#include "parameter.h"
#include "pluginterfaces/base/ftypes.h"
#include "pluginterfaces/base/ustring.h"
#include <cmath>
#include "vstgui\plugin-bindings/vst3editor.h"

namespace Steinberg {
	namespace Vst {
			// パラメータを操作するためのControllerクラス
			class Controller : public EditController
	
			{
			public:
				// クラスを初期化する関数
				tresult PLUGIN_API initialize(FUnknown* context);

				IPlugView* PLUGIN_API createView(const char* name);

				// 自作VST Controllerクラスのインスタンスを作成するための関数
				static FUnknown* createInstance(void*) { return (IEditController*) new Controller(); }

				// VST基本インターフェースをオーバーライドした場合に必要な宣言
				OBJ_METHODS(Controller, EditController)
					DEFINE_INTERFACES
					END_DEFINE_INTERFACES(EditController)
					REFCOUNT_METHODS(EditController)
			};

	}
}
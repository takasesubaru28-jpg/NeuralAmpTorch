cmake_minimum_required(VERSION 3.15)
set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "Windows SDK version" FORCE)
project(NeuralAmpTorch VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 1. パス設定 ---
# 現在のソースディレクトリ（CMakeLists.txtがある場所）の「親」を取得
get_filename_component(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
# その親ディレクトリの中にある vst3sdk を指す
set(VST3_SDK_ROOT "${/vst3sdk")
set(LIBTORCH_ROOT "${/LibTorch/release/libtorch")

# --- 2. ターゲット作成 (通常のDLLとして定義) ---
add_library(NeuralAmpTorch MODULE
    source/factory.cpp
    source/processor.cpp
    source/processor.h
    source/controller.cpp
    source/controller.h
    "${VST3_SDK_ROOT}/public.sdk/source/main/dllmain.cpp"
)

# --- 3. インクルードディレクトリの設定 ---
target_include_directories(NeuralAmpTorch PRIVATE 
    source
    "${VST3_SDK_ROOT}"
    "${VST3_SDK_ROOT}/vstgui4"
    "${LIBTORCH_ROOT}/include"
    "${LIBTORCH_ROOT}/include/torch/csrc/api/include"
)

# --- 4. プリプロセッサの定義 ---
target_compile_definitions(NeuralAmpTorch PRIVATE 
    VSTGUI_LIVE_EDITING=1
    DEVELOPMENT=1
    CPU_ONLY=1
    WIN32=1
    _WINDOWS=1
)

# --- 5. リンカー設定 (.lib を直接指定) ---
# うつぼ箱さんの設定通り、SDKの out フォルダにある lib を直接リンクします
# ※パスはビルド済みの lib が置かれている場所に書き換えてください
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(VST3_LIB_DIR "${VST3_SDK_ROOT}/out/build/x64-Debug/lib")
else()
    # Releaseモードのときはこっちのパスを使う
    set(VST3_LIB_DIR "${VST3_SDK_ROOT}/out/build/x64-Release/lib")
endif()

target_link_libraries(NeuralAmpTorch PRIVATE 
    "${VST3_LIB_DIR}/base.lib"
    "${VST3_LIB_DIR}/sdk.lib"
    "${VST3_LIB_DIR}/sdk_common.lib"
    "${VST3_LIB_DIR}/pluginterfaces.lib"
    "${VST3_LIB_DIR}/vstgui.lib"
    "${VST3_LIB_DIR}/vstgui_support.lib"
    "${VST3_LIB_DIR}/vstgui_uidescription.lib"
    "${LIBTORCH_ROOT}/lib/torch.lib"
    "${LIBTORCH_ROOT}/lib/torch_cpu.lib"
    "${LIBTORCH_ROOT}/lib/c10.lib"
)

# --- 6. 仕上げの設定 ---
# 拡張子を .vst3 に変更
set_target_properties(NeuralAmpTorch PROPERTIES SUFFIX ".vst3")

# 例外処理とランタイム設定
target_compile_options(NeuralAmpTorch PRIVATE /EHsc)

# Windows用のエクスポート関数を明示
if(WIN32)
    target_link_options(NeuralAmpTorch PRIVATE /EXPORT:GetPluginFactory /EXPORT:InitDll /EXPORT:ExitDll)
endif()
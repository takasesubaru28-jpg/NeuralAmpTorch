cmake_minimum_required(VERSION 3.15)
set(CMAKE_SYSTEM_VERSION 10.0 CACHE STRING "Windows SDK version" FORCE)
project(NeuralAmpTorch VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 1. パス設定 ---
# 現在のソースディレクトリ（CMakeLists.txtがある場所）の「親」を取得
get_filename_component(PROJECT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(WORKSPACE_ROOT "${PROJECT_DIR}" DIRECTORY)
set(SHARED_DEPENDENCIES "${WORKSPACE_ROOT}/Shared/Dependencies")
set(VST3_SDK_ROOT "${SHARED_DEPENDENCIES}/vst3sdk")
set(LIBTORCH_ROOT "${SHARED_DEPENDENCIES}/LibTorch/release/libtorch")

# --- 2. ターゲット作成 (通常のDLLとして定義) ---
add_library(NeuralAmpTorch MODULE
    source/factory.cpp
    source/processor.cpp
    source/processor.h
    source/libtorch_delay_load.cpp
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
    target_link_libraries(NeuralAmpTorch PRIVATE delayimp)
    target_link_options(NeuralAmpTorch PRIVATE
        /EXPORT:GetPluginFactory
        /EXPORT:InitDll
        /EXPORT:ExitDll
        /DELAYLOAD:torch_cpu.dll
        /DEPENDENTLOADFLAG:0x900
    )

    # Keep the runtime next to the plug-in binary so DAWs can load it without
    # adding LibTorch to the machine-wide PATH.  This build links only the CPU
    # backend, but the installed cu126 package makes torch_cpu.dll depend on
    # the CUDA runtime/profiler DLLs below as well.
    set(PLUGIN_BINARY_DIR
        "${PROJECT_DIR}/dist/NeuralAmpTorch.vst3/Contents/x86_64-win")
    set(LIBTORCH_RUNTIME_DLLS
        torch_cpu.dll
        c10.dll
        libiomp5md.dll
        cudart64_12.dll
        cupti64_2024.3.2.dll
    )

    foreach(RUNTIME_DLL IN LISTS LIBTORCH_RUNTIME_DLLS)
        if(NOT EXISTS "${LIBTORCH_ROOT}/lib/${RUNTIME_DLL}")
            message(FATAL_ERROR
                "Required LibTorch runtime is missing: ${LIBTORCH_ROOT}/lib/${RUNTIME_DLL}")
        endif()
    endforeach()

    add_custom_command(TARGET NeuralAmpTorch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${PLUGIN_BINARY_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:NeuralAmpTorch>"
            "${PLUGIN_BINARY_DIR}/NeuralAmpTorch.vst3"
        VERBATIM
    )
    foreach(RUNTIME_DLL IN LISTS LIBTORCH_RUNTIME_DLLS)
        add_custom_command(TARGET NeuralAmpTorch POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LIBTORCH_ROOT}/lib/${RUNTIME_DLL}"
                "${PLUGIN_BINARY_DIR}/${RUNTIME_DLL}"
            VERBATIM
        )
    endforeach()
endif()

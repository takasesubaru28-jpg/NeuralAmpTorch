cmake_minimum_required(VERSION 3.20)
project(NeuralAmpOnnx VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VST3_SDK_ROOT "" CACHE PATH "Path to the Steinberg VST3 SDK")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime package")

if(NOT EXISTS "${VST3_SDK_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "Set VST3_SDK_ROOT to the Steinberg VST3 SDK directory")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an extracted ONNX Runtime package")
endif()

add_subdirectory("${VST3_SDK_ROOT}" "${CMAKE_BINARY_DIR}/vst3sdk")

smtg_enable_vst3_sdk()

smtg_add_vst3plugin(NeuralAmpOnnx
    source/factory.cpp
    source/processor.cpp
    source/processor.h
    source/controller.cpp
    source/controller.h
    source/onnx_stateful_engine.cpp
    source/onnx_stateful_engine.h
    source/onnx_delay_load.cpp
    source/fuid.h
    source/parameter.h
    source/design.uidesc
)

target_include_directories(NeuralAmpOnnx PRIVATE
    source
    "${ONNXRUNTIME_ROOT}/include"
)

# Do not retain a library path from a dependency directory used by an older
# workspace layout.
unset(ONNXRUNTIME_LIBRARY CACHE)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
target_link_libraries(NeuralAmpOnnx PRIVATE
    sdk
    vstgui_support
    ${ONNXRUNTIME_LIBRARY}
    delayimp
)

smtg_target_add_plugin_resources(NeuralAmpOnnx
    RESOURCES
        source/design.uidesc
        source/Amp.png
        source/knob_strip.png
        source/power_strip.png
        source/togle_strip.png
)
target_compile_definitions(NeuralAmpOnnx PRIVATE
    FULL_VERSION_STR="${PROJECT_VERSION}"
    NOMINMAX
)

if(MSVC)
    target_compile_options(NeuralAmpOnnx PRIVATE /EHsc /permissive-)
    target_link_options(NeuralAmpOnnx PRIVATE /DELAYLOAD:onnxruntime.dll)
endif()

set_target_properties(NeuralAmpOnnx PROPERTIES
    SMTG_PLUGIN_BUNDLE_NAME "NeuralAmpOnnx"
)

if(WIN32)
    add_custom_command(TARGET NeuralAmpOnnx PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:NeuralAmpOnnx>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:NeuralAmpOnnx>"
    )
    foreach(MODEL_NAME gru lstm lru wavenet)
        add_custom_command(TARGET NeuralAmpOnnx PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/../Python/exports/stateful_${MODEL_NAME}.onnx"
                "$<TARGET_FILE_DIR:NeuralAmpOnnx>"
        )
    endforeach()
endif()

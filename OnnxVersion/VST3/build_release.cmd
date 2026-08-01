@echo off
setlocal

rem Some sandboxed shells expose both PATH and Path. MSBuild rejects duplicates.
set "PATH="

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo Visual Studio 2022 C++ developer environment was not found.
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

cmake -S "%~dp0." -B "%~dp0build-release-final" ^
    -DVST3_SDK_ROOT="%~dp0..\..\Shared\Dependencies\vst3sdk" ^
    -DONNXRUNTIME_ROOT="%~dp0..\..\Shared\Dependencies\onnxruntime-win-x64-1.28.0" ^
    -DSMTG_CREATE_PLUGIN_LINK=OFF
if errorlevel 1 exit /b %errorlevel%

cmake --build "%~dp0build-release-final" --config Release --target NeuralAmpOnnx
exit /b %errorlevel%

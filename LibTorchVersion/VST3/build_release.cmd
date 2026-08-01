@echo off
setlocal

rem Some shells expose both PATH and Path. MSBuild rejects duplicates.
set "PATH="

set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo Visual Studio 2022 C++ developer environment was not found.
    exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

cmake -S "%~dp0." -B "%~dp0build-release" ^
    -G "Visual Studio 17 2022" -A x64
if errorlevel 1 exit /b %errorlevel%

cmake --build "%~dp0build-release" --config Release --target NeuralAmpTorch
exit /b %errorlevel%

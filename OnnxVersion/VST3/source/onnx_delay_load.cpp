#if defined(_WIN32)

#include <windows.h>
#include <delayimp.h>

#include <cstring>
#include <filesystem>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {
FARPROC WINAPI loadBundledOnnxRuntime(
    unsigned notification, PDelayLoadInfo delayInfo)
{
    if (notification != dliNotePreLoadLibrary || !delayInfo ||
        !delayInfo->szDll ||
        _stricmp(delayInfo->szDll, "onnxruntime.dll") != 0)
        return nullptr;

    wchar_t modulePath[MAX_PATH]{};
    if (GetModuleFileNameW(
            reinterpret_cast<HMODULE>(&__ImageBase),
            modulePath,
            MAX_PATH) == 0)
        return nullptr;

    const auto runtimePath =
        std::filesystem::path(modulePath).parent_path() / L"onnxruntime.dll";
    const auto module = LoadLibraryExW(
        runtimePath.c_str(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    return reinterpret_cast<FARPROC>(module);
}
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = loadBundledOnnxRuntime;

#endif

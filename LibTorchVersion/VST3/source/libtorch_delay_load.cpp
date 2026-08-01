#if defined(_WIN32)

#include <windows.h>
#include <delayimp.h>

#include <cstring>
#include <filesystem>
#include <string>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {
HMODULE loadRuntimeModule(const char* dllName)
{
    wchar_t modulePath[MAX_PATH]{};
    if (GetModuleFileNameW(
            reinterpret_cast<HMODULE>(&__ImageBase),
            modulePath,
            MAX_PATH) == 0)
        return nullptr;

    const auto runtimePath =
        std::filesystem::path(modulePath).parent_path() /
        std::filesystem::path(dllName);
    return LoadLibraryExW(
        runtimePath.c_str(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}

FARPROC WINAPI loadBundledLibTorch(
    unsigned notification, PDelayLoadInfo delayInfo)
{
    if (notification != dliNotePreLoadLibrary || !delayInfo ||
        !delayInfo->szDll)
        return nullptr;

    const bool isLibTorch =
        _stricmp(delayInfo->szDll, "torch_cpu.dll") == 0;
    if (!isLibTorch)
        return nullptr;

    return reinterpret_cast<FARPROC>(loadRuntimeModule(delayInfo->szDll));
}
}

bool loadBundledLibTorchRuntime(std::string& error)
{
    if (GetModuleHandleW(L"torch_cpu.dll") ||
        loadRuntimeModule("torch_cpu.dll"))
        return true;
    error = "Failed to load bundled torch_cpu.dll (Windows error " +
        std::to_string(GetLastError()) + ")";
    return false;
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = loadBundledLibTorch;

#endif

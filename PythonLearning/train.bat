@echo off
setlocal enabledelayedexpansion
chcp 932 > nul

:: --- 環境準備 ---
if exist venv\Scripts\activate (
    call venv\Scripts\activate
) else (
    echo Can not found env path
    pause
    exit /b
)

:: --- TensorBoard 起動 ---
start "TensorBoard" /min cmd /c "tensorboard --logdir=runs --host 0.0.0.0"
timeout /t 3 > nul
start http://localhost:6006

:: --- 探索の起点 ---
set "target=.\configs"

if not exist "%target%" (
    echo [エラー] %target% フォルダが見つかりません。
    pause
    exit /b
)


set "base_dir=%cd%\"

:: --- 再帰的に探索 ---
for /r "%target%" %%i in (*.json) do (
    if exist "%%i" (
        set "full_path=%%i"

        :: フルパスから実行場所のパスを削除して相対パスにする
        set "rel_path=!full_path:%base_dir%=.\!"
        
        echo [Training] !rel_path! ...
        python train.py --config "!rel_path!"
    )
)

echo All Train Finished!
pause
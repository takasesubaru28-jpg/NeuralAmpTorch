# Shared：共通データと依存ライブラリ

LibTorch版とONNX版の両方が参照する大容量ファイルを一か所に集約します。

## 完成形

```text
Shared/
├─ Data/
│  ├─ sample/                  # Gitに含まれる約4MBのお試しデータ
│  │  ├─ input/input.wav
│  │  └─ target/*.wav
│  ├─ main/                    # 本学習データ（Git対象外）
│  │  ├─ input/input.wav
│  │  ├─ target/*.wav
│  │  └─ valid/                # LibTorch版で使用する任意の検証データ
│  └─ legacy_libtorch/         # 移行前データの保管。現在は未使用
└─ Dependencies/               # すべてGit対象外
   ├─ LibTorch/release/libtorch/
   │  ├─ include/
   │  └─ lib/
   ├─ vst3sdk/
   │  ├─ CMakeLists.txt
   │  ├─ public.sdk/
   │  └─ vstgui4/
   └─ onnxruntime-win-x64-1.28.0/
      ├─ include/
      └─ lib/onnxruntime.dll
```

## お試しデータの形式

- WAV、44.1kHz
- `input/input.wav`はアンプへ入力した信号
- `target/*.wav`は同じ信号をアンプへ通した出力
- 入力とターゲットはサンプルレート、長さ、時間位置を一致させる
- ターゲット名の末尾を`bass&middle&treble&gain.wav`にする
- 値は0.0～10.0で記述する

例:

```text
0000_8.8&7.0&7.8&2.9.wav
```

## 本学習データの追加

大容量データは`Shared/Data/main`へ置きます。両プロジェクトの通常設定はここを参照します。
新しいデータを追加した後、ONNX版では次で整合性を検査できます。

```powershell
cd OnnxVersion\Python
python validate_data.py `
  --input ..\..\Shared\Data\main\input\input.wav `
  --targets ..\..\Shared\Data\main\target
```

## 外部ライブラリの準備

1. Steinberg VST3 SDKを`Shared/Dependencies/vst3sdk`へ展開します。
2. Windows x64用LibTorchを展開し、最終的に
   `Shared/Dependencies/LibTorch/release/libtorch/include`と`lib`が存在するようにします。
3. ONNX Runtime CPU Windows x64 1.28.0を
   `Shared/Dependencies/onnxruntime-win-x64-1.28.0`へ展開します。

確認対象:

```text
Shared/Dependencies/vst3sdk/CMakeLists.txt
Shared/Dependencies/LibTorch/release/libtorch/lib/torch_cpu.dll
Shared/Dependencies/onnxruntime-win-x64-1.28.0/lib/onnxruntime.dll
```

外部ライブラリはリポジトリへcommitしません。完成プラグインに必要なDLLだけが各`dist`の
バンドル内へコピーされます。


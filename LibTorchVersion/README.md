# NeuralAmp LibTorch版

PyTorchで学習し、TorchScriptモデルをLibTorchで推論するVST3です。

## 1. 必要環境

- Windows 10/11 x64
- Visual Studio 2022の「C++によるデスクトップ開発」
- CMake
- Python 3
- 学習にはPyTorch、SciPyなど（`Python/requirements.txt`）
- ビルドには[Sharedの依存ライブラリ](../Shared/README.md)

## 2. Python環境

```powershell
cd LibTorchVersion\Python
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU学習を行う場合は、GPU・ドライバーに対応したPyTorchを使用してください。プラグインの
ビルドで使用するLibTorchも、保存したTorchScriptモデルと互換性のある版を選びます。

## 3. お試し学習

Gitに含まれる約4MBのサンプルだけを使用し、小型LSTMを1エポック学習します。

```powershell
cd LibTorchVersion\Python
.\venv\Scripts\Activate.ps1
python train.py --config configs\sample_lstm.json
```

結果:

- TensorBoardログ: `Python/runs/sample_lstm`
- checkpointとTorchScript: `Python/checkpoints/sample_lstm`

サンプルは処理確認用なので、生成モデルの音質は実用水準になりません。

## 4. 本学習

1. `Shared/Data/main`へ入力とターゲットを配置します。
2. `Python/configs/*.json`からモデル設定を選びます。
3. 学習を実行します。

```powershell
cd LibTorchVersion\Python
python train.py --config configs\LSTM.json
```

設定の`paths.input_wav`と`paths.target_dir`は共通データを指します。検証用データを分ける場合は
`validation_input_wav`と`validation_target_dir`も指定できます。

## 5. 学習モデルをプラグインへ入れる

学習終了時に`checkpoints/<experiment_name>/`へTorchScript `.pt`が保存されます。使用するモデルを
プラグインが期待する名前に合わせ、次へコピーします。

```text
LibTorchVersion/dist/NeuralAmpTorch.vst3/Contents/x86_64-win/
```

主な名前は`LSTM_1024.pt`、`LSTM_2lay_1024.pt`、`WaveNet_1024.pt`などです。モデル構造と
ファイル名を`VST3/source/processor.cpp`の選択処理に合わせてください。

## 6. VST3をビルド

```powershell
LibTorchVersion\VST3\build_release.cmd
```

完成品:

```text
LibTorchVersion/dist/NeuralAmpTorch.vst3
```

ビルド時に`torch_cpu.dll`、`c10.dll`などがバンドル内部へコピーされます。`PATH`へのLibTorch追加は
不要です。LibTorchパッケージが要求するDLL名が変わった場合は`VST3/CMakeLists.txt`の
`LIBTORCH_RUNTIME_DLLS`を更新します。

## 7. Cubase 13 Proへ導入

1. Cubaseを終了します。
2. 古い`NeuralAmpTorch.vst3`がある場合は別の場所へ退避します。
3. `LibTorchVersion/dist/NeuralAmpTorch.vst3`フォルダー全体を
   `C:\Program Files\Common Files\VST3\`へコピーします。
4. Cubaseを起動し、VSTプラグインマネージャーで再スキャンします。
5. オーディオトラックのInsertから`NeuralAmpTorch`を選びます。

本体、`.pt`、`torch_cpu.dll`などは必ず同じ`.vst3`バンドル内に置きます。プラグインが見つからない
場合はCubaseのブロックリスト、DLL不足、x64版であることを確認してください。

## 8. 更新時

Cubaseを終了してから完成バンドルを上書きします。読み込みキャッシュが残る場合はプラグイン
マネージャーから再スキャンし、それでも反映されなければCubaseを再起動します。


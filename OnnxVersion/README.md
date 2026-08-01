# NeuralAmp ONNX Runtime版

StatefulモデルをONNXへ変換し、ONNX Runtime CPUで推論するVST3です。UIからモデル、Frame Size、
Gain、3バンドEQ、Volume、Bypassを変更できます。

## 1. 必要環境

- Windows 10/11 x64
- Visual Studio 2022の「C++によるデスクトップ開発」
- CMake
- Python 3
- 学習・変換には`Python/requirements.txt`
- ビルドには[Sharedの依存ライブラリ](../Shared/README.md)

## 2. Python環境

```powershell
cd OnnxVersion\Python
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. お試し学習

Gitに含まれる約4MBのサンプルで、小型WaveNetを1エポック学習します。

```powershell
cd OnnxVersion\Python
.\venv\Scripts\Activate.ps1
python train.py --config configs\sample_wavenet.json
```

結果:

- TensorBoardログ: `Python/runs/sample_wavenet`
- checkpoint: `Python/checkpoints/sample_wavenet`
- 評価音声: `Python/evaluations/sample_wavenet`

サンプルは学習処理の確認専用で、実用的な音質には不足します。

## 4. 本学習

1. `Shared/Data/main`へ入力とターゲットを配置します。
2. データを検査します。
3. GRU、LSTM、LRU、WaveNetなどの設定を選んで学習します。

```powershell
cd OnnxVersion\Python
python validate_data.py `
  --input ..\..\Shared\Data\main\input\input.wav `
  --targets ..\..\Shared\Data\main\target
python train.py --config configs\wavenet_direct_nogan.json
```

ログは`runs`、checkpointは`checkpoints`、比較用音声は`evaluations`へ保存され、すべてGit対象外です。

## 5. ONNXへ変換・検証

checkpointを指定してエクスポートします。

```powershell
cd OnnxVersion\Python
python export_onnx.py `
  --config configs\wavenet_direct_nogan.json `
  --checkpoint checkpoints\stateful_wavenet_direct_nogan\best.pt `
  --output exports\stateful_wavenet.onnx
python verify_onnx.py --model exports\stateful_wavenet.onnx
```

VST3ビルドは`Python/exports`の次の名前を参照します。

```text
stateful_gru.onnx
stateful_lstm.onnx
stateful_lru.onnx
stateful_wavenet.onnx
```

## 6. VST3をビルド

```powershell
OnnxVersion\VST3\build_release.cmd
```

完成品:

```text
OnnxVersion/dist/NeuralAmpOnnx.vst3
```

ビルド後のバンドルには`onnxruntime.dll`と4モデルを同梱します。現在のFrame Sizeは
64 / 128 / 256 / 512で、LRUモデルはモデル仕様により64を使用します。

## 7. Cubase 13 Proへ導入

1. Cubaseを終了します。
2. 古い`NeuralAmpOnnx.vst3`がある場合は別の場所へ退避します。
3. `OnnxVersion/dist/NeuralAmpOnnx.vst3`フォルダー全体を
   `C:\Program Files\Common Files\VST3\`へコピーします。
4. Cubaseを起動し、VSTプラグインマネージャーで再スキャンします。
5. オーディオトラックのInsertから`NeuralAmpOnnx`を選びます。

本体、`onnxruntime.dll`、`.onnx`は必ず同じバンドル内部に置きます。プラグインが見つからない
場合はCubaseのブロックリスト、x64版、DLLとモデルの配置を確認してください。

## 8. Frame Sizeの選び方

小さい値はレイテンシーを抑えますが推論回数が増えます。大きい値は呼び出し回数を減らせますが、
モデルやCPUによっては一回の計算が重くなります。Cubaseのオーディオドロップアウトを確認しながら
64、128、256、512を切り替えてください。


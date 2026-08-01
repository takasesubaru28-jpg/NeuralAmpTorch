# NeuralAmp VST3

ニューラルネットワークでギターアンプの特性を再現するWindows x64向けVST3です。
LibTorch版とONNX Runtime版を収録し、学習データと外部ライブラリを共有します。

## 構成

```text
NeuralAmpTorch/
├─ LibTorchVersion/
│  ├─ Python/       # 学習
│  ├─ VST3/         # プラグインソース
│  └─ dist/         # 完成プラグイン
├─ OnnxVersion/
│  ├─ Python/       # 学習・ONNX変換
│  ├─ VST3/         # プラグインソース
│  └─ dist/         # 完成プラグイン
└─ Shared/
   ├─ Data/         # 共通学習データ
   └─ Dependencies/ # LibTorch、VST3 SDK、ONNX Runtime
```

詳しい手順:

- [LibTorch版：学習・ビルド・導入](LibTorchVersion/README.md)
- [ONNX版：学習・変換・ビルド・導入](OnnxVersion/README.md)
- [共有データと外部ライブラリの配置](Shared/README.md)

## 最短で試す

完成品を使うだけなら学習環境やSDKは不要です。Cubaseを終了し、どちらかのフォルダーを
丸ごと`C:\Program Files\Common Files\VST3\`へコピーします。

- `LibTorchVersion/dist/NeuralAmpTorch.vst3`
- `OnnxVersion/dist/NeuralAmpOnnx.vst3`

DLLやモデルをバンドル外へ移動しないでください。LibTorchのシステム`PATH`設定も不要です。

## お試し学習データ

`Shared/Data/sample`に約4MBの短い44.1kHz WAVを収録しています。入力1本と、異なるアンプ
パラメーターで録音したターゲット2本です。学習コードの動作確認用であり、高品質なモデルを
作るためのデータ量ではありません。

## Git管理

サンプル以外の学習データ、外部SDK、仮想環境、ログ、checkpoint、評価音声、ビルド生成物は
`.gitignore`で除外します。完成バンドル内のDLL・モデル・WAVなどのバイナリはGit LFS対象です。

```powershell
git lfs install
git lfs pull
```

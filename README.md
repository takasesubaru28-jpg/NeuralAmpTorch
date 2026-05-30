# NeuralAmpTorch 🎸
Neural Networkを用いた音声合成およびアンプシミュレーションの実験プロジェクト

## 概要
本プロジェクトは，実機アンプの非線形特性をニューラルネットワーク（LibTorch/Python）で再現するシミュレータの開発，および音声合成技術の応用を目的としています．
VSTとしてCubaseで動作させたときにまだ不安定な部分があります．
今後はStatefull推論実装による軽量化や入力パラメータをgainのみにしてEQはモデリングにするなどの改善点が考えられます．


[![Demo Video](https://img.youtube.com/vi/5a-L87VCdNo/0.jpg)](https://youtu.be/5a-L87VCdNo)

## 環境
Windows11，RTX306012GBのCUDA環境
プラグインを動かすにはcu126GPUのLibTorchを環境変数に追加する必要があります！！！
モデルの入力長の都合上バッファサイズ512だと安定しやすいです．

## ディレクトリ構成
```text
NeuralAmpTorch/           # VST用ソースコード
├── LibTorch/             # LibTorchを置いておく
├── vst3sdk/              # vst3sdkを置いておく
├── source/               # VST用のソースコードが入っています
├── README.md
└── .gitignore
NeuralAmpTorch.vst3/contents/  # VST本体
PythonLearning/           # 機械学習用のコード
├── configs/              # 学習設定ファイル
├── data/                 # 学習，評価データ
├── xxx.py                # 学習スクリプト
├── train.bat             # これを実行することで学習を実行（Windows）
└── .gitignore
```

## プラグイン本体
NeuralAmpTorch.vst3を自身のvst3ファイルが認識されるフォルダに丸ごと移動して下さい
#### ⚠️ バージョンに関する重要事項
なおcu126GPUのLibTorchをシステム変数に追加してください．そうしないと動きません．VSTにこのライブラリを組み込む方法は模索中です．．．

# 実行手順 (Usage)

本リポジトリでは、GitHubの容量制限（100MB）を回避しつつ、効率的に学習を行うためのワークフローを採用しています。

## モデル学習（PythonLearning）
### 0. 学習データ
Gitに含まれるのは一部のデータです．実際にモデル学習に使用したのは20分程度のwavデータに対し，パラメータの組み合わせが100通りのものです．Gitの容量の都合上inputの長さを4分の1に，パラメータの組み合わせを3つのものを同梱しています．

### 1. 環境構築
まず、必要なライブラリをインストールします。
```bash
pip install -r requirements.txt
```
### 2．実行
```bash
cd ./PythonLearning
./train.bat
```

## VSTの作成（NeuralAmpTorch）
### 0．ライブラリ配置
#### 外部ライブラリの配置
本リポジトリにはライブラリ本体は含まれていません。以下の構成になるように各自で配置してください。

1. **VST3 SDK**:
   - [Steinberg公式サイト](https://www.steinberg.net/developers/)からダウンロードし、`NeuralAmpTorch/vst3sdk/` に配置します。
2. **LibTorch**:
   - [PyTorch公式サイト](https://pytorch.org/)から **LibTorch (C++/Java) ABI** をダウンロードします（CUDA版またはCPU版を選択）。
   - `NeuralAmpTorch/LibTorch/` に解凍・配置します。

#### ⚠️ バージョンに関する重要事項
本プロジェクトの学習環境（Python）は **CUDA 12.6** を使用しています。
C++側の LibTorch も、必ず以下の条件に一致するものをダウンロードしてください。

- **Version**: 2.x.x (Python側のPyTorchバージョンと一致させる)
- **Compute Platform**: **CUDA 12.6 (cu126)**
- **ABI**: Windowsの場合は通常 **Pre-cxx11 ABI** を使用（MSVCのバージョンに依存）

### 1．環境構築
```bash
# VisualStudioで（NeuralAmpTorchフォルダを開いて以下のコマンドを打ってください
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64
```

### 2．ビルド
Releaseでビルドをしてください．.vst3ファイルが生成されます．


### Third-party software and resources
- **HiFi-GAN Discriminator Implementation**: Based on [jik876/hifi-gan](https://github.com/jik876/hifi-gan). 
  - Copyright (c) 2020 Jungil Kong
  - Licensed under the MIT License.
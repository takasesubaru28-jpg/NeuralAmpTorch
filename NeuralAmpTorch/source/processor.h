#pragma once

#include "public.sdk/source/vst/vstaudioeffect.h"
#include "public.sdk/source/vst/vsteditcontroller.h"
#include <torch/script.h>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace Steinberg {
    namespace Vst {

        class MyVSTProcessor : public AudioEffect
        {
        public:
            MyVSTProcessor();
            virtual ~MyVSTProcessor();
            // VST3 標準メソッド
            tresult PLUGIN_API initialize(FUnknown* context) override;
            tresult PLUGIN_API setBusArrangements(SpeakerArrangement* inputs, int32 numIns,
                SpeakerArrangement* outputs, int32 numOuts) override;
            tresult PLUGIN_API canProcessSampleSize(int32 symbolicSampleSize) override;
            tresult PLUGIN_API process(ProcessData& data) override;
            tresult processBypass(Sample32* inL, Sample32* inR, Sample32* outL, Sample32* outR, int32 numSamples);

            // オーディオ処理のセットアップ（必要に応じて追加）
            tresult PLUGIN_API setupProcessing(ProcessSetup& setup) override;
            // クラス作成メソッド
            static FUnknown* createInstance(void* /*context*/) { return (IAudioProcessor*)new MyVSTProcessor(); }


        protected:
            float bypass = 0.0f;

            size_t buffer_size = 512;
            size_t input_size = 1024;
            int64_t param_dim = 4;

            float treble=0.5f;
            float middle = 0.5f;
            float bass = 0.5f;
            float volume=0.5f;
            float gain = 0.5f;
            bool gan=false;
            int use_model=0;
            bool changedModel = false;
            bool inferenceDone = false;
            std::string modelPath;
            std::string folder;

        private:
            torch::Tensor inputTensorRef; // 入力音声用
            torch::Tensor paramTensorRef;
            std::pair<std::vector<float>, std::vector<float>> latestTask; // 追加
            bool hasNewTask = false;


            void writeLog(const std::string& text);
            uint64_t requestCount = 0; // 送信したリクエスト数
            uint64_t processedCount = 0; // 完了した推論数

            // 推論ループ関数
            void inferenceLoop();

            // --- LibTorch 関連 ---
            torch::jit::script::Module model;

            // --- バッファ・推論制御 関連 ---
            size_t bufferPos;
            std::vector<float> inputBuffer;  
            std::vector<float> inputHistory; 
            std::vector<float> outputBuffer;
            std::vector<float> readBuffer; 
            std::vector<float> writeBuffer;
            std::vector<float> paramVec;

            // --- スレッド制御 関連 ---
            std::thread inferThread;
            std::mutex queueMutex;
            std::condition_variable queueCV;
            std::atomic<bool> running{ false };
            std::atomic<bool> firstInferenceDone{ false };

            // 推論タスクキュー (AudioData, Params)
            std::queue<std::pair<std::vector<float>, std::vector<float>>> outputQueue;

        };

    } // namespace Vst
} // namespace Steinberg
#include "processor.h"
#include "parameter.h"
#include "fuid.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/vstspeaker.h"
#include <torch/script.h>
#include <iostream>
#include <windows.h>
#include <ATen/Parallel.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace Steinberg {
    namespace Vst {
        bool modelLoaded = false;

        tresult PLUGIN_API MyVSTProcessor::initialize(FUnknown* context)
        {
            tresult result = AudioEffect::initialize(context);
            if (result == kResultTrue)
            {
                addAudioInput(STR16("AudioInput"), SpeakerArr::kStereo);
                addAudioOutput(STR16("AudioOutput"), SpeakerArr::kStereo);
                at::set_num_threads(1);

                char dllPath[MAX_PATH];
                GetModuleFileNameA((HINSTANCE)&__ImageBase, dllPath, MAX_PATH);
                std::string folder(dllPath);
                folder = folder.substr(0, folder.find_last_of("\\/"));
                this->folder = folder;
                std::string modelPath = folder + "\\LSTM_1024.pt";
                bufferPos = 0;
                paramVec.assign({ 0.5f, 0.5f, 0.5f, 0.5f });
                inputTensorRef = torch::zeros({ 1, static_cast<int64_t>(input_size), 1 }, torch::kFloat32);
                paramTensorRef = torch::zeros({ 1, param_dim }, torch::kFloat32);

                try {
                    this->model = torch::jit::load(modelPath);
                    this->model.eval();
                    model.eval();
                    modelLoaded = true;
                }
                catch (const c10::Error& e) {
                    model = torch::jit::script::Module();
                    std::cerr << "Warning: model not loaded\n" << e.what() << std::endl;
                    modelLoaded = false;
                    return kResultFalse;
                }

                running = true;
                inferThread = std::thread(&MyVSTProcessor::inferenceLoop, this);

                firstInferenceDone = false;
                return kResultTrue;
            }
            return result;
        }

        tresult PLUGIN_API MyVSTProcessor::setBusArrangements(
            SpeakerArrangement* inputs, int32 numIns,
            SpeakerArrangement* outputs, int32 numOuts)
        {
            if (numIns >= 1 && numOuts >= 1)
            {
                for (int i = 0; i < numIns; ++i)
                    if (inputs[i] != SpeakerArr::kStereo && inputs[i] != SpeakerArr::kMono)
                        return kResultFalse;
                for (int i = 0; i < numOuts; ++i)
                    if (outputs[i] != SpeakerArr::kStereo && outputs[i] != SpeakerArr::kMono)
                        return kResultFalse;

                return AudioEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
            }
            return kResultFalse;
        }


        tresult PLUGIN_API MyVSTProcessor::canProcessSampleSize(int32 symbolicSampleSize)
        {
            if (symbolicSampleSize == kSample32 || symbolicSampleSize == kSample64)
                return kResultTrue;
            return kResultFalse;
        }

        std::queue<std::pair<std::vector<float>, std::vector<float>>> outputQueue;
        std::pair<std::vector<float>, std::vector<float>> data;

        void MyVSTProcessor::inferenceLoop()
        {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
            while (running)
            {
                std::pair<std::vector<float>, std::vector<float>> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    queueCV.wait(lock, [this] { return hasNewTask || !running; });
                    if (!running) break;

                    task = std::move(latestTask);
                    hasNewTask = false;

                    if (changedModel) {
                        try {
                            auto newModel = torch::jit::load(modelPath);
                            newModel.eval();
                            this->model = newModel;
                            modelLoaded = true;
                        }
                        catch (...) { modelLoaded = false; }
                        changedModel = false;
                    }
                }

                try {
                    torch::NoGradGuard no_grad;

                    std::memcpy(inputTensorRef.data_ptr<float>(), task.first.data(), task.first.size() * sizeof(float));
                    std::memcpy(paramTensorRef.data_ptr<float>(), task.second.data(), task.second.size() * sizeof(float));
                    torch::Tensor outputTensor = model.forward({ inputTensorRef, paramTensorRef }).toTensor();
                    int64_t startIdx = static_cast<int64_t>(input_size - buffer_size);
                    auto latestOutput = outputTensor.slice(1, startIdx, static_cast<int64_t>(input_size)).contiguous();

                    {
                        std::memcpy(outputBuffer.data(), latestOutput.data_ptr<float>(), buffer_size * sizeof(float));
                        firstInferenceDone = true;
                        inferenceDone = true;
                    }
                }
                catch (const std::exception& e) {
                    OutputDebugStringA(e.what());
                }
            }
        }

        MyVSTProcessor::MyVSTProcessor()
        {
            setControllerClass(ControllerUID);
        }
        tresult PLUGIN_API MyVSTProcessor::process(ProcessData& data)
        {
            if (data.inputParameterChanges != NULL)
            {
                int32 paramChangeCount = data.inputParameterChanges->getParameterCount();

                for (int32 i = 0; i < paramChangeCount; i++)
                {
                    IParamValueQueue* queue = data.inputParameterChanges->getParameterData(i);

                    if (queue != NULL)
                    {
                        int32 tag = queue->getParameterId();
                        int32 valueChangeCount = queue->getPointCount();
                        ParamValue value;
                        int32 sampleOffset;

                        if (queue->getPoint(valueChangeCount - 1, sampleOffset, value) == kResultTrue)
                        {
                            switch (tag)
                            {
                            case TREBLE:
                                treble = (float)value;
                                paramVec[2] = treble; 
                                break;
                            case MIDDLE:
                                middle = (float)value;
                                paramVec[1] = middle;
                                break;
                            case BASS:
                                bass = (float)value;
                                paramVec[0] = bass;   
                                break;
                            case GAIN:
                                gain = (float)value;
                                paramVec[3] = value;
                                break;
                            case VOLUME:
                                    volume = value;
                                    break;
                            case MODEL:
                            {
                                use_model = static_cast<int>(value * 3.999f);

                                switch (use_model)
                                {
                                case 0:
                                {
                                    std::string fileName = (gan == 0) ? "LSTM_1024.pt" : "LSTM_GAN_1024.pt";
                                    modelPath = folder + "\\" + fileName; 

                                    changedModel = true; 
                                    break;
                                }
                                case 1:
                                {
                                    std::string fileName = (gan == 0) ? "LSTM_2lay_1024.pt" : "LSTM_2lay_GAN_1024.pt";
                                    modelPath = folder + "\\" + fileName;

                                    changedModel = true; 
                                    break;
                                }
                                case 2:
                                {
                                    std::string fileName = (gan == 0) ? "WaveNet_1024.pt" : "WaveNet_GAN_1024.pt";
                                    modelPath = folder + "\\" + fileName;

                                    changedModel = true; 
                                    break;
                                }
                                case 3:
                                {
                                    std::string fileName = (gan == 0) ? "WaveNet_LSTM_1024.pt" : "WaveNet_LSTM_GAN_1024.pt";
                                    modelPath = folder + "\\" + fileName;

                                    changedModel = true; 
                                    break;
                                }
                                }
                                break;
                            }
                            case GAN:
                            {
                                int old_gan = gan;
                                gan = static_cast<int>(value * 1.999f);

                                if (gan != old_gan) {
                                    std::string fileName = "";
                                    switch (use_model)
                                    {
                                    case 0:
                                        fileName = (gan == 0) ? "LSTM_1024.pt" : "LSTM_GAN_1024.pt";
                                        break;
                                    case 1:
                                        fileName = (gan == 0) ? "LSTM_2lay_1024.pt" : "LSTM_2lay_GAN_1024.pt";
                                        break;
                                    case 2:
                                        fileName = (gan == 0) ? "WaveNet_1024.pt" : "WaveNet_GAN_1024.pt";
                                        break;
                                    case 3:
                                        fileName = (gan == 0) ? "WaveNet_LSTM_1024.pt" : "WaveNet_LSTM_GAN_1024.pt";
                                        break;
                                    }
                                    this->modelPath = this->folder + "\\" + fileName;
                                    this->changedModel = true;
                                }
                                break;
                            }
                            case BYPASS_TAG:
                            {
                                bypass = (value > 0.5f);
                                break;
                            }
                            }
                        }
                    }
                }
            }

            if (data.numSamples == 0) return kResultOk;

            int32 numInChannels = data.inputs[0].numChannels;
            int32 numOutChannels = data.outputs[0].numChannels;
            Sample32* inL = data.inputs[0].channelBuffers32[0];
            Sample32* inR = (numInChannels > 1) ? data.inputs[0].channelBuffers32[1] : inL;
            Sample32* outL = data.outputs[0].channelBuffers32[0];
            Sample32* outR = (numOutChannels > 1) ? data.outputs[0].channelBuffers32[1] : nullptr;

            if (bypass == 1.0)
            {
                bufferPos = 0;
                firstInferenceDone = false; 
                std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
                std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0f);
                std::fill(inputHistory.begin(), inputHistory.end(), 0.0f);
                return processBypass(inL, inR, outL, outR, data.numSamples);
            }

            for (int32 i = 0; i < data.numSamples; ++i) {

                if (bufferPos == 0 && modelLoaded && firstInferenceDone) {
                    if (!inferenceDone) {
                        for (int32 sample = 0; sample < data.numSamples; ++sample) {
                            outL[sample] = inL[sample]; 
                            if (outR) outR[sample] = inR[sample];
                        }
                        inferenceDone = false;
                        return kResultTrue;
                    }
                }
                float inputSample = 0.0f;
                if (inR != nullptr) {
                    inputSample = (inL[i] + inR[i]) * 0.5f; 
                }
                else {
                    inputSample = inL[i];
                }

                inputBuffer[bufferPos] = inputSample;

                float outSample = 0.0f;
                if (modelLoaded && firstInferenceDone) {
                    outSample = outputBuffer[bufferPos];
                }
                else
                {
                    outSample = inputSample;
                }

                float finalVolumeOut = outSample * volume;
                outL[i] = finalVolumeOut;
                if (outR != nullptr) {
                    outR[i] = finalVolumeOut; 
                }

                bufferPos++;

                if (bufferPos >= buffer_size) {
                    bufferPos = 0;

                    std::memmove(inputHistory.data(), inputHistory.data() + buffer_size, (input_size - buffer_size) * sizeof(float));
                    std::memcpy(inputHistory.data() + (input_size - buffer_size), inputBuffer.data(), buffer_size * sizeof(float));

                    if (queueMutex.try_lock()) {
                        latestTask = { inputHistory, paramVec };
                        hasNewTask = true;
                        queueMutex.unlock();
                        queueCV.notify_one(); 
                    }


                }
            }
            return kResultOk;
        }

        tresult MyVSTProcessor::processBypass(Sample32* inL, Sample32* inR, Sample32* outL, Sample32* outR, int32 numSamples)
        {
            if (inL != outL && inL != nullptr && outL != nullptr)
            {
                std::memcpy(outL, inL, numSamples * sizeof(float));
            }

            if (outR != nullptr)
            {
                if (inR != nullptr && inR != inL)
                {
                    if (inR != outR) {
                        std::memcpy(outR, inR, numSamples * sizeof(float));
                    }
                }
                else
                {
                    std::memcpy(outR, outL, numSamples * sizeof(float));
                }
            }

            return kResultOk;
        }

        tresult PLUGIN_API MyVSTProcessor::setupProcessing(ProcessSetup& setup)
        {
            // DAWの現在の最大バッファサイズを取得
            int32 maxBlockSize = setup.maxSamplesPerBlock;
            int32 divisions = 1;
            while ((maxBlockSize + divisions - 1) / divisions >= input_size) {
                divisions++;
            }
            buffer_size = (maxBlockSize + divisions - 1) / divisions;
            inputBuffer.assign(buffer_size, 0.0f);
            outputBuffer.assign(buffer_size, 0.0f);
            inputHistory.assign(input_size, 0.0f);

            bufferPos = 0;

            return AudioEffect::setupProcessing(setup);
        }

        MyVSTProcessor::~MyVSTProcessor()
        {
            running = false;
            queueCV.notify_all();
            if (inferThread.joinable())
                inferThread.join();
        }
    }
}
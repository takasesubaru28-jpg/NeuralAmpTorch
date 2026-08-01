import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, lfilter
from tqdm import tqdm
from LSTM import LSTMAmplifier
from WaveNet import WaveNetAmplifire
from WaveNet_LSTM import WaveNetLSTMAmplifier

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate, n_mels, n_fft, hop_length):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            center=False
        )

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)

        s_pred = torch.log1p(self.mel_transform(y_pred))
        s_true = torch.log1p(self.mel_transform(y_true))

        return F.l1_loss(s_pred, s_true)
    

def stable_error_to_signal(y_pred, y_true, eps=1e-5):

    mse_map = torch.mean((y_true - y_pred)**2, dim=1) 

    target_energy = torch.mean(y_true**2, dim=1) + eps
    
    esr = torch.mean(mse_map / target_energy)
    
    return torch.clamp(esr, max=10.0)


def calculate_total_loss(y_pred, y_true, mel_criterion, w_mse, w_mel):

    loss_esr = stable_error_to_signal(y_pred, y_true)
    loss_mel = mel_criterion(y_pred, y_true)    
    combined_loss = (w_mse * loss_esr) + (w_mel * loss_mel)
    
    return combined_loss, loss_esr, loss_mel


def get_wav_files(directory):

    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

def get_latest_checkpoint(checkpoint_dir):

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoint_files:
        return None
    
    def extract_number(f):

        numbers = re.findall(r'\d+', os.path.basename(f))
        return int(numbers[-1]) if numbers else -1

    return max(checkpoint_files, key=extract_number)

def process_audio_overlap(model, input_tensor, param, input_size, warmup, device):
    num_samples = input_tensor.size(1)
    input_size
    warmup_size = warmup
    step_size = input_size - warmup_size
    predictions = []
    model.eval()
    
    with torch.no_grad():

        for i in range(0, num_samples, step_size):

            input_chunk = input_tensor[:, i : i + input_size, :]
            if input_chunk.size(1) < input_size:
                pad_end = torch.zeros((1, input_size - input_chunk.size(1), 1), device=device)
                input_chunk = torch.cat([input_chunk, pad_end], dim=1)
            
            chunk_pred = model(input_chunk, param)
            valid_output = chunk_pred[:, warmup_size:, :]            
            remaining = num_samples - i
            if remaining < step_size:

                valid_output = valid_output[:, :remaining, :]
            
            predictions.append(valid_output.cpu())

    return torch.cat(predictions, dim=1).squeeze().numpy()

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_hpf(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype(np.float32)

def process_recursive_hpf(input_root, output_root, fs=44100, cutoff=20):

    search_pattern = os.path.join(input_root, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    
    if not wav_files:
        print(f"No WAV files found in {input_root}")
        return

    print(f"Found {len(wav_files)} files. Processing...")
    
    sos = signal.butter(4, cutoff, 'hp', fs=fs, output='sos')

    for file_path in tqdm(wav_files):

        rel_path = os.path.relpath(file_path, input_root)
        save_path = os.path.join(output_root, rel_path)        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            sample_rate, data = wavfile.read(file_path)
            data = data.astype(np.float32)
            filtered_data = signal.sosfilt(sos, data).astype(np.float32)            
            wavfile.write(save_path, sample_rate, filtered_data)

        except Exception as e:
            
            print(f"Error processing {file_path}: {e}")

def save_model(model, CHECKPOINT_DIR, file_name):
    """
    ディスクから読み込まず、メモリ上のモデルを直接TorchScript化して保存する
    """
    device_cpu = torch.device('cpu')
    
    import copy
    model_for_script = copy.deepcopy(model).to(device_cpu).eval()
    scripted_model = torch.jit.script(model_for_script)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    scripted_model.save(f"{CHECKPOINT_DIR}/{file_name}")

def validate(model, device, val_input_path, val_target_dir, mel_criterion, warmup, input_size, sample_rate, w_mse, w_mel):
    model.eval()
    total_val_loss = 0
    count = 0
    
    # Loss計算のためにCPU版の criterion を用意
    mel_criterion_cpu = mel_criterion.cpu()

    for filename in os.listdir(val_target_dir):
        if filename.endswith(".wav"):
            tar_file = os.path.join(val_target_dir, filename)
            with torch.no_grad():
                # --- 入力データの準備 ---
                _, x_valid = wavfile.read(val_input_path)
                x_valid = x_valid.astype(np.float32).reshape(-1, 1)
                
                p_valid_str = os.path.splitext(filename)[0].split("_")[-1]
                p_valid = np.array([float(p) for p in p_valid_str.split("&")], dtype=np.float32) / 10
                
                _, y_valid = wavfile.read(tar_file)
                y_valid = y_valid.astype(np.float32).reshape(-1, 1)

                input_tensor = torch.from_numpy(x_valid).to(device).view(1, -1, 1)
                param_tensor = torch.from_numpy(p_valid).unsqueeze(0).to(device)

                # --- 推論実行 ---
                pad_front = torch.zeros((1, warmup, 1), device=device)
                padded_input = torch.cat([pad_front, input_tensor], dim=1)
                
                # full_prediction を取得
                full_prediction = process_audio_overlap(model, padded_input, param_tensor, warmup=warmup, input_size=input_size, device=device)

                # --- CPU転送と形状の正規化 (IndexError対策) ---
                # 1. NumPy なら Tensor に変換
                if isinstance(full_prediction, np.ndarray):
                    full_prediction = torch.from_numpy(full_prediction)
                
                # 2. CPUへ転送
                full_prediction = full_prediction.cpu()
                target_tensor = torch.from_numpy(y_valid).cpu() # (L, 1)

                # 3. 形状を (Batch=1, Length, Channel=1) に統一
                if full_prediction.dim() == 1: # (L,) の場合
                    full_prediction = full_prediction.view(1, -1, 1)
                elif full_prediction.dim() == 2: # (L, 1) の場合
                    full_prediction = full_prediction.unsqueeze(0)
                
                if target_tensor.dim() == 2: # (L, 1) の場合
                    target_tensor = target_tensor.unsqueeze(0)

                # --- サイズ合わせと計算 ---
                min_len = min(full_prediction.shape[1], target_tensor.shape[1])
                y_pred_cpu = full_prediction[:, :min_len, :]
                y_true_cpu = target_tensor[:, :min_len, :]

                v_loss, _, _ = calculate_total_loss(y_pred_cpu, y_true_cpu, mel_criterion_cpu, w_mse, w_mel)
                
                total_val_loss += v_loss.item()
                count += 1
    
    # 元のデバイスに戻す
    mel_criterion.to(device)
                
    return total_val_loss / count if count > 0 else float('inf')
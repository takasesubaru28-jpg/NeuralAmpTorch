import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from LSTM import LSTMAmplifier
from WaveNet import WaveNetAmplifire
from WaveNet_LSTM import WaveNetLSTMAmplifier
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import sys
from Discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, discriminator_loss, generator_loss
import itertools
from utils import *
from Dataset import GuitarAmpDataset

def train(config):
    
    Exp_name = config["experiment_name"]
    num_workers = config["num_workers"]
    use_gan = config["use_gan"]
    model_name = config["model_name"]
    paths = config["paths"]

    WaveNet_params = config["WaveNet"]
    LSTM_params = config["LSTM"]
    t_params = config["train_params"]
    a_params = config["audio_params"]
    w_params = config["loss_weights"]

    input_wav = paths["input_wav"]
    target_dir = paths["target_dir"]

    WaveNet_units = WaveNet_params["WaveNet_units"]
    kernel_size = WaveNet_params["kernel_size"]
    WaveNet_layers = WaveNet_params["WaveNet_layers"]

    LSTM_units = LSTM_params["LSTM_units"]
    LSTM_layers = LSTM_params["LSTM_layers"]

    input_size = t_params["input_size"]
    batch_size = t_params["batch_size"]
    epochs = t_params["epochs"]
    param_dim = t_params["param_dim"]
    lr = t_params["lr"]
    clip_grad = t_params["clip_grad"]

    sample_rate = a_params["sample_rate"]
    n_fft = a_params["n_fft"]
    hop_length = a_params["hop_length"]
    n_mels = a_params["n_mels"]

    w_mel = w_params["w_mel"]
    w_mse = w_params["w_mse"]
    w_gan = w_params["w_gan"]

    warmup = input_size // 4
    step_size = input_size - warmup

    writer = SummaryWriter(f'runs/{Exp_name}')
    ckpt_dir = f"checkpoints/{Exp_name}"
    min_delta_ratio = 0.00001
    early_stop_patience = 50

    writer.add_text('Hyperparameters', f"```json\n{json.dumps(config, indent=4)}\n```")

    process_recursive_hpf("./data", "./data_hpf", fs=sample_rate, cutoff=20)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("Available devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is NOT available. Using CPU.")

    if model_name == "LSTM":
        model = LSTMAmplifier(input_size=input_size, hidden_units=LSTM_units, param_dim=param_dim, num_layers=LSTM_layers).to(device)
    elif model_name == "WaveNet":
        model = WaveNetAmplifire(input_size=input_size, res_hidden_units=WaveNet_units, param_dim=param_dim, num_layers=WaveNet_layers, kernel_size=kernel_size).to(device)
    elif model_name == "WaveNet_LSTM":
        model = WaveNetLSTMAmplifier(input_size=input_size, res_hidden_units=WaveNet_units, kernel_size=kernel_size, num_wn_layers=WaveNet_layers, lstm_hidden_units=LSTM_layers, num_lstm_layers=LSTM_layers, param_dim=param_dim).to(device)
    else:
        print("存在しないモデル名です")
        return
    
    if use_gan == True:
        mpd = MultiPeriodDiscriminator().to(device)
        msd = MultiScaleDiscriminator().to(device)
        optim_d = torch.optim.Adam(itertools.chain(msd.parameters(), mpd.parameters()), lr=lr/2, betas=(0.5, 0.9))
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)

    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=early_stop_patience, threshold=min_delta_ratio)
    

    dataset = GuitarAmpDataset(input_wav, get_wav_files(target_dir), input_size=input_size, step_size=step_size, sample_rate=sample_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None)

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    os.makedirs(ckpt_dir, exist_ok=True)

    # --- チェックポイントから再開 ---
    latest_ckpt = None
    if os.listdir(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_epoch_") and f.endswith(".pt")]
        if ckpt_files:

            latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
            checkpoint = torch.load(ckpt_path, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'mpd_state_dict' in checkpoint:
                mpd.load_state_dict(checkpoint['mpd_state_dict'])
                print("MPD state restored.")
            
            if 'msd_state_dict' in checkpoint:
                msd.load_state_dict(checkpoint['msd_state_dict'])
                print("MSD state restored.")

            if 'optimizer_d_state_dict' in checkpoint:
                optim_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
                print("Discriminator optimizer restored.")
            
            start_epoch = checkpoint['epoch']
            print(f"Resuming from checkpoint {latest_ckpt}, starting at epoch {start_epoch}")
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    mel_criterion = MelSpectrogramLoss(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length).to(device)
    no_improve_epochs = 0
    best_loss = float('inf')

    print(f"StartTrain{Exp_name}")

    for epoch in range(start_epoch, epochs):
            
        model.train()

        epoch_loss = 0
        dataset.shuffle_files()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

        for i, (X_batch, y_batch, p_batch) in pbar:

            if X_batch.size(0) < batch_size:
                continue
            
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            p_batch = p_batch.to(device, non_blocking=True) 

            with torch.amp.autocast('cuda'):
                y_pred = model(X_batch, p_batch)
                y_p_w = y_pred[:, warmup:].transpose(1, 2)
                y_b_w = y_batch[:, warmup:].transpose(1, 2)

            if use_gan and i % 10 == 0:
                optim_d.zero_grad()

                with torch.amp.autocast('cuda'):                    

                    y_df_hat_r, y_df_hat_g, _, _ = mpd(y_b_w, y_p_w.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_b_w, y_p_w.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                    loss_disc_all = loss_disc_s + loss_disc_f

                scaler_d.scale(loss_disc_all).backward()
                scaler_d.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(itertools.chain(msd.parameters(), mpd.parameters()), max_norm=1.0)
                scaler_d.step(optim_d)
                scaler_d.update()


            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                
                loss, esr_val, mel_val = calculate_total_loss(y_pred[:, warmup:], y_batch[:, warmup:], mel_criterion, w_mse, w_mel)


                if use_gan:
                    
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y_b_w, y_p_w)
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y_b_w, y_p_w)
                    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                    gan_val = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f
                    gan_val = torch.clamp(gan_val, max=10.0)
                    loss += gan_val*w_gan

            scaler_g.scale(loss).backward()
            scaler_g.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)        
            scaler_g.step(optimizer)
            scaler_g.update()

            epoch_loss += loss.item()

            if i % 1000 == 0:
                stats = {
                    "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "AvgLoss": f"{epoch_loss/(i+1):.4f}",
                    "Loss": f"{loss.item():.4f}",
                    "ESR": f"{esr_val.item():.4f}",
                    "Mel": f"{mel_val.item():.4f}"
                }
                
                if use_gan:
                    stats["Gan"] = f"{gan_val.item():.4f}"
                    stats["Disc"] = f"{loss_disc_all.item():.4f}"
                
                pbar.set_postfix(stats)

                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Total', loss.item(), global_step)
                writer.add_scalar('Loss/ESR', esr_val.item(), global_step)
                writer.add_scalar('Loss/Mel', mel_val.item(), global_step)
                
                if use_gan:
                    writer.add_scalar('Loss/Gan', gan_val.item(), global_step)
                    writer.add_scalar('Disc/Total', loss_disc_all.item(), global_step)
                    writer.add_scalar('Disc/MPD', loss_disc_f.item(), global_step)
                    writer.add_scalar('Disc/MSD', loss_disc_s.item(), global_step)
                                    

        if (epoch + 1) % 10 == 0:
                model.eval()

                val_input = "./data/valid/input/input.wav"
                val_target = "./data/valid/target"

                for filename in os.listdir(val_target):

                    if filename.endswith(".wav"):

                        tar_file = os.path.join(val_target, filename)

                        with torch.no_grad():

                            _, x_valid = wavfile.read(val_input)
                            x_valid = x_valid.astype(np.float32)
                            x_valid = x_valid.reshape(-1, 1)

                            p_valid_str = os.path.splitext(filename)[0].split("_")[-1]
                            p_valid = np.array([float(p) for p in p_valid_str.split("&")], dtype=np.float32) / 10

                            _, y_valid = wavfile.read(tar_file)
                            y_valid = y_valid.astype(np.float32)
                            y_valid = y_valid.reshape(-1, 1)
                            
                            input_tensor = torch.from_numpy(x_valid).to(device).view(1, -1, 1)
                            param_tensor = torch.from_numpy(p_valid).unsqueeze(0).to(device)
                            
                            pad_front = torch.zeros((1, warmup, 1), device=device)
                            padded_input = torch.cat([pad_front, input_tensor], dim=1)
                            
                            full_prediction = process_audio_overlap(model, padded_input, param_tensor, warmup=warmup, input_size=input_size, device=device)
                           
                            writer.add_audio(f"Audio/{p_valid_str}_Target", torch.from_numpy(y_valid).squeeze(), epoch, sample_rate=sample_rate)                    
                            writer.add_audio(f"Audio/{p_valid_str}_Prediction", full_prediction.squeeze(), epoch, sample_rate=sample_rate)

                            pred_wave = full_prediction.squeeze()
                            target_wave = y_valid.flatten()

                            fig = plt.figure(figsize=(12, 4))
                            plt.plot(target_wave, label=f'{p_valid_str}_Target', alpha=0.5)
                            plt.plot(pred_wave, label=f'{p_valid_str}_Prediction', alpha=0.5)
                            plt.legend()
                            writer.add_figure(f'Visual/{p_valid_str}_Waveform', fig, global_step=epoch)
                            plt.close(fig)

                ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1:04d}.pt")

                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }

                if use_gan:
                    save_dict.update({
                        'mpd_state_dict': mpd.state_dict(),
                        'msd_state_dict': msd.state_dict(),
                        'optimizer_d_state_dict': optim_d.state_dict(),
                    })

                torch.save(save_dict, ckpt_path)

                model.train()

        # 1. 検証用データでLossを計算
        val_input = "./data/valid/input/input.wav"
        val_target = "./data/valid/target"
        
        val_loss_avg = validate(model, device, val_input, val_target, mel_criterion, warmup, input_size, sample_rate, w_mse, w_mel)
        
        # 2. TensorBoardに検証Lossを記録
        writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
        writer.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # 3. スケジューラーを検証Lossに基づいて更新
        scheduler.step(val_loss_avg) 

        if use_gan:
            scheduler_d.step()

        improvement_threshold = best_loss * (1 - min_delta_ratio)

        if val_loss_avg < improvement_threshold:
            # 改善したとみなす条件（前回のベストより10%以上低下）
            best_loss = val_loss_avg
            no_improve_epochs = 0

        else:
            # 改善が不十分、または悪化した場合
            no_improve_epochs += 1

        model.eval()
        save_model(model=model, CHECKPOINT_DIR=ckpt_dir, file_name=f"{Exp_name}.pt")
        model.train()

        if no_improve_epochs >= early_stop_patience*2:
            print(f"Early stopping triggered. No {min_delta_ratio:.1%} improvement for {early_stop_patience} epochs.")
            break

    print("Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Amplifier Training")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    train(config)

if __name__ == "__main__":
    main()
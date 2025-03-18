import torch
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from model import VoiceConverter
from torch.utils.data import DataLoader, TensorDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    SAMPLE_RATE = 16000
    HOP_LENGTH = 256
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def load_and_process_audio(path, sr=SAMPLE_RATE, augment=False):
        y, sr = librosa.load(path, sr=sr)
        if augment:
            y += 0.01 * np.random.randn(len(y))  # Mehr Rauschen
            if np.random.rand() > 0.3:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-5, 5))
            if np.random.rand() > 0.3:
                y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.7, 1.3))
        
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, 
            n_mfcc=13, 
            n_fft=2048, 
            hop_length=HOP_LENGTH,
            n_mels=128
        ).T
        delta = librosa.feature.delta(mfcc.T, width=5).T
        ddelta = librosa.feature.delta(mfcc.T, order=2, width=5).T
        return np.concatenate([mfcc, delta, ddelta], axis=1)

    # Load target voice
    print("Loading target voice...")
    mfcc_target = load_and_process_audio("dataset/raw/murad.wav")

    # Generate synthetic sources
    print("Generating synthetic sources...")
    source_voices = []
    y, sr = librosa.load("dataset/raw/murad.wav", sr=SAMPLE_RATE)
    chunk_size = SAMPLE_RATE * 5

    def process_chunk(chunk):
        if len(chunk) < 100:
            return None
        chunk = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=np.random.choice([-3, -2, 0, 2, 3]))
        chunk = librosa.effects.time_stretch(chunk, rate=np.random.uniform(0.8, 1.2))
        return load_and_process_audio_from_array(chunk)

    def load_and_process_audio_from_array(y):
        if len(y) == 0:
            return None
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SAMPLE_RATE,
            n_mfcc=13,
            n_fft=2048,
            hop_length=256,
            n_mels=128
        ).T
        if mfcc.shape[0] == 0:
            return None
        delta = librosa.feature.delta(mfcc.T, width=5).T
        ddelta = librosa.feature.delta(mfcc.T, order=2, width=5).T
        return np.concatenate([mfcc, delta, ddelta], axis=1)

    for i in range(0, len(y)-chunk_size, chunk_size//2):
        chunk = y[i:i+chunk_size]
        mfcc = process_chunk(chunk)
        if mfcc is not None and mfcc.shape[0] > 10:
            source_voices.append(mfcc)
            print(f"Generated chunk {i//sr}s-{(i+chunk_size)//sr}s: {mfcc.shape} frames")

    # Normalization
    source_stack = np.vstack(source_voices)
    mfcc_mean_source = source_stack.mean(axis=0)
    mfcc_max_source = np.max(np.abs(source_stack))

    mfcc_mean_target = mfcc_target.mean(axis=0)
    mfcc_max_target = np.max(np.abs(mfcc_target))

    def normalize(voice, mean, max_val):
        return (voice - mean) / (max_val + 1e-8)

    mfcc_target_norm = normalize(mfcc_target, mfcc_mean_target, mfcc_max_target)
    source_voices_norm = [normalize(v, mfcc_mean_source, mfcc_max_source) for v in source_voices]

    # Verwende den Mittelwert statt der Rohdaten
    target_stats = torch.tensor(mfcc_target_norm.mean(axis=0), dtype=torch.float32).to(device)

    # Model setup
    model = VoiceConverter(
        input_dim=39,
        target_dim=39,  # Muss mit der Ausgabedimension übereinstimmen
        hidden_dim=512,
        num_layers=4,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # Höhere LR
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(  # Neuer Scheduler
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    # Data preparation
    def create_segments(voice, segment_length=40):
        segments = []
        for i in range(0, len(voice) - segment_length + 1, segment_length // 2):
            segment = voice[i:i+segment_length]
            if len(segment) == segment_length:
                padded = np.pad(segment, ((5,5),(0,0)), mode='reflect')
                shift = np.random.randint(-5, 5)
                shifted = padded[5+shift:5+shift+segment_length]
                if shifted.shape[0] == segment_length:
                    segments.append(shifted + 0.03*np.random.randn(*shifted.shape))
        return np.array(segments)

    input_data = []
    for voice in source_voices_norm:
        input_data.extend(create_segments(voice))
    input_data = np.array(input_data)
    np.random.shuffle(input_data)

    # Data loaders
    val_size = int(0.15 * len(input_data))
    train_data = torch.tensor(input_data[:-val_size], dtype=torch.float32)
    val_data = torch.tensor(input_data[-val_size:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=128,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        TensorDataset(val_data),
        batch_size=128,
        shuffle=False
    )

    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/200 [Train]")
        
        for batch in pbar:
            batch = batch[0].to(device)
            
            # 🚀 Batch-spezifische Statistik erstellen
            batch_target_stats = target_stats.unsqueeze(0).expand(batch.size(0), -1)
            
            optimizer.zero_grad()
            with autocast():
                recon, mu, logvar = model(batch, batch_target_stats)
                
                loss_mse = F.mse_loss(recon, batch)
                loss_mae = F.l1_loss(recon, batch)
                
                # STFT-basierter Loss
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kld_weight = min(0.1 * (epoch / 20), 0.1)
                
                # NEUE LOSS-BERECHNUNG
                total_loss = 0.6 * loss_mse + 0.4 * loss_mae + kld_weight * kld
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                batch_target_stats = target_stats.unsqueeze(0).expand(batch.size(0), -1)
                recon, _, _ = model(batch, batch_target_stats)
                loss = 0.4*F.mse_loss(recon, batch) + 0.4*F.l1_loss(recon, batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= 10:  # Früher Stopp
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

    print("Training complete!")
    model.load_state_dict(torch.load("outputs/best_model.pth"))
    torch.save(model.state_dict(), "outputs/final_model.pth")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

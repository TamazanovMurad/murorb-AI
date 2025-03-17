import librosa
import numpy as np
import os

def extract_mfcc(file_path):
    y, _ = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13).astype(np.float32).T
    return mfcc

os.makedirs("dataset/processed/mfcc", exist_ok=True)

# Verarbeite Zielstimme
partner_mfcc = extract_mfcc("dataset/raw/partner.wav")
np.save("dataset/processed/mfcc/partner.npy", partner_mfcc)

# Verarbeite Trainingdata (beliebige Audiodateien)
for file in os.listdir("dataset/train"):
    if file.endswith(".wav"):
        mfcc_data = extract_mfcc(f"dataset/train/{file}")
        np.save(f"dataset/processed/mfcc/{file}.npy", mfcc_data)
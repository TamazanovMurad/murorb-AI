import torch
import numpy as np
import librosa

from model import VoiceConversionModel  # Modul aus deinem Projekt

# Lade das trainierte Modell
model = VoiceConversionModel()
model.load_state_dict(torch.load("outputs/trained_model.pth"))
model.eval()  # Setze das Modell in Evaluierungsmodus

def convert_audio(input_path, output_path):
    # Lade Audio
    y, _ = librosa.load(input_path, sr=48000)
    
    # Extrahiere MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=48000, n_mfcc=13).astype(np.float32).T
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Füge Batch-Dimension hinzu
    
    # Vorhersage
    with torch.no_grad():
        converted_mfcc = model(mfcc_tensor).squeeze().numpy()
    
    # Speichere das konvertierte Audio
    librosa.output.write_wav(output_path, converted_mfcc, sr=48000)

# Beispielaufruf
convert_audio("murad", "converted_audio.wav")
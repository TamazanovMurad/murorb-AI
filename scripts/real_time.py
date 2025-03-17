# Voice Converter - Real-Time Processing
# ======================================

import torch
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread
import time
import os
import traceback

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create necessary directories
os.makedirs("dataset/raw", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("converted_audio", exist_ok=True)

# Define model class
class VoiceConverter(nn.Module):
    def __init__(self, target_mfcc):
        super().__init__()
        # Store target voice characteristics
        self.register_buffer('target_mfcc', target_mfcc if isinstance(target_mfcc, torch.Tensor) 
                              else torch.tensor(target_mfcc, dtype=torch.float32))
        
        # Simpler encoder matching the saved model architecture
        self.encoder = nn.Sequential(
            nn.Linear(13, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128*2)
        )
        
        # Simpler decoder matching the saved model
        self.decoder = nn.Sequential(
            nn.Linear(128 + 13, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 13)
        )
        
        # Voice style controller
        self.style_controller = nn.Sequential(
            nn.Linear(13, 128),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        original_shape = x.size()
        if len(original_shape) == 3:
            # If input is [batch, seq_len, features]
            batch_size, seq_len, features = original_shape
            # Reshape to [batch*seq_len, features]
            x = x.reshape(-1, features)
            
        # Encoder
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=1)
        
        # Sampling with controlled variance
        z = self.reparameterize(mu, logvar)
        
        # Process target voice characteristics through style controller
        target = self.target_mfcc.expand(z.size(0), -1)
        style_vector = self.style_controller(target)
        
        # Combine content (z) with style for better voice transfer
        # Using element-wise multiplication for style transfer
        z_styled = z * style_vector
        
        # Concatenate with target for additional guidance
        decoder_input = torch.cat([z_styled, target], dim=1)
        
        # Decoder
        output = self.decoder(decoder_input)
        
        # Reshape output back to original shape if needed
        if len(original_shape) == 3:
            output = output.reshape(batch_size, seq_len, -1)
            mu = mu.reshape(batch_size, seq_len, -1)
            logvar = logvar.reshape(batch_size, seq_len, -1)
            
        return output, mu, logvar

class AudioProcessor:
    def __init__(self, model_path, params_path, target_voice_path=None):
        # Audio settings - adjusted for better real-time performance
        self.sample_rate = 16000
        self.hop_length = 256  # Reduced for lower latency
        # Adjusted buffer size for more stable processing
        self.buffer_size = 4096  # Keeping larger buffer to reduce artifacts
        self.overlap = 512      # Reduced for lower latency
        
        # Initialize audio buffers and queues
        self.input_buffer = np.zeros(self.buffer_size)
        self.output_buffer = Queue(maxsize=30)  # Increased queue size for stability
        self.processing_active = False
        self.processing_thread = None
        
        # Noise gate threshold - adjusted to reduce background noise
        self.noise_gate_threshold = 0.005
        
        # Previous output frame for crossfading
        self.prev_output_frame = np.zeros(self.overlap)
        
        # Load normalization parameters
        self.load_parameters(params_path)
        
        # Load target voice if provided, otherwise use murad.wav
        if target_voice_path is None:
            target_voice_path = "dataset/raw/murad.wav"
            
        if not os.path.exists(target_voice_path):
            raise FileNotFoundError(f"Target voice file not found at {target_voice_path}")
            
        self.load_target_voice(target_voice_path)
        
        # Load model
        self.load_model(model_path)
        
        # Current audio being processed for file conversion
        self.current_audio = None
        self.current_mfcc = None
        self.current_converted = None

        # Audio stream objects
        self.input_stream = None
        self.output_stream = None
        
        # Audio frame buffer
        self.frame_buffer = []
        self.buffer_lock = False
        
        # Selected devices for audio I/O
        self.input_device = None
        self.output_device = None
        
        # Window function for smoother transitions
        self.window = np.hanning(self.overlap * 2)
        
    def load_parameters(self, params_path):
        try:
            params = np.load(params_path, allow_pickle=True).item()
            self.mfcc_mean = params["mean"]
            self.mfcc_max = params["max"]
            print(f"Loaded normalization parameters: max={self.mfcc_max}, mean shape={self.mfcc_mean.shape}")
        except Exception as e:
            print(f"Error loading parameters: {e}")
            traceback.print_exc()
            raise
            
    def load_target_voice(self, target_path):
        try:
            y_target, _ = librosa.load(target_path, sr=self.sample_rate)
            
            # Apply preemphasis to target before extracting MFCC
            y_target = librosa.effects.preemphasis(y_target)
            
            mfcc_target = librosa.feature.mfcc(
                y=y_target,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,  # Increased for better voice characteristics capture
                hop_length=self.hop_length
            ).T
            
            # Normalize target mfcc
            mfcc_target_norm = (mfcc_target - self.mfcc_mean) / self.mfcc_max
            
            # Take more consistent center frames for target characteristics
            middle_idx = len(mfcc_target_norm) // 2
            window_size = min(50, len(mfcc_target_norm) // 2)
            representative_frames = mfcc_target_norm[middle_idx-window_size:middle_idx+window_size]
            
            # Use mean of representative frames
            self.target_tensor = torch.tensor(
                representative_frames.mean(axis=0), 
                dtype=torch.float32
            ).to(device)
            
            print(f"Loaded target voice from {target_path}")
            
        except Exception as e:
            print(f"Error loading target voice: {e}")
            traceback.print_exc()
            raise
            
    def load_model(self, model_path):
        try:
            self.model = VoiceConverter(self.target_tensor).to(device)
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if this is a full checkpoint or just a state dict
            if 'model_state_dict' in checkpoint:
                # It's a full checkpoint - extract just the model state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from checkpoint at {model_path}")
            else:
                # Try loading directly
                self.model.load_state_dict(checkpoint)
                print(f"Model loaded directly from {model_path}")
                
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    def visualize_audio(self, audio, title="Audio", show=True):
        """Visualize audio waveform"""
        plt.figure(figsize=(10, 3))
        plt.plot(audio)
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        if show:
            plt.show(block=False)
            plt.pause(0.5)
            
    def convert_file(self, input_path, output_path=None):
        """Convert an audio file and save the result"""
        print(f"Converting file: {input_path}")
        
        try:
            # Load input audio
            y_input, _ = librosa.load(input_path, sr=self.sample_rate)
            
            # Check if audio has content
            if np.max(np.abs(y_input)) < 0.01:
                print("Warning: Input audio has very low volume. Results may be poor.")
            
            self.current_audio = y_input
            
            # Apply preemphasis to input before extracting MFCC - this helps with high frequencies
            y_input_preemph = librosa.effects.preemphasis(y_input)
            
            # Extract MFCCs from input
            mfccs = librosa.feature.mfcc(
                y=y_input_preemph,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=self.hop_length
            ).T
            
            # Store the original mfcc
            self.current_mfcc = mfccs
            
            # Normalize MFCCs
            mfccs_norm = (mfccs - self.mfcc_mean) / self.mfcc_max
            
            # Convert to tensor
            mfccs_tensor = torch.tensor(mfccs_norm, dtype=torch.float32).to(device)
            
            # Process in batches to avoid memory issues
            with torch.no_grad():
                batch_size = 64
                converted_mfccs = []
                
                for i in range(0, len(mfccs_tensor), batch_size):
                    batch = mfccs_tensor[i:i+batch_size]
                    converted, _, _ = self.model(batch)
                    converted_mfccs.append(converted.cpu().numpy())
                    
            # Combine batches
            converted_mfccs = np.vstack(converted_mfccs)
            
            # Denormalize
            converted_mfccs = converted_mfccs * self.mfcc_max + self.mfcc_mean
            
            # Store the converted mfcc
            self.current_converted = converted_mfccs
            
            # IMPROVED AUDIO RECONSTRUCTION
            # ==============================
            
            # Step 1: Convert from MFCC back to mel spectrograms with better params
            n_mels = 128
            
            # Convert MFCC back to mel using librosa's inverse transform
            mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(
                mfcc=converted_mfccs.T,  # Transpose to match librosa's expected shape
                n_mels=n_mels,
                dct_type=2,
                norm='ortho'
            )
            
            # Ensure non-negative values for the mel spectrogram (needed for Griffin-Lim)
            mel_spectrogram = np.maximum(mel_spectrogram, 1e-10)
            
            # Step 2: Apply Griffin-Lim with more iterations for better phase reconstruction
            audio_reconstructed = librosa.feature.inverse.mel_to_audio(
                mel_spectrogram,
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=self.hop_length,
                n_iter=100  # More iterations for better quality
            )
            
            # Apply a light de-emphasis filter to reduce high frequency noise
            audio_reconstructed = librosa.effects.preemphasis(audio_reconstructed, coef=-0.95)
            
            # Apply light normalization to prevent loudness issues
            if np.max(np.abs(audio_reconstructed)) > 0:
                audio_reconstructed = audio_reconstructed / np.max(np.abs(audio_reconstructed)) * 0.9
            
            # Visualize both input and output
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(y_input)
            plt.title("Input Audio")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            
            plt.subplot(2, 1, 2)
            plt.plot(audio_reconstructed)
            plt.title("Converted Audio")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)
            
            # Play the converted audio
            print("Playing converted audio...")
            sd.play(audio_reconstructed, self.sample_rate)
            sd.wait()
            
            # Save output if specified
            if output_path is None:
                timestamp = int(time.time())
                output_path = f"converted_audio/converted_{timestamp}.wav"
                
            sf.write(output_path, audio_reconstructed, self.sample_rate)
            print(f"Converted audio saved to {output_path}")
            
            return audio_reconstructed
            
        except Exception as e:
            print(f"Error during file conversion: {e}")
            traceback.print_exc()
            return None

    def set_devices(self, input_device=None, output_device=None):
        """Set input and output devices"""
        self.input_device = input_device
        self.output_device = output_device
        print(f"Audio devices set: input={input_device}, output={output_device}")

    def process_audio_thread(self):
        """Background thread for processing audio frames"""
        print("Audio processing thread started")
        
        while self.processing_active:
            # Only process if we have frames and processing is not locked
            if len(self.frame_buffer) > 0 and not self.buffer_lock:
                try:
                    # Mark as locked to prevent concurrent access
                    self.buffer_lock = True
                    
                    # Get a frame to process
                    frame = self.frame_buffer.pop(0)
                    
                    # Release lock
                    self.buffer_lock = False
                    
                    # Process the frame
                    self._process_single_frame(frame)
                    
                except Exception as e:
                    print(f"Error in audio processing thread: {e}")
                    traceback.print_exc()
                    self.buffer_lock = False
            else:
                # Sleep a bit to reduce CPU usage when idle
                time.sleep(0.01)
    
    def _process_single_frame(self, audio_chunk):
        """Process a single audio frame"""
        try:
            # Check if the audio is above the noise gate
            audio_level = np.max(np.abs(audio_chunk))
            is_silence = audio_level < self.noise_gate_threshold
            
            # If silent, send a very small amount of noise instead of complete silence
            # This helps prevent audio driver issues with complete silence
            if is_silence:
                # Generate very low-level noise (better than complete silence for audio drivers)
                output_chunk = np.random.randn(self.overlap) * 0.0005
                output_chunk = output_chunk.reshape(-1, 1)
                
                if self.output_buffer.qsize() < self.output_buffer.maxsize:
                    self.output_buffer.put(output_chunk)
                return
                
            # Add new audio to buffer with overlap
            self.input_buffer = np.roll(self.input_buffer, -len(audio_chunk))
            self.input_buffer[-len(audio_chunk):] = audio_chunk
            
            # Apply preemphasis to enhance high frequencies
            input_preemph = librosa.effects.preemphasis(self.input_buffer)
            
            # Extract features
            mfcc = librosa.feature.mfcc(
                y=input_preemph,
                sr=self.sample_rate,
                n_mfcc=13,
                n_fft=1024,  # Smaller FFT for lower latency
                hop_length=self.hop_length
            ).T
            
            # Skip if we don't have enough frames
            if len(mfcc) < 2:
                return
                
            # Normalize
            mfcc_norm = (mfcc - self.mfcc_mean) / self.mfcc_max
            
            # Convert to tensor
            mfcc_tensor = torch.tensor(mfcc_norm, dtype=torch.float32).to(device)
            
            # Process with model
            with torch.no_grad():
                converted, _, _ = self.model(mfcc_tensor)
                
            # Convert back to numpy
            converted = converted.cpu().numpy()
            
            # Denormalize
            converted = converted * self.mfcc_max + self.mfcc_mean
            
            # IMPROVED RECONSTRUCTION FOR REAL-TIME
            # ====================================
            n_mels = 80  # Increased for better quality
            
            # Convert MFCC back to mel using librosa's inverse transform
            mel_spectrogram = librosa.feature.inverse.mfcc_to_mel(
                mfcc=converted.T,  # Transpose to [n_mfcc, time]
                n_mels=n_mels,
                dct_type=2,
                norm='ortho'
            )
            
            # Ensure positive values
            mel_spectrogram = np.maximum(mel_spectrogram, 1e-10)
            
            # Griffin-Lim for audio reconstruction
            audio_converted = librosa.feature.inverse.mel_to_audio(
                mel_spectrogram,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=self.hop_length,
                n_iter=32  # Balanced for quality vs latency
            )
            
            # De-emphasis to reduce high frequency noise
            audio_converted = librosa.effects.preemphasis(audio_converted, coef=-0.97)
            
            # Normalize audio output
            if np.max(np.abs(audio_converted)) > 0:
                audio_converted = audio_converted / np.max(np.abs(audio_converted)) * 0.9
            
            # Add to output buffer if we have sufficient data
            if len(audio_converted) >= self.overlap:
                # Apply crossfade with previous frame to reduce clicking/popping
                new_frame = audio_converted[:self.overlap]
                
                # Apply crossfade window
                fade_in = self.window[:self.overlap]
                fade_out = self.window[self.overlap:]
                
                # Apply crossfade between frames
                crossfaded = (new_frame * fade_in) + (self.prev_output_frame * fade_out)
                
                # Store current frame for next crossfade
                self.prev_output_frame = new_frame
                
                # Reshape for output
                output_chunk = crossfaded.reshape(-1, 1)
                
                # Add to output buffer if not full
                if self.output_buffer.qsize() < self.output_buffer.maxsize:
                    self.output_buffer.put(output_chunk)
                    
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            traceback.print_exc()
            # Create low-level noise output in case of error
            output_chunk = np.random.randn(self.overlap) * 0.0005
            output_chunk = output_chunk.reshape(-1, 1)
            
            if self.output_buffer.qsize() < self.output_buffer.maxsize:
                self.output_buffer.put(output_chunk)

    def start_realtime_processing(self):
        """Start real-time audio processing"""
        if self.processing_active:
            print("Real-time processing already active")
            return
            
        self.processing_active = True
        
        # Clear any existing audio in the output buffer
        while not self.output_buffer.empty():
            self.output_buffer.get()
        
        # Clear the frame buffer
        self.frame_buffer = []
        
        # Reset input buffer
        self.input_buffer = np.zeros(self.buffer_size)
        
        # Reset previous output frame
        self.prev_output_frame = np.zeros(self.overlap)
        
        # Start the processing thread
        self.processing_thread = Thread(target=self.process_audio_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Pre-fill output buffer with low-level noise to prevent initial underruns
        for _ in range(10):
            noise = np.random.randn(self.overlap) * 0.0005
            self.output_buffer.put(noise.reshape(-1, 1))
        
        # Setup audio input stream with adjusted parameters
        def audio_callback(indata, frames, time, status):
            """Callback for audio input stream"""
            if status and status != sd.CallbackFlags.input_overflow:
                print(f"Input status: {status}")
                
            # Get audio data
            audio_chunk = indata[:, 0]  # Use first channel if stereo
            
            # Add to frame buffer if not locked
            if not self.buffer_lock and self.processing_active:
                self.frame_buffer.append(audio_chunk)
                # Limit buffer size to prevent memory issues
                if len(self.frame_buffer) > 20:
                    self.frame_buffer = self.frame_buffer[-20:]
        
        # Setup audio output stream with adjusted parameters
        def output_callback(outdata, frames, time, status):
            """Callback for audio output stream"""
            if status and status != sd.CallbackFlags.output_underflow:
                print(f"Output status: {status}")
                
            try:
                # Get processed audio from queue or use very low noise if none available
                if not self.output_buffer.empty():
                    outdata[:] = self.output_buffer.get()
                else:
                    # Use very low noise instead of silence
                    noise = np.random.randn(frames, 1) * 0.0005
                    outdata[:] = noise
                    if self.processing_active:
                        print("Output buffer underrun - filling with low-level noise")
            except Exception as e:
                print(f"Error in output callback: {e}")
                # Use very low noise in case of error
                noise = np.random.randn(frames, 1) * 0.0005
                outdata[:] = noise
        
        try:
            # Configure lower latency but still stable
            latency = 'medium'
            
            # Start audio streams with adjusted parameters
            self.input_stream = sd.InputStream(
                callback=audio_callback, 
                channels=1, 
                samplerate=self.sample_rate,
                blocksize=self.overlap,
                device=self.input_device,
                latency=latency
            )
            
            self.output_stream = sd.OutputStream(
                callback=output_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.overlap,
                device=self.output_device,
                latency=latency
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            print("Real-time voice conversion started. Press Ctrl+C to stop.")
            
            try:
                while self.processing_active:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopping real-time processing...")
                self.processing_active = False
            finally:
                self.stop_realtime_processing()
                
        except Exception as e:
            print(f"Error starting audio streams: {e}")
            self.processing_active = False
            traceback.print_exc()
    
    def stop_realtime_processing(self):
        """Stop real-time audio processing"""
        self.processing_active = False
        
        # Give processing thread time to finish
        if self.processing_thread and self.processing_thread.is_alive():
            time.sleep(0.5)
        
        # Close audio streams if they exist
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            
        print("Real-time processing stopped")

def list_available_devices():
    """List all available audio devices"""
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    
    print("Input devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']}")
    
    print("\nOutput devices:")
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"  {i}: {device['name']}")
    
    print("\nDefault devices:")
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    print(f"  Default input: {default_input['name']}")
    print(f"  Default output: {default_output['name']}")

def select_audio_device():
    """Let user select input and output devices"""
    list_available_devices()
    
    devices = sd.query_devices()
    
    try:
        # Input device selection
        input_choice = input("\nSelect input device number (leave blank for default): ")
        if input_choice.strip():
            input_device = int(input_choice)
            if input_device < 0 or input_device >= len(devices) or devices[input_device]['max_input_channels'] == 0:
                print("Invalid input device selected. Using default.")
                input_device = None
        else:
            input_device = None
            
        # Output device selection
        output_choice = input("Select output device number (leave blank for default): ")
        if output_choice.strip():
            output_device = int(output_choice)
            if output_device < 0 or output_device >= len(devices) or devices[output_device]['max_output_channels'] == 0:
                print("Invalid output device selected. Using default.")
                output_device = None
        else:
            output_device = None
            
        return input_device, output_device
        
    except ValueError:
        print("Invalid input. Using default devices.")
        return None, None

def main():
    print("Voice Converter - Real-Time Processing")
    print("======================================")
    
    # Check for model and parameter files
    model_path = "outputs/best_model.pth"
    if not os.path.exists(model_path):
        # Try alternative model files
        model_found = False
        for epoch in [50, 30, 20, 10, 5]:
            alt_path = f"outputs/model_epoch{epoch}.pth"
            if os.path.exists(alt_path):
                model_path = alt_path
                model_found = True
                print(f"Using model: {model_path}")
                break
        
        if not model_found:
            print("No model file found! Please train a model first.")
            return
    
    params_path = "outputs/mfcc_params.npy"
    if not os.path.exists(params_path):
        print("MFCC parameters not found! Please train a model first.")
        return
    
    target_path = "dataset/raw/murad.wav"
    if not os.path.exists(target_path):
        print("Target voice file not found! Please place your target voice in dataset/raw/murad.wav")
        input("Press Enter to continue anyway (will use a default voice model)...")
    
    # Initialize processor
    try:
        processor = AudioProcessor(model_path, params_path, target_path if os.path.exists(target_path) else None)
    except Exception as e:
        print(f"Error initializing audio processor: {e}")
        traceback.print_exc()
        return
    
    # Select default audio devices
    input_device, output_device = None, None
    
    # Menu
    while True:
        print("\nVoice Converter Menu:")
        print("1. Convert an audio file")
        print("2. Start real-time voice conversion")
        print("3. Show available audio devices")
        print("4. Select audio devices")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            input_file = input("Enter path to input audio file: ")
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                continue
                
            output_file = input("Enter output file path (leave blank for auto-naming): ")
            if output_file.strip() == "":
                output_file = None
                
            try:
                processor.convert_file(input_file, output_file)
            except Exception as e:
                print(f"Error converting file: {e}")
                traceback.print_exc()
            
        elif choice == "2":
            print("Starting real-time voice conversion...")
            print("Speak into your microphone to convert your voice.")
            print("Press Ctrl+C to stop.")
            
            # Set selected devices before starting
            processor.set_devices(input_device, output_device)
            
            try:
                processor.start_realtime_processing()
            except Exception as e:
                print(f"Error in real-time processing: {e}")
                traceback.print_exc()
                
        elif choice == "3":
            list_available_devices()
            
        elif choice == "4":
            input_device, output_device = select_audio_device()
            print(f"Selected input device: {input_device}, output device: {output_device}")
            processor.set_devices(input_device, output_device)
            
        elif choice == "5":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()  # Keep console window open on error
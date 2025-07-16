import os
import json
import numpy as np
import torch
import librosa
import whisper
from tqdm import tqdm
import random
from typing import List, Dict, Tuple
import scipy.io.wavfile
from spec_augment import SpecAugment

import gc

# Define a class to add time warping capability to SpecAugment
class EnhancedSpecAugment:
    def __init__(self, freq_mask_param=30, time_mask_param=40, 
                 n_freq_mask=2, n_time_mask=8, mask_value=0,
                 warp_param=40):
        # Use the official SpecAugment library for freq and time masking
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            n_freq_mask=n_freq_mask,
            n_time_mask=n_time_mask,
            mask_value=mask_value
        )
        self.warp_param = warp_param

    def time_warp(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply time warping to the spectrogram
        """
        time_steps = spec.shape[1]
        if time_steps < 2:  # No warping needed for very short specs
            return spec
            
        center_point = random.randint(self.warp_param, time_steps - self.warp_param)
        warped_point = center_point + random.randint(-self.warp_param, self.warp_param)
        
        # Ensure warped_point is within valid range
        warped_point = max(0, min(time_steps - 1, warped_point))
        
        # Apply the warping transformation
        warped_spec = np.zeros_like(spec)
        
        # Map each column through the warping
        for t in range(time_steps):
            if t < center_point:
                scale = warped_point / center_point if center_point > 0 else 1.0
                new_t = int(t * scale)
            else:
                scale = (time_steps - warped_point) / (time_steps - center_point) if time_steps > center_point else 1.0
                new_t = int(warped_point + (t - center_point) * scale)
                
            new_t = max(0, min(time_steps - 1, new_t))
            warped_spec[:, new_t] = spec[:, t]
        
        # Fill any gaps (columns that weren't assigned values)
        for t in range(time_steps):
            if np.sum(warped_spec[:, t]) == 0:
                if t > 0:
                    warped_spec[:, t] = warped_spec[:, t-1]
                elif t < time_steps - 1:
                    warped_spec[:, t] = warped_spec[:, t+1]
        
        return warped_spec

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply full SpecAugment with time warping, frequency masking, and time masking
        """
        # First apply time warping
        augmented = self.time_warp(spec)
        
        # Then apply frequency and time masking using spec-augment library
        augmented = self.spec_augment(augmented)
        
        return augmented


def parse_wav_scp(wav_scp_path: str) -> Dict[str, str]:
    """
    Parse the wav.scp file to get utterance IDs and paths
    """
    utterance_to_wav = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utterance_id, wav_path = parts
                utterance_to_wav[utterance_id] = wav_path
    return utterance_to_wav


def parse_text_file(text_path: str) -> Dict[str, str]:
    """
    Parse the text file to get utterance IDs and transcriptions
    """
    utterance_to_text = {}
    with open(text_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utterance_id, transcription = parts
                utterance_to_text[utterance_id] = transcription
    return utterance_to_text


def apply_spec_augment(audio_path: str, output_path: str, augmenter: EnhancedSpecAugment) -> None:
    """
    Apply SpecAugment to audio and save the result
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Convert to log mel spectrogram
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Apply SpecAugment
    augmented_spec = augmenter(log_mel_spec)
    
    # Convert back to power mel spectrogram
    augmented_power_spec = librosa.db_to_power(augmented_spec)
    
    # Approximate inverse mel spectrogram
    augmented_stft = librosa.feature.inverse.mel_to_stft(augmented_power_spec, sr=sr)
    
    # Convert to audio
    augmented_audio = librosa.griffinlim(augmented_stft)
    
    # Save the augmented audio
    scipy.io.wavfile.write(output_path, sr, (augmented_audio * 32767).astype(np.int16))


def transcribe_with_whisper(audio_path: str, model) -> List[str]:
    """
    Transcribe audio using Whisper model and return top 5 transcriptions
    """
    result = model.transcribe(audio_path, beam_size=5)
    top_5_transcriptions = [result[i]["text"] for i in range(5)]

    torch.cuda.empty_cache()

    return top_5_transcriptions


def main():
    # Paths
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    wav_scp_path = "../SPAPL_KidsASR/egs/MyST/data/train_wav.scp"
    text_path = "../SPAPL_KidsASR/egs/MyST/data/train_text_edited"
    output_dir = "data"
    output_json_path = "nbest_train.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse input files
    utterance_to_wav = parse_wav_scp(wav_scp_path)
    utterance_to_text = parse_text_file(text_path)
    
    # Initialize the augmenter
    augmenter = EnhancedSpecAugment(
        freq_mask_param=30,
        time_mask_param=40,
        n_freq_mask=2,
        n_time_mask=8,
        mask_value=0,
        warp_param=40
    )
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("whisper-small-myst.pt")
    
    # Process each utterance
    dataset = []

    i = 0 
    
    for utterance_id, wav_path in tqdm(utterance_to_wav.items(), desc="Processing utterances"):
        if utterance_id not in utterance_to_text:
            print(f"Warning: No transcription found for utterance {utterance_id}")
            continue
        
        # Original transcription
        original_transcription = utterance_to_text[utterance_id]
        
        # Apply SpecAugment and save augmented audio
        augmented_wav_path = os.path.join(output_dir, f"{utterance_id}_augmented.wav")
        apply_spec_augment(wav_path, augmented_wav_path, augmenter)
        
        # Transcribe with Whisper
        top_5_transcriptions = transcribe_with_whisper(augmented_wav_path, model)
        
        # Add to dataset
        dataset.append({
            "input": top_5_transcriptions,
            "output": original_transcription
        })

        print("Processed 1 File!")

        i = i + 1
        if i % 10 == 0: 
            gc.collect()
            torch.cuda.empty_cache()
            i = 0
    
    # Save dataset to JSON
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Process completed. Dataset saved to {output_json_path}")


if __name__ == "__main__":
    main()

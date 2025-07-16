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
import re
from jiwer import wer

# EnhancedSpecAugment class
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

    # Clear some memory
    del y, mel_spec, log_mel_spec, augmented_spec, augmented_power_spec, augmented_stft, augmented_audio
    gc.collect()

import signal
import time
from functools import wraps

def timeout_handler(func, timeout=60):
    """Decorator to add timeout to a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Handler for timeout
        def alarm_handler(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")

        # Set the timeout handler
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)

        try:
            # Run the function
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Function {func.__name__} completed in {elapsed:.2f} seconds")
            return result
        except TimeoutError as e:
            print(str(e))
            return None
        finally:
            # Cancel the alarm
            signal.alarm(0)

    return wrapper

def transcribe_with_whisper(audio_path: str, model) -> List[str]:
    """
    Transcribe audio using Whisper model and return top 5 transcriptions
    """
    result = model.transcribe(audio_path, beam_size=5)
    top_5_transcriptions = [result[i]["text"] for i in range(5)]

    torch.cuda.empty_cache()

    return top_5_transcriptions


# Using it with your transcribe function
@timeout_handler
def safe_transcribe_with_whisper(audio_path, model):
    """Safely transcribe audio with timeout."""
    try:
        return transcribe_with_whisper(audio_path, model)
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate between reference and hypothesis
    """
    # Normalize text: remove punctuation, convert to lowercase
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    ref_normalized = normalize_text(reference)
    hyp_normalized = normalize_text(hypothesis)

    # Calculate WER
    return wer(ref_normalized, hyp_normalized)


def check_transcription_quality(transcriptions: List[str], reference: str, wer_threshold: float = 0.25) -> bool:
    """
    Check if at least one of the transcriptions has WER <= threshold
    """
    for transcription in transcriptions:
        current_wer = calculate_wer(reference, transcription)
        if current_wer <= wer_threshold:
            return True
    return False


def main():
    # Control which stages to run
    # startstage = 1 and endstage = 1 for creating augmented audio
    # startstage = 2 and endstage = 3 for creating n-best
    startstage = 2
    endstage = 3

    # Paths
    torch.cuda.empty_cache()
    torch.cuda.set_device(0)
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    wav_scp_path = "../SPAPL_KidsASR/egs/MyST/data/test_myst/wav.scp"
    text_path = "../SPAPL_KidsASR/egs/MyST/data/test_myst/text"
    output_dir = "data"
    augmented_dir = os.path.join(output_dir, "augmented_test")
    output_json_path = "nbest_test.json"

    # WER threshold for filtering
    wer_threshold = 0.25

    # JSON for tracking already processed files to enable resumption if needed
    processed_file = os.path.join(output_dir, "processed_files_test.json")
    filtered_stats_file = os.path.join(output_dir, "filtered_stats_test.json")

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)

    # Parse input files - needed for all stages
    if startstage <= 1:
        print("Parsing .scp and text...")
        utterance_to_wav = parse_wav_scp(wav_scp_path)
        utterance_to_text = parse_text_file(text_path)
        print("Parsing of .scp and text files completed!")
    
    # STAGE 1: Create augmented audio data
    if startstage <= 1 <= endstage:
        print("STAGE 1: Augmenting audio files and saving to disk...")
        
        # Initialize the augmenter (change the params here) 
        augmenter = EnhancedSpecAugment(
            freq_mask_param=10,
            time_mask_param=10,
            n_freq_mask=1,
            n_time_mask=1,
            mask_value=0,
            warp_param=0
        )

        # Track processed utterances and create a mapping of utterance ID to output path
        utterance_to_augmented_path = {}

        # Load already processed files if any
        processed_utterances = set()
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
                processed_utterances = set(processed_data.get("augmented", []))
                utterance_to_augmented_path = processed_data.get("mapping", {})

        for utterance_id, wav_path in tqdm(utterance_to_wav.items(), desc="Augmenting audio"):
            print("Augmenting: " + str(utterance_id))

            if utterance_id not in utterance_to_text:
                print(f"Warning: No transcription found for utterance {utterance_id}")
                continue

            # Skip if already processed
            if utterance_id in processed_utterances:
                continue

            # Define augmented audio path
            augmented_wav_path = os.path.join(augmented_dir, f"{utterance_id}_augmented.wav")

            # Apply SpecAugment and save augmented audio
            try:
                apply_spec_augment(wav_path, augmented_wav_path, augmenter)

                # Save the mapping
                utterance_to_augmented_path[utterance_id] = augmented_wav_path
                processed_utterances.add(utterance_id)

                # Save progress periodically
                if len(processed_utterances) % 10 == 0:
                    with open(processed_file, 'w') as f:
                        json.dump({
                            "augmented": list(processed_utterances),
                            "mapping": utterance_to_augmented_path
                        }, f)
                    gc.collect()

            except Exception as e:
                print(f"Error processing {utterance_id}: {e}")

        # Save final augmentation progress
        with open(processed_file, 'w') as f:
            json.dump({
                "augmented": list(processed_utterances),
                "mapping": utterance_to_augmented_path
            }, f)

        # Clear memory before finishing stage 1
        gc.collect()
        torch.cuda.empty_cache()
        
        print("STAGE 1 completed: Created augmented audio files")
    
    # STAGE 2: Load Whisper model
    model = None
    if 2 <= endstage and startstage <= 2:
        print("STAGE 2: Loading Whisper model...")
        model = whisper.load_model("whisper-fullft-pp.pt")
        print("STAGE 2 completed: Whisper model loaded")
    
    # STAGE 3: Transcribe augmented audio and create JSON dataset
    if 3 <= endstage and startstage <= 3:
        print("STAGE 3: Transcribing augmented audio files and filtering by WER...")
        
        # Load model if not already loaded in stage 2
        if model is None and (startstage > 2 or endstage < 2):
            print("Loading Whisper model...")
            model = whisper.load_model("whisper-fullft-pp.pt")
        
        # Load processed files mapping if starting directly at stage 3
        utterance_to_augmented_path = {}
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
                utterance_to_augmented_path = processed_data.get("mapping", {})
        
        # Check if utterance_to_text is already loaded, if not, load it
        try:
            _ = utterance_to_text
        except NameError:
            print("Loading text transcriptions...")
            utterance_to_text = parse_text_file(text_path)
        
        dataset = []
        filtered_stats = {
            "included": 0,
            "filtered_out": 0,
            "total_processed": 0
        }

        # Check if partial dataset exists and load it
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    dataset = json.load(f)
                print(f"Loaded existing dataset with {len(dataset)} entries")
            except:
                print("Error loading existing dataset, starting fresh")
                dataset = []

        # Check if filtered stats exist and load them
        if os.path.exists(filtered_stats_file):
            try:
                with open(filtered_stats_file, 'r') as f:
                    filtered_stats = json.load(f)
                print(f"Loaded existing filtered stats: {filtered_stats}")
            except:
                print("Error loading filtered stats, starting fresh")

        # Track which utterances have been transcribed
        transcribed_utterances = {item.get("utterance_id") for item in dataset if "utterance_id" in item}

        # Process each augmented file
        count = 0
        total = len(utterance_to_augmented_path)

        for utterance_id, augmented_wav_path in tqdm(utterance_to_augmented_path.items(),
                                                    desc="Transcribing audio"):

            print(f"Starting to process: {utterance_id} - {augmented_wav_path}")
            
            # Skip if already transcribed
            if utterance_id in transcribed_utterances:
                continue

            # Original transcription
            original_transcription = utterance_to_text[utterance_id]

            try:
                # Transcribe with Whisper
                top_5_transcriptions = safe_transcribe_with_whisper(augmented_wav_path, model)

                # Check if any transcription has acceptable WER
                filtered_stats["total_processed"] += 1
                if check_transcription_quality(top_5_transcriptions, original_transcription, wer_threshold):
                    # Add to dataset only if passes quality check
                    dataset.append({
                        "utterance_id": utterance_id,
                        "input": top_5_transcriptions,
                        "output": original_transcription
                    })
                    filtered_stats["included"] += 1
                    print(f"Added example with acceptable WER: {utterance_id}")
                else:
                    filtered_stats["filtered_out"] += 1
                    print(f"Filtered out example with high WER: {utterance_id}")

                count += 1
                print(f"Transcribed {count}/{total} files")

                # Save progress periodically
                if count % 10 == 0:
                    with open(output_json_path, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    with open(filtered_stats_file, 'w') as f:
                        json.dump(filtered_stats, f, indent=2)
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error transcribing {utterance_id}: {e}")

            # Clean up memory more aggressively
            if count % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Save final dataset to JSON
        with open(output_json_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        # Save final filtered stats
        with open(filtered_stats_file, 'w') as f:
            json.dump(filtered_stats, f, indent=2)

        print("STAGE 3 completed: Created JSON dataset")
        print(f"Filtering stats: {filtered_stats}")
        print(f"Included {filtered_stats['included']} examples and filtered out {filtered_stats['filtered_out']} examples")


if __name__ == "__main__":
    main()

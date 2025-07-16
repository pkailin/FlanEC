import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MEMORY_ALLOCATION'] = 'growth'

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
import jiwer
import argparse
from whisper_normalizer.english import EnglishTextNormalizer
import re
from num2words import num2words

# For testing original HuggingFace Model 
#from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperForConditionalGeneration, AutoConfig
#from peft import PeftModel, PeftConfig
#from huggingface_hub import hf_hub_download

# Completely reset CUDA
torch.cuda.empty_cache()
gc.collect()

# Force re-initialization of CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.set_device(0)
    dummy = torch.zeros(1).to(device)
    del dummy
    torch.cuda.empty_cache()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Set this environment variable to help with fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
sys.path.append(os.path.abspath("/home/klp65/rds/hpc-work/SPAPL_KidsASR/src"))

from arguments import PEFTArguments

# Initialize the Whisper text normalizer
normalizer = EnglishTextNormalizer()

def normalize_text(text):
    """
    Normalize text with the following steps:
    1. Convert to lowercase
    2. Apply Whisper's normalizer
    3. Remove all characters except letters, numbers, and spaces
    """
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Apply Whisper's normalizer
    text = normalizer(text)

    # Step 3: Remove all characters except letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)


    # Step 4: Convert numbers into text 
    # First, handle ordinal numbers like 1st, 2nd, 3rd, 4th
    def replace_ordinal(match):
        num_str = match.group(1)
        try:
            return num2words(int(num_str), ordinal=True)
        except ValueError:
            return match.group(0)

    # Handle regular numbers
    def replace_number(match):
        num = match.group(0)

        # Handle special cases like decimal numbers
        if '.' in num:  # Handle decimal numbers
            try:
                return num2words(float(num))
            except ValueError:
                return num
        else:
            try:
                return num2words(int(num))
            except ValueError:
                return num

    # First replace ordinals (must be done before regular numbers)
    # Pattern for ordinals like 1st, 2nd, 3rd, 4th, etc.
    ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    text = re.sub(ordinal_pattern, replace_ordinal, text)

    # Then replace regular numbers
    # Pattern for regular numbers
    number_pattern = r'\b\d+\b|\b\d+\.\d+\b'
    text = re.sub(number_pattern, replace_number, text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

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


def calculate_wer(reference, hypothesis) -> float:
    """
    # Calculate WER using jiwer
    """

    wer = jiwer.wer(reference, hypothesis)
    
    return wer


def transcribe_audio(audio_path: str, model) -> str:
    """
    Transcribe audio using the Whisper model
    """
    #result = model.transcribe(audio_path, initial_prompt="children speech aged 8 to 11 years old")
    result = model.transcribe(audio_path)
    text = result["text"].strip()
    text = normalize_text(text)
    
    return text

# Function to transcribe audio with HuggingFace
def transcribe_audio_hf(audio_path, model, processor):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Process audio
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")

    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode the predicted ids
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    transcription = normalize_text(transcription)

    return transcription


def main_fn(augment=False, freq_mask_param=30, time_mask_param=40, n_freq_mask=2, n_time_mask=8, warp_param=40):
    parser = argparse.ArgumentParser(description="Evaluate ASR with SpecAugment")
    #parser.add_argument("--wav_scp", type=str, default="/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/train_myst_nocomb/wav.scp", help="Path to wav.scp file")
    parser.add_argument("--wav_scp", type=str, default="/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/wav.scp", help="Path to wav.scp file")
    #parser.add_argument("--text", type=str, default="/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/train_myst_nocomb/text", help="Path to text file with transcriptions")
    parser.add_argument("--text", type=str, default="/home/klp65/rds/hpc-work/SPAPL_KidsASR/egs/MyST/data/test_myst/text", help="Path to text file with transcriptions")
    parser.add_argument("--output_dir", type=str, default="augmented_audio", help="Directory to save augmented audio")
    parser.add_argument("--model_path", type=str, default="whisper-fullft-pp.pt", help="Path to Whisper model")
    parser.add_argument("--num_utterances", type=int, default=9150, help="Number of utterances to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run model on (cuda/cpu)")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse input files
    utterance_to_wav = parse_wav_scp(args.wav_scp)
    utterance_to_text = parse_text_file(args.text)

    # Get common utterance IDs (those that appear in both files)
    common_utterances = list(set(utterance_to_wav.keys()) & set(utterance_to_text.keys()))

    # Sort the list to ensure consistent ordering before sampling
    common_utterances.sort()

    # Set a fixed random seed for reproducibility
    random.seed(args.seed)  # You can use any integer value as the seed

    if len(common_utterances) < args.num_utterances:
        print(f"Warning: Only {len(common_utterances)} utterances available. Using all of them.")
        num_to_process = len(common_utterances)
    else:
        num_to_process = args.num_utterances
    
    # Randomly select utterances
    selected_utterances = random.sample(common_utterances, num_to_process)

    # Initialize SpecAugment
    spec_augmenter = EnhancedSpecAugment(
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        n_freq_mask=n_freq_mask,
        n_time_mask=n_time_mask,
        warp_param=warp_param
    )

    
    # Load Whisper model
    # Check GPU memory before loading the model
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Free GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    print(f"Loading Whisper model from {args.model_path}...")
    model = whisper.load_model(args.model_path, device=args.device)
    
    '''
    print("Testing HF model, Loading...") 
    #Load model and processor from Hub
    #model_id = "pkailin2002/child_asr"
    #model_id = "balaji1312/whisper-small-myst-adapter"
    model_id = "balaji1312/whisper-small-myst-fullfinetune-pp"
    
    # Load config first to check for PEFT configurations
    #config = AutoConfig.from_pretrained(model_id)
    """
    # Load processor components
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    processor.current_processor = feature_extractor
    processor.feature_extractor = feature_extractor
    """
    #config.peft_config = PEFTArguments(**config.peft_config)
    #print(config.peft_config)

    """
    # Load the model with config
    base_model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        ignore_mismatched_sizes=True,
        config=config,
        force_download=True, 
        cache_dir=None,
    ).to("cuda")

    # Download the model weights file using HF API
    pytorch_model_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
    
    state_dict = torch.load(pytorch_model_path, map_location="cpu")

    # Examine the keys to understand the model architecture
    keys = list(state_dict.keys())
    print(f"Total keys in state dict: {len(keys)}")
    print("Sample keys:")
    for key in keys:  # Print first 10 keys
        print(f"  - {key}")

    # Check for any keys containing specific patterns that might indicate adapters
    adapter_patterns = ["adapter", "bottleneck", "prefix", "prompt", "lora", "peft"]
    for pattern in adapter_patterns:
        matching_keys = [k for k in keys if pattern in k.lower()]
        print(f"Keys containing '{pattern}': {len(matching_keys)}")
        if matching_keys:
            print(f"  Sample: {matching_keys[:3]}")
    
    # Filter only adapter weights
    adapter_weights = {k: v for k, v in state_dict.items() if "adapter" in k    }

    # Check what adapter weights we have
    print(f"Found {len(adapter_weights)} adapter weights")
    print("Sample adapter keys:", list(adapter_weights.keys())[:5])

    # Load just the adapter weights into the model
    load_result = base_model.load_state_dict(adapter_weights, strict=False)
    print(f"Loaded {len(adapter_weights)} adapter weights")
    print(f"Missing keys: {len(load_result.missing_keys)}")
    print(f"Unexpected keys: {len(load_result.unexpected_keys)}")

    model = base_model
    """

    #model = WhisperForConditionalGeneration.from_pretrained(model_id, force_download=True, cache_dir=None, local_files_only=False, ignore_mismatched_sizes=True, config=config).to("cuda")
    #model = WhisperForConditionalGeneration.from_pretrained(model_id, force_download=True, cache_dir=None, local_files_only=False).to("cuda")
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cuda")
    processor = WhisperProcessor.from_pretrained(model_id)
    '''
    
    
    # Process each utterance
    print(f"Processing {num_to_process} utterances...")

    # Store results
    results = []
    ref_lst = [] 
    hyp_lst = []
    
    for utt_id in tqdm(selected_utterances):
        # Get audio path and reference text
        audio_path = utterance_to_wav[utt_id]
        reference_text = utterance_to_text[utt_id]
        
        if augment == True: 
            # Create path for augmented audio
            augmented_audio_path = os.path.join(args.output_dir, f"{utt_id}_augmented.wav")
        
            try:
                # Apply SpecAugment
                apply_spec_augment(audio_path, augmented_audio_path, spec_augmenter)
            
                # Transcribe augmented audio
                hypothesis_text = transcribe_audio(augmented_audio_path, model)
            
                # Calculate WER
                wer = calculate_wer(reference_text, hypothesis_text)
            
                # Store result
                results.append({
                    "utterance_id": utt_id,
                    "reference": reference_text,
                    "hypothesis": hypothesis_text,
                    "wer": wer
                })
            
                # Print intermediate result
                print(f"Utterance {utt_id}: WER = {wer:.4f}")
                print(f"  Reference: {reference_text}")
                print(f"  Hypothesis: {hypothesis_text}")
                print()
            
            except Exception as e:
                print(f"Error processing utterance {utt_id}: {e}")

        else: 

            # HF model Testing 
            #hypothesis_text = transcribe_audio_hf(audio_path, model, processor)
            
            # Transcribe original audio
            hypothesis_text = transcribe_audio(audio_path, model)
            

            # Calculate WER
            wer = calculate_wer(reference_text, hypothesis_text)

            # Store result
            results.append({
                "utterance_id": utt_id,
                "reference": reference_text,
                "hypothesis": hypothesis_text,
                "wer": wer
            })

            # Print intermediate result
            print(f"Utterance {utt_id}: WER = {wer:.4f}")
            print(f"  Reference: {reference_text}")
            print(f"  Hypothesis: {hypothesis_text}")
            print()

        ref_lst.append(reference_text)
        hyp_lst.append(hypothesis_text)


    # Calculate overall WER
    if results:
        #overall_wer = sum(result["wer"] for result in results) / len(results)
        overall_wer = calculate_wer(ref_lst, hyp_lst)
        print(f"\nOverall WER: {overall_wer:.4f}")
        
        # Save detailed results
        with open("transcription_results.txt", "w") as f:
            f.write(f"Overall WER: {overall_wer:.4f}\n\n")
            for result in results:
                f.write(f"Utterance: {result['utterance_id']}\n")
                f.write(f"Reference: {result['reference']}\n")
                f.write(f"Hypothesis: {result['hypothesis']}\n")
                f.write(f"WER: {result['wer']:.4f}\n\n")
        
        print(f"Detailed results saved to transcription_results.txt")
    else:
        print("No results to report.")

    return overall_wer

main_fn()

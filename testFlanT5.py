import sys
import torch

# GPU Detection - Exit immediately if no GPU is available
def check_gpu_availability():
    """Check if GPU is available and exit if not found."""
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        print("CUDA is not available. This script requires a GPU to run.")
        print(f"Available devices: {torch.cuda.device_count()}")
        sys.exit(1)
    
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Call GPU check immediately
check_gpu_availability()

import json
import os
import transformers
import torch
from datasets import Dataset, DatasetDict
import whisper
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
import numpy as np
nltk.download('punkt')
import signal
import time
from functools import wraps
from tqdm import tqdm

import re
import jiwer
import random

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

from peft import PeftModel

model_path = "./flan-t5-comb-models/checkpoint-84350"  

# Load model 
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)
    
max_input_length = 1280
max_target_length = 256 

# takes in [hyp1, hyp2, hyp3, hyp4, hyp5]
def format_hypotheses_for_inference(
    hypotheses_list, 
    tokenizer,
    max_input_length=1280,
    max_output_length=256, 
    prefix_prompt="Generate the correct transcription for the following n-best list of ASR hypotheses:",
    suffix_prompt=""
):
    input_text = prefix_prompt + "\n\n"
    
    # Add each hypothesis with numbering
    for i, hypothesis in enumerate(hypotheses_list):
        input_text += f"{i+1}. {hypothesis}\n"
    
    input_text += f"{suffix_prompt}"
    
    # Tokenize exactly like in your dataset
    input_encoded = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,  
        return_tensors='pt',
        return_attention_mask=True,
        padding='max_length'
    )
    
    return input_encoded, input_text

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

"""
# Initialize the Whisper text normalizer
normalizer = EnglishTextNormalizer()

def normalize_text(text):
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
"""

print("Loading Whisper model...")
whisper_model = whisper.load_model("whisper-fullft-pp.pt")
print("Whisper model loaded")

torch.cuda.empty_cache()
torch.cuda.set_device(0)
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

wav_scp_path = "../SPAPL_KidsASR/egs/MyST/data/test_myst/wav.scp"
text_path = "../SPAPL_KidsASR/egs/MyST/data/test_myst/text"

print("Parsing .scp and text...")
utterance_to_wav = parse_wav_scp(wav_scp_path) # outputs dictionary [utterance_id: wav_path]
utterance_to_text = parse_text_file(text_path) # outputs dictionary [utterance_id: wav_path]
print("Parsing of .scp and text files completed!")

pred_lst = []
actual_lst = []

with open("testFlanT5-comb-corr_sclite.txt", "w", encoding="utf-8") as results_file:
    for utterance_id, wav_path in tqdm(utterance_to_wav.items(), desc="Transcribing audio"):
        print("Transcribing: " + str(utterance_id))

        if utterance_id not in utterance_to_text:
            print(f"Warning: No transcription found for utterance {utterance_id}")
            continue

        hypotheses = safe_transcribe_with_whisper(wav_path, whisper_model)
    
        if hypotheses != None:

            # Remove random entry and add correct answer in
            actual = utterance_to_text[utterance_id]
            actual = normalize_text(actual)

            removed = hypotheses.pop(random.randrange(len(hypotheses)))
            print('Removed: ' + str(removed))
            hypotheses.append(actual)

            print("Hypotheses: " + str(hypotheses)) # should output a list
            # Format the input
            input_encoded, formatted_text = format_hypotheses_for_inference(hypotheses, tokenizer)

            print("Formatted input:")
            print("-" * 50)
            print(formatted_text)
            print("-" * 50)
    
            # Move to same device as model
            input_ids = input_encoded['input_ids'].to("cuda")
            attention_mask = input_encoded['attention_mask'].to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_target_length, 
                    num_beams=3
                )

            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.replace("<s>", "").replace("</s>", "").strip()

            actual = utterance_to_text[utterance_id]

            # Apply normalizations
            prediction = normalize_text(prediction) 
            actual = normalize_text(actual) 

            print("Prediction: " + prediction)
            print("Actual: " + actual)
    
            pred_lst.append(prediction)
            actual_lst.append(actual)

            results_file.write(f"{utterance_id}<DIV>{prediction}<DIV>{actual}\n")

wer = jiwer.wer(pred_lst, actual_lst)
print("Overall WER: " + str(wer))


# Get Detailed Measures
measures = jiwer.compute_measures(pred_lst, actual_lst)
total_ref_words = measures['substitutions'] + measures['deletions'] + measures['hits']
sub_rate = measures['substitutions'] / total_ref_words
del_rate = measures['deletions'] / total_ref_words
ins_rate = measures['insertions'] / total_ref_words

print(f"SUB: {sub_rate:.4f}, DEL: {del_rate:.4f}, INS: {ins_rate:.4f}")


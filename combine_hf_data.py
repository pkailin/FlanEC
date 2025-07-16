import json
import re
import string
import argparse
import os
from datasets import load_dataset

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

def normalize_input_output(item):
    """
    Normalize input and output fields in a dataset item
    """
    # Normalize output (single string)
    if 'output' in item:
        item['output'] = normalize_text(item['output'])
    
    # Normalize input (can be string or list of strings)
    if 'input' in item:
        if isinstance(item['input'], list):
            item['input'] = [normalize_text(inp) for inp in item['input']]
        else:
            item['input'] = normalize_text(item['input'])
    
    return item

def load_combined_dataset(json_file_path, normalize=True):
    """
    Load and combine HyPoradise dataset from HuggingFace with local JSON dataset
    Args:
        json_file_path: Path to local JSON file
        normalize: Whether to normalize text (default: True)
    """
    # Load your local JSON dataset
    with open(json_file_path, 'r') as f:
        local_dataset = json.load(f)
    
    # Load HyPoradise dataset from HuggingFace with multiple fallback strategies
    hf_converted = []
    
    # Strategy 1: Try loading the dataset normally
    try:
        print("Attempting to load HuggingFace dataset (Strategy 1)...")
        hf_dataset = load_dataset("PeacefulData/HyPoradise-v0", split="train")
        print(f"Successfully loaded {len(hf_dataset)} entries from HuggingFace")
        hf_converted = process_hf_dataset(hf_dataset, normalize)
        
    except Exception as e1:
        print(f"Strategy 1 failed: {e1}")
        
        # Strategy 2: Try loading with streaming
        try:
            print("Attempting to load HuggingFace dataset with streaming (Strategy 2)...")
            hf_dataset = load_dataset("PeacefulData/HyPoradise-v0", split="train", streaming=True)
            hf_converted = process_hf_streaming_dataset(hf_dataset, normalize)
            
        except Exception as e2:
            print(f"Strategy 2 failed: {e2}")
            
            # Strategy 3: Try loading without specifying split
            try:
                print("Attempting to load HuggingFace dataset without split (Strategy 3)...")
                hf_dataset = load_dataset("PeacefulData/HyPoradise-v0")
                
                hf_converted = process_hf_dataset(hf_dataset, normalize)

                """
                if 'train' in hf_dataset:
                    hf_converted = process_hf_dataset(hf_dataset['train'], normalize)
                else:
                    # Use the first available split
                    first_split = next(iter(hf_dataset.keys()))
                    print(f"Using split: {first_split}")
                    hf_converted = process_hf_dataset(hf_dataset[first_split], normalize)
                """

            except Exception as e3:
                print(f"Strategy 3 failed: {e3}")
                print("All HuggingFace loading strategies failed. Continuing with only local JSON dataset...")
    
    # Normalize local dataset if requested
    if normalize:
        local_dataset = [normalize_input_output(item) for item in local_dataset]
    
    # Combine both datasets
    combined_dataset = hf_converted + local_dataset
    
    print(f"HuggingFace dataset size: {len(hf_converted)}")
    print(f"Local JSON dataset size: {len(local_dataset)}")
    print(f"Combined dataset size: {len(combined_dataset)}")
    print(f"Text normalization: {'enabled' if normalize else 'disabled'}")
    
    return combined_dataset

def process_hf_dataset(hf_dataset, normalize=True):
    """Process HuggingFace dataset into our format"""
    hf_converted = []
    
    for i, item in enumerate(hf_dataset):
        try:
            # Handle different possible column names and structures
            utterance_id = get_field_value(item, ['id', 'utterance_id'], f"hf_{i}")
            input_data = get_input_data(item, i)
            output_data = get_field_value(item, ['output', 'target'], None)
            
            if input_data is None or output_data is None:
                continue
            
            # Create converted item
            converted_item = {
                "utterance_id": str(utterance_id),
                "input": input_data,
                "output": str(output_data)
            }
            
            # Normalize text if requested
            if normalize:
                converted_item = normalize_input_output(converted_item)
            
            hf_converted.append(converted_item)
            
        except Exception as e:
            print(f"Warning: Error processing HF item {i}: {e}")
            continue
    
    return hf_converted

def process_hf_streaming_dataset(hf_dataset, normalize=True, max_items=None):
    """Process streaming HuggingFace dataset into our format"""
    hf_converted = []
    
    try:
        for i, item in enumerate(hf_dataset):
            if max_items and i >= max_items:
                break
                
            try:
                utterance_id = get_field_value(item, ['id', 'utterance_id'], f"hf_{i}")
                input_data = get_input_data(item, i)
                output_data = get_field_value(item, ['output', 'target'], None)
                
                if input_data is None or output_data is None:
                    continue
                
                converted_item = {
                    "utterance_id": str(utterance_id),
                    "input": input_data,
                    "output": str(output_data)
                }
                
                if normalize:
                    converted_item = normalize_input_output(converted_item)
                
                hf_converted.append(converted_item)
                
                # Print progress for streaming
                if i % 1000 == 0:
                    print(f"Processed {i} items from streaming dataset...")
                    
            except Exception as e:
                print(f"Warning: Error processing streaming item {i}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in streaming processing: {e}")
    
    print(f"Successfully processed {len(hf_converted)} items from streaming dataset")
    return hf_converted

def get_field_value(item, possible_keys, default=None):
    """Get value from item using possible key names"""
    for key in possible_keys:
        if key in item and item[key] is not None:
            return item[key]
    return default

def get_input_data(item, index):
    """Extract input data handling different formats"""
    # Try standard input field first
    if "input" in item and item["input"] is not None:
        return item["input"]
    
    # Try input1 and input2 combination
    if "input1" in item and "input2" in item:
        input1 = item["input1"] if item["input1"] is not None else ""
        input2 = item["input2"] if item["input2"] is not None else ""
        return [input1, input2]
    
    # Try just input1
    if "input1" in item and item["input1"] is not None:
        return item["input1"]
    
    # Try other possible input field names
    for field in ["text", "sentence", "source"]:
        if field in item and item[field] is not None:
            return item[field]
    
    print(f"Warning: No input field found for item {index}")
    return None

# Alternative: Class-based approach
class CombinedDatasetLoader:
    def __init__(self, json_file_path=None, include_huggingface=True, normalize=True):
        self.dataset = []
        self.normalize = normalize
        
        if include_huggingface:
            self.load_huggingface_data()
        
        if json_file_path:
            self.load_json_data(json_file_path)
    
    def load_huggingface_data(self):
        """Load HyPoradise dataset from HuggingFace with robust error handling"""
        try:
            print("Loading HuggingFace dataset...")
            hf_dataset = load_dataset("PeacefulData/HyPoradise-v0", split="train", 
                                     ignore_verifications=True, trust_remote_code=True)
            
            for i, item in enumerate(hf_dataset):
                try:
                    # Handle different possible column names and structures
                    utterance_id = None
                    input_data = None
                    output_data = None
                    
                    # Try to get ID from different possible fields
                    if "id" in item:
                        utterance_id = item["id"]
                    elif "utterance_id" in item:
                        utterance_id = item["utterance_id"]
                    else:
                        utterance_id = f"hf_{i}"
                    
                    # Try to get input from different possible fields
                    if "input" in item:
                        input_data = item["input"]
                    elif "input1" in item and "input2" in item:
                        input_data = [item["input1"], item["input2"]]
                    elif "input1" in item:
                        input_data = item["input1"]
                    else:
                        continue  # Skip if no input found
                    
                    # Try to get output from different possible fields
                    if "output" in item:
                        output_data = item["output"]
                    elif "target" in item:
                        output_data = item["target"]
                    else:
                        continue  # Skip if no output found
                    
                    converted_item = {
                        "utterance_id": str(utterance_id),
                        "input": input_data,
                        "output": str(output_data)
                    }
                    
                    # Normalize text if enabled
                    if self.normalize:
                        converted_item = normalize_input_output(converted_item)
                    
                    self.dataset.append(converted_item)
                    
                except Exception as e:
                    print(f"Warning: Error processing HF item {i}: {e}")
                    continue
            
            print(f"Loaded {len([item for item in self.dataset if item.get('utterance_id', '').startswith('hf_') or 'hf_' not in item.get('utterance_id', '')])} entries from HuggingFace")
        
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            print("Continuing without HuggingFace dataset...")
    
    def load_json_data(self, json_file_path):
        """Load local JSON dataset"""
        with open(json_file_path, 'r') as f:
            local_data = json.load(f)
        
        # Normalize local data if enabled
        if self.normalize:
            local_data = [normalize_input_output(item) for item in local_data]
        
        self.dataset.extend(local_data)
        print(f"Added {len(local_data)} entries from JSON file")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

# Example usage with normalization:
def main():
    parser = argparse.ArgumentParser(description='Combine HuggingFace HyPoradise dataset with local JSON dataset')
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument('output_json', help='Path to output JSON file')
    parser.add_argument('--no-normalize', action='store_true', help='Disable text normalization')
    parser.add_argument('--no-huggingface', action='store_true', help='Skip HuggingFace dataset')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input file '{args.input_json}' not found!")
        return
    
    # Determine settings
    normalize = not args.no_normalize
    include_hf = not args.no_huggingface
    
    print(f"Input file: {args.input_json}")
    print(f"Output file: {args.output_json}")
    print(f"Normalization: {'enabled' if normalize else 'disabled'}")
    print(f"Include HuggingFace dataset: {'yes' if include_hf else 'no'}")
    print("-" * 50)
    
    try:
        # Load and combine datasets
        if include_hf:
            combined_data = load_combined_dataset(args.input_json, normalize=normalize)
        else:
            # Load only local JSON with normalization
            with open(args.input_json, 'r') as f:
                combined_data = json.load(f)
            
            if normalize:
                combined_data = [normalize_input_output(item) for item in combined_data]
                print(f"Normalized {len(combined_data)} entries from local JSON")
        
        # Save combined dataset
        with open(args.output_json, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"\nSuccessfully saved combined dataset to '{args.output_json}'")
        print(f"Total entries: {len(combined_data)}")
        
        # Show sample entry
        if combined_data:
            print("\nSample entry:")
            sample = combined_data[0]
            print(f"  ID: {sample.get('utterance_id', 'N/A')}")
            print(f"  Input: {sample.get('input', 'N/A')}")
            print(f"  Output: {sample.get('output', 'N/A')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main()

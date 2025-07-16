import re
import jiwer
import string
import argparse
import sys

def fix_contractions(text):
    """
    Add apostrophes to common contractions in text.

    Args:
        text (str): Input text with missing apostrophes in contractions

    Returns:
        str: Text with apostrophes added to contractions
    """
    # Dictionary of contractions without apostrophes -> with apostrophes
    contractions = {
        # Common negative contractions
        "didn t": "didn't",
        "don t": "don't",
        "won t": "won't",
        "can t": "can't",
        "isn t": "isn't",
        "aren t": "aren't",
        "wasn t": "wasn't",
        "weren t": "weren't",
        "hasn t": "hasn't",
        "haven t": "haven't",
        "hadn t": "hadn't",
        "shouldn t": "shouldn't",
        "wouldn t": "wouldn't",
        "couldn t": "couldn't",
        "mustn t": "mustn't",
        "needn t": "needn't",
        "daren t": "daren't",
        "oughtn t": "oughtn't",

        # Contractions with "will"
        "i ll": "I'll",
        "you ll": "you'll",
        "he ll": "he'll",
        "she ll": "she'll",
        "it ll": "it'll",
        "we ll": "we'll",
        "they ll": "they'll",
        "that ll": "that'll",
        "who ll": "who'll",

        # Contractions with "am/is/are"
        "i m": "I'm",
        "you re": "you're",
        "he s": "he's",
        "she s": "she's",
        "it s": "it's",
        "we re": "we're",
        "they re": "they're",
        "that s": "that's",
        "who s": "who's",
        "what s": "what's",
        "where s": "where's",
        "when s": "when's",
        "how s": "how's",
        "why s": "why's",
        "there s": "there's",
        "here s": "here's",

        # Contractions with "have"
        "i ve": "I've",
        "you ve": "you've",
        "we ve": "we've",
        "they ve": "they've",
        "could ve": "could've",
        "should ve": "should've",
        "would ve": "would've",
        "might ve": "might've",
        "must ve": "must've",

        # Contractions with "had/would"
        "i d": "I'd",
        "you d": "you'd",
        "he d": "he'd",
        "she d": "she'd",
        "it d": "it'd",
        "we d": "we'd",
        "they d": "they'd",
        "that d": "that'd",
        "who d": "who'd",

        # Other common contractions
        "let s": "let's",
        "y all": "y'all",
        "o clock": "o'clock",
    }

    # Special cases to avoid - patterns that should NOT be converted
    avoid_patterns = {
        "i d cell": True,  # Specific case mentioned
        # Add other patterns as needed
    }

    # Convert to lowercase for matching, but preserve original case
    result = text
    for contraction, fixed in contractions.items():
        # Check for special avoidance patterns first
        should_skip = False
        for avoid_pattern in avoid_patterns:
            if contraction in avoid_pattern:
                # Use word boundaries to check if this contraction is part of the avoid pattern
                pattern = r'\b' + re.escape(avoid_pattern) + r'\b'
                if re.search(pattern, result, re.IGNORECASE):
                    should_skip = True
                    break

        if should_skip:
            continue

        # Use word boundaries to ensure we only match complete contractions
        pattern = r'\b' + re.escape(contraction) + r'\b'
        matches = list(re.finditer(pattern, result, re.IGNORECASE))

        # Replace from right to left to avoid position shifts
        for match in reversed(matches):
            start, end = match.span()
            original_text = result[start:end]

            # Preserve the original case pattern
            if original_text.isupper():
                replacement = fixed.upper()
            elif original_text.istitle():
                replacement = fixed.capitalize()
            elif original_text.islower():
                replacement = fixed.lower()
            else:
                # Mixed case - try to preserve the pattern
                replacement = ""
                for i, char in enumerate(fixed):
                    if i < len(original_text):
                        if original_text[i].isupper():
                            replacement += char.upper()
                        else:
                            replacement += char.lower()
                    else:
                        replacement += char.lower()
            result = result[:start] + replacement + result[end:]

    return result

from whisper_normalizer.english import EnglishTextNormalizer
import re
from num2words import num2words

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

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text2(text):
    return normalize_processor.tokenizer._normalize(text)

def parse_file(file_path):
    """
    Parse the input file and extract utterance_id, prediction, and actual text.
    Returns lists of predictions and actual text, plus data for output file.
    """
    pred_lst = []
    actual_lst = []
    parsed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split by <DIV> delimiter
                parts = line.split('<DIV>')
                
                if len(parts) != 3:
                    print(f"Warning: Line {line_num} doesn't have expected format (3 parts): {line}")
                    continue
                
                utterance_id = parts[0].strip()
                prediction = parts[1].strip()
                actual = parts[2].strip()
                
                # Normalize text
                actual = fix_contractions(actual)
                actual = normalize_text2(actual)

                norm_prediction = normalize_text(prediction)
                norm_actual = normalize_text(actual)
                
                pred_lst.append(norm_prediction)
                actual_lst.append(norm_actual)
                
                # Store for output file
                parsed_data.append({
                    'utterance_id': utterance_id,
                    'original_prediction': prediction,
                    'original_actual': actual,
                    'norm_prediction': norm_prediction,
                    'norm_actual': norm_actual
                })
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return pred_lst, actual_lst, parsed_data

def calculate_wer(predictions, actuals):
    """Calculate Word Error Rate using jiwer"""
    if len(predictions) != len(actuals):
        raise ValueError("Number of predictions and actual texts must be equal")
    
    if not predictions or not actuals:
        print("Warning: Empty lists provided for WER calculation")
        return 0.0
    
    # Calculate WER
    wer = jiwer.wer(actuals, predictions)
    
    # Get Detailed Measures
    measures = jiwer.compute_measures(actuals, predictions)
    total_ref_words = measures['substitutions'] + measures['deletions'] + measures['hits']
    sub_rate = measures['substitutions'] / total_ref_words
    del_rate = measures['deletions'] / total_ref_words
    ins_rate = measures['insertions'] / total_ref_words

    print(f"SUB: {sub_rate:.4f}, DEL: {del_rate:.4f}, INS: {ins_rate:.4f}")

    return wer


def write_normalized_file(output_path, parsed_data):
    """Write normalized data to output file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in parsed_data:
                line = f"{data['utterance_id']}<DIV>{data['norm_prediction']}<DIV>{data['norm_actual']}\n"
                f.write(line)
        print(f"Normalized file written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Calculate WER with text normalization')
    parser.add_argument('input_file', help='Input text file path')
    parser.add_argument('--output', '-o', help='Output file path (default: input_name_norm.txt)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        if input_file.endswith('.txt'):
            output_file = input_file[:-4] + '_norm.txt'
        else:
            output_file = input_file + '_norm.txt'
    
    print(f"Processing file: {input_file}")
    
    # Parse the input file
    pred_lst, actual_lst, parsed_data = parse_file(input_file)
    
    print(f"Parsed {len(pred_lst)} utterances")
    
    if not pred_lst:
        print("No valid data found in input file")
        sys.exit(1)
    
    # Calculate WER
    try:
        wer_score = calculate_wer(pred_lst, actual_lst)
        print(f"Word Error Rate (WER): {wer_score:.4f} ({wer_score * 100:.2f}%)")
    except Exception as e:
        print(f"Error calculating WER: {e}")
        sys.exit(1)
    
    # Write normalized file
    write_normalized_file(output_file, parsed_data)
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total utterances: {len(pred_lst)}")
    print(f"Average prediction length: {sum(len(p.split()) for p in pred_lst) / len(pred_lst):.1f} words")
    print(f"Average actual length: {sum(len(a.split()) for a in actual_lst) / len(actual_lst):.1f} words")

if __name__ == "__main__":
    main()

import re

from transformers import WhisperProcessor
normalize_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

def normalize_text(text):
    return normalize_processor.tokenizer._normalize(text)

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

def analyze_asr_log(log_content):
    """
    Analyze ASR log to calculate:
    1. Percentage of predictions matching actual transcriptions
    2. Percentage of actual transcriptions appearing in the hypotheses list
    """
    
    # Split the log into individual entries
    entries = log_content.strip().split('Function safe_transcribe_with_whisper completed in')
    
    prediction_matches = 0
    actual_in_hypotheses = 0
    total_entries = 0
    
    results = []
    
    for entry in entries:
        if not entry.strip():
            continue
            
        lines = entry.strip().split('\n')
        
        # Find hypotheses, prediction, and actual
        hypotheses = []
        prediction = None
        actual = None
        
        # Extract hypotheses from the first line
        hypotheses_match = re.search(r"Hypotheses: \[(.*?)\]", entry)
        if hypotheses_match:
            hypotheses_str = hypotheses_match.group(1)
            # Parse the list of hypotheses
            hypotheses = [h.strip().strip("'\"") for h in hypotheses_str.split("', '")]
            # Clean up any remaining quotes
            hypotheses = [h.strip("'\"") for h in hypotheses]
            hypotheses = [normalize_text(h) for h in hypotheses]
        
        # Extract prediction and actual
        for line in lines:
            if line.startswith('Prediction: '):
                prediction = line.replace('Prediction: ', '').strip()
                prediction = normalize_text(prediction)
            elif line.startswith('Actual: '):
                actual = line.replace('Actual: ', '').strip()
                #actual = fix_contractions(actual)
                actual = normalize_text(actual) 
        
        # Only process if we have all required data
        if hypotheses and prediction is not None and actual is not None:
            total_entries += 1
            
            # Check if prediction matches actual
            prediction_match = prediction.strip() == actual.strip()
            if prediction_match:
                prediction_matches += 1

            
            # Check if actual appears in any of the hypotheses
            actual_in_hyp = actual.strip() in [h.strip() for h in hypotheses]
            if actual_in_hyp:
                actual_in_hypotheses += 1
            else: 
                print(hypotheses)
                print(actual)
            
            results.append({
                'entry_num': total_entries,
                'hypotheses': hypotheses,
                'prediction': prediction,
                'actual': actual,
                'prediction_matches': prediction_match,
                'actual_in_hypotheses': actual_in_hyp
            })
    
    # Calculate percentages
    prediction_accuracy = (prediction_matches / total_entries * 100) if total_entries > 0 else 0
    actual_coverage = (actual_in_hypotheses / total_entries * 100) if total_entries > 0 else 0
    
    return {
        'total_entries': total_entries,
        'prediction_matches': prediction_matches,
        'actual_in_hypotheses': actual_in_hypotheses,
        'prediction_accuracy_percent': prediction_accuracy,
        'actual_coverage_percent': actual_coverage,
        'detailed_results': results
    }

def print_analysis_summary(analysis):
    """Print a summary of the analysis results"""
    print("ASR Log Analysis Results")
    print("=" * 40)
    print(f"Total entries analyzed: {analysis['total_entries']}")
    print(f"Prediction matches actual: {analysis['prediction_matches']}")
    print(f"Actual found in hypotheses: {analysis['actual_in_hypotheses']}")
    print()
    print(f"Prediction accuracy: {analysis['prediction_accuracy_percent']:.1f}%")
    print(f"Actual coverage in hypotheses: {analysis['actual_coverage_percent']:.1f}%")
    print()
    
    # Show examples of mismatches
    mismatches = [r for r in analysis['detailed_results'] if not r['prediction_matches']]
    if mismatches:
        print("Examples of prediction mismatches:")
        print("-" * 30)
        for i, mismatch in enumerate(mismatches[:3]):  # Show first 3 mismatches
            print(f"Entry {mismatch['entry_num']}:")
            print(f"  Prediction: '{mismatch['prediction']}'")
            print(f"  Actual: '{mismatch['actual']}'")
            print()

# Main execution
if __name__ == "__main__":
    import sys
    
    # Check if log file path is provided
    if len(sys.argv) != 2:
        print("Usage: python asr_analyzer.py <log_file_path>")
        print("Example: python asr_analyzer.py my_asr_log.txt")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    try:
        # Read the log file
        with open(log_file_path, 'r', encoding='utf-8') as file:
            log_content = file.read()
        
        print(f"Analyzing log file: {log_file_path}")
        print("=" * 50)
        
        # Analyze the log
        analysis = analyze_asr_log(log_content)
        
        # Print results
        print_analysis_summary(analysis)
        
        # Optionally save detailed results to a file
        output_file = log_file_path.replace('.', '_analysis.')
        if not output_file.endswith('.txt'):
            output_file += '.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ASR Log Analysis Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total entries analyzed: {analysis['total_entries']}\n")
            f.write(f"Prediction matches actual: {analysis['prediction_matches']}\n")
            f.write(f"Actual found in hypotheses: {analysis['actual_in_hypotheses']}\n\n")
            f.write(f"Prediction accuracy: {analysis['prediction_accuracy_percent']:.1f}%\n")
            f.write(f"Actual coverage in hypotheses: {analysis['actual_coverage_percent']:.1f}%\n\n")
            
            # Write detailed results
            f.write("Detailed Results:\n")
            f.write("-" * 20 + "\n")
            for result in analysis['detailed_results']:
                f.write(f"Entry {result['entry_num']}:\n")
                f.write(f"  Hypotheses: {result['hypotheses']}\n")
                f.write(f"  Prediction: '{result['prediction']}'\n")
                f.write(f"  Actual: '{result['actual']}'\n")
                f.write(f"  Prediction Match: {result['prediction_matches']}\n")
                f.write(f"  Actual in Hypotheses: {result['actual_in_hypotheses']}\n\n")
        
        print(f"\nDetailed analysis saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        sys.exit(1)

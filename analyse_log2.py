import re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
from difflib import SequenceMatcher
import statistics

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

    # Convert to lowercase for matching, but preserve original case
    result = text

    for contraction, fixed in contractions.items():
        # Replace case-insensitively but preserve the case of the original
        import re

        # Find all matches with their positions
        matches = list(re.finditer(re.escape(contraction), result, re.IGNORECASE))

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

class ASRHypothesesAnalyzer:
    def __init__(self):
        self.results = {}
    
    def parse_log_file(self, log_content):
        """Parse the log file and extract hypothesis groups"""
        entries = []
        current_entry = {}
        
        lines = log_content.strip().split('\n')
        i = 0

        #print(lines)
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for hypothesis section
            if line.startswith("Generate the correct transcription"):
                
                hypotheses = []
                i += 2

                # Extract numbered hypotheses
                while i < len(lines) and re.match(r'^\d+\.\s', lines[i].strip()):

                    hypothesis = re.sub(r'^\d+\.\s*', '', lines[i].strip())
                    hypothesis = fix_contractions(hypothesis)

                    hypotheses.append(hypothesis)
                    i += 1

                print(hypotheses)
                
                # Skip separator line
                while i < len(lines) and lines[i].strip().startswith('-'):
                    i += 1
                
                # Extract prediction and actual
                prediction = ""
                actual = ""
                filename = ""
                
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("Prediction:"):
                        prediction = line.replace("Prediction:", "").strip()
                    elif line.startswith("Actual:"):
                        actual = line.replace("Actual:", "").strip()
                    elif line.startswith("Transcribing:"):
                        filename = i 
                        break
                    i += 1

                print(prediction)
                print(actual)
                
                if hypotheses:
                    entries.append({
                        'hypotheses': hypotheses,
                        'prediction': prediction,
                        'actual': actual,
                        'filename': filename
                    })
            
            i += 1
        
        return entries
    
    def calculate_edit_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.calculate_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_word_level_differences(self, hyp1, hyp2):
        """Get word-level insertions, deletions, and substitutions"""
        words1 = hyp1.split()
        words2 = hyp2.split()
        
        matcher = SequenceMatcher(None, words1, words2)
        operations = {'insertions': [], 'deletions': [], 'substitutions': []}
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                operations['deletions'].extend(words1[i1:i2])
            elif tag == 'insert':
                operations['insertions'].extend(words2[j1:j2])
            elif tag == 'replace':
                # Count as substitutions
                for w1, w2 in zip(words1[i1:i2], words2[j1:j2]):
                    operations['substitutions'].append((w1, w2))
        
        return operations
    
    def analyze_hypothesis_group(self, hypotheses):
        """Analyze a group of 5 hypotheses"""
        n = len(hypotheses)
        results = {
            'num_hypotheses': n,
            'unique_hypotheses': len(set(hypotheses)),
            'diversity_ratio': len(set(hypotheses)) / n,
            'pairwise_distances': [],
            'avg_distance': 0,
            'word_operations': defaultdict(list),
            'common_substitutions': Counter(),
            'common_insertions': Counter(),
            'common_deletions': Counter(),
            'length_variance': 0,
            'word_position_variance': {},
            'consensus_words': [],
            'disagreement_positions': []
        }
        
        # Calculate pairwise edit distances
        distances = []
        all_operations = {'insertions': [], 'deletions': [], 'substitutions': []}
        
        for i, j in combinations(range(n), 2):
            distance = self.calculate_edit_distance(hypotheses[i], hypotheses[j])
            distances.append(distance)
            
            # Get word-level differences
            ops = self.get_word_level_differences(hypotheses[i], hypotheses[j])
            all_operations['insertions'].extend(ops['insertions'])
            all_operations['deletions'].extend(ops['deletions'])
            all_operations['substitutions'].extend(ops['substitutions'])
        
        results['pairwise_distances'] = distances
        results['avg_distance'] = np.mean(distances) if distances else 0
        results['distance_std'] = np.std(distances) if distances else 0
        
        # Count common operations
        results['common_insertions'] = Counter(all_operations['insertions'])
        results['common_deletions'] = Counter(all_operations['deletions'])
        results['common_substitutions'] = Counter(all_operations['substitutions'])
        
        # Length analysis
        lengths = [len(hyp.split()) for hyp in hypotheses]
        results['length_variance'] = np.var(lengths)
        results['avg_length'] = np.mean(lengths)
        results['length_range'] = max(lengths) - min(lengths)
        
        # Position-wise word analysis
        max_len = max(len(hyp.split()) for hyp in hypotheses)
        position_disagreements = 0
        
        for pos in range(max_len):
            words_at_pos = []
            for hyp in hypotheses:
                words = hyp.split()
                if pos < len(words):
                    words_at_pos.append(words[pos])
                else:
                    words_at_pos.append('<END>')
            
            unique_words = set(words_at_pos)
            if len(unique_words) > 1:
                position_disagreements += 1
                results['disagreement_positions'].append({
                    'position': pos,
                    'words': words_at_pos,
                    'unique_count': len(unique_words)
                })
        
        results['position_disagreement_ratio'] = position_disagreements / max_len if max_len > 0 else 0
        
        return results
    
    def analyze_log_file(self, log_content):
        """Analyze entire log file"""
        entries = self.parse_log_file(log_content)
        
        overall_results = {
            'total_entries': len(entries),
            'entry_analyses': [],
            'global_substitutions': Counter(),
            'global_insertions': Counter(),
            'global_deletions': Counter(),
            'diversity_stats': [],
            'distance_stats': []
        }
        
        for i, entry in enumerate(entries):
            
            print(f"Analysing Entry {i}")

            analysis = self.analyze_hypothesis_group(entry['hypotheses'])
            analysis['filename'] = entry['filename']
            analysis['prediction'] = entry['prediction']
            analysis['actual'] = entry['actual']
            
            overall_results['entry_analyses'].append(analysis)
            overall_results['diversity_stats'].append(analysis['diversity_ratio'])
            overall_results['distance_stats'].extend(analysis['pairwise_distances'])
            
            # Aggregate global statistics
            overall_results['global_substitutions'].update(analysis['common_substitutions'])
            overall_results['global_insertions'].update(analysis['common_insertions'])
            overall_results['global_deletions'].update(analysis['common_deletions'])
        
        # Calculate overall statistics
        if overall_results['diversity_stats']:
            overall_results['avg_diversity'] = np.mean(overall_results['diversity_stats'])
            overall_results['diversity_std'] = np.std(overall_results['diversity_stats'])
        
        if overall_results['distance_stats']:
            overall_results['avg_global_distance'] = np.mean(overall_results['distance_stats'])
            overall_results['distance_global_std'] = np.std(overall_results['distance_stats'])
        
        return overall_results
    
    def print_analysis_report(self, results):
        """Print a comprehensive analysis report"""
        print("=" * 80)
        print("ASR HYPOTHESES ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total entries analyzed: {results['total_entries']}")
        print(f"Average diversity ratio: {results.get('avg_diversity', 0):.3f}")
        print(f"Average pairwise distance: {results.get('avg_global_distance', 0):.2f}")
        
        print(f"\nTOP GLOBAL SUBSTITUTIONS:")
        for (w1, w2), count in results['global_substitutions'].most_common(10):
            print(f"  '{w1}' → '{w2}': {count} times")
        
        print(f"\nTOP GLOBAL INSERTIONS:")
        for word, count in results['global_insertions'].most_common(10):
            print(f"  '{word}': {count} times")
        
        print(f"\nTOP GLOBAL DELETIONS:")
        for word, count in results['global_deletions'].most_common(10):
            print(f"  '{word}': {count} times")
        
        print(f"\nPER-ENTRY ANALYSIS:")
        for i, entry in enumerate(results['entry_analyses']):
            print(f"\n--- Entry {i+1}: {entry['filename']} ---")
            print(f"Unique hypotheses: {entry['unique_hypotheses']}/5")
            print(f"Diversity ratio: {entry['diversity_ratio']:.3f}")
            print(f"Average distance: {entry['avg_distance']:.2f}")
            print(f"Length variance: {entry['length_variance']:.2f}")
            print(f"Position disagreement ratio: {entry['position_disagreement_ratio']:.3f}")
            
            if entry['common_substitutions']:
                print("Top substitutions:")
                for (w1, w2), count in entry['common_substitutions'].most_common(3):
                    print(f"  '{w1}' → '{w2}': {count} times")

def analyze_log_file_from_path(log_file_path):
    """Load and analyze a log file from the given path"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        analyzer = ASRHypothesesAnalyzer()
        results = analyzer.analyze_log_file(log_content)
        analyzer.print_analysis_report(results)
        
        return analyzer, results
    
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python asr_analysis.py <log_file_path>")
        print("Example: python asr_analysis.py my_log_file.txt")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    print(f"Analyzing log file: {log_file_path}")
    print("-" * 50)
    
    analyzer, results = analyze_log_file_from_path(log_file_path)
    
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Processed {results['total_entries']} entries")
       
        analyzer.print_analysis_report(results)

        
        # Optionally save results to a file
        save_results = input("\nSave detailed results to a file? (y/n): ").lower().strip()
        if save_results == 'y':
            with open('analyse_log2.txt', 'w', encoding='utf-8') as f:
                # Redirect print output to file
                import sys
                original_stdout = sys.stdout
                sys.stdout = f
                analyzer.print_analysis_report(results)
                sys.stdout = original_stdout
            print(f"Results saved!")
    

# Run the analysis
if __name__ == "__main__":
    main()

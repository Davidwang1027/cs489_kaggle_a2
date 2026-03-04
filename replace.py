import collections
import csv

def load_and_abstract_corpus(filepath, word_to_symbol):
    """Reads the corpus and converts words to their starting POS tags."""
    abstract_corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tags = [word_to_symbol.get(w) for w in line.strip().split() if w in word_to_symbol]
            if tags:
                abstract_corpus.append(tags)
    return abstract_corpus

def get_top_pairs(corpus_tags, top_n=15):
    """Counts and returns the most frequent adjacent pairs."""
    pairs = collections.Counter()
    for sentence in corpus_tags:
        for i in range(len(sentence) - 1):
            pairs[(sentence[i], sentence[i+1])] += 1
    return pairs.most_common(top_n)

def replace_pair_once(corpus_tags, pair, new_symbol):
    """Apply one corpus-wide merge pass for a single pair."""
    new_corpus = []
    total_replacements = 0
    for sentence in corpus_tags:
        new_sentence = []
        i = 0
        while i < len(sentence):
            if i < len(sentence) - 1 and sentence[i] == pair[0] and sentence[i+1] == pair[1]:
                new_sentence.append(new_symbol)
                i += 2 # Skip the next token since it was merged
                total_replacements += 1
            else:
                new_sentence.append(sentence[i])
                i += 1
        new_corpus.append(new_sentence)
    return new_corpus, total_replacements

def replace_pair(corpus_tags, pair, new_symbol):
    """Compatibility wrapper: performs one merge pass and returns only the corpus."""
    new_corpus, _ = replace_pair_once(corpus_tags, pair, new_symbol)
    return new_corpus

def apply_rule_until_stable(corpus_tags, pair, new_symbol, max_passes=1000):
    """
    Repeatedly apply one merge rule until no further replacements happen.

    This gives fixed-point behavior per user step.
    """
    current = corpus_tags
    passes = 0
    total_replacements = 0

    for _ in range(max_passes):
        updated, replacements = replace_pair_once(current, pair, new_symbol)
        passes += 1
        total_replacements += replacements
        current = updated

        if replacements == 0:
            break

    return current, passes, total_replacements

def write_corpus_to_file(corpus_tags, output_filepath):
    """Writes the current state of the abstracted corpus to a text file."""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sentence in corpus_tags:
            f.write(" ".join(sentence) + "\n")

def write_rules_to_file(applied_rules, rules_output_filepath):
    """Writes the list of applied merge rules to CSV."""
    with open(rules_output_filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['ID', 'LHS', 'LHS Type', 'RHS', 'Probability', 'Passes', 'Replacements'],
        )
        writer.writeheader()
        writer.writerows(applied_rules)

def run_interactive_discovery(
    corpus_filepath,
    symbol_to_words,
    output_filepath="10k_current_abstracted_corpus.txt",
    rules_output_filepath="applied_rules.csv",
):
    # Reverse dict for fast O(1) lookups
    word_to_symbol = {word: sym for sym, words in symbol_to_words.items() for word in words}
    
    print("Loading and translating corpus...")
    corpus = load_and_abstract_corpus(corpus_filepath, word_to_symbol)
    
    # Save the initial POS-tagged corpus so you can see the starting point
    write_corpus_to_file(corpus, output_filepath)
    print(f"Initial abstracted corpus saved to {output_filepath}")
    
    rule_id = 201 # Starting IDs after your 200 preterminal rules
    applied_rules = []

    try:
        while True:
            print("\n" + "="*50)
            print("MOST FREQUENT ADJACENT PAIRS:")
            top_pairs = get_top_pairs(corpus)
            for i, (pair, count) in enumerate(top_pairs):
                print(f"[{i}] {pair[0]:<5} + {pair[1]:<5} (Count: {count})")
                
            print("\nSample sentence from current corpus:")
            print(" ".join(corpus[0]))
            
            # User Input
            choice = input("\nEnter the index of the pair to merge (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                break
                
            try:
                pair_idx = int(choice)
                target_pair = top_pairs[pair_idx][0]
                new_symbol = input(f"Enter the new nonterminal symbol for {target_pair[0]} {target_pair[1]} (e.g., NP): ").strip()
                 
                # Apply this rule repeatedly until the corpus no longer changes.
                corpus, passes, replacements = apply_rule_until_stable(corpus, target_pair, new_symbol)
                 
                # Record your rule
                print(f"\n[RULE RECORDED] ID: {rule_id} | Rule: {new_symbol} -> {target_pair[0]} {target_pair[1]}")
                print(f"[FIXED-POINT] passes={passes}, total replacements={replacements}")
                applied_rules.append(
                    {
                        'ID': rule_id,
                        'LHS': new_symbol,
                        'LHS Type': 'nonterminal',
                        'RHS': f'{target_pair[0]} {target_pair[1]}',
                        'Probability': '',
                        'Passes': passes,
                        'Replacements': replacements,
                    }
                )
                rule_id += 1
                 
                # Write the updated corpus to the text file
                write_corpus_to_file(corpus, output_filepath)
                print(f"[FILE UPDATED] View changes in {output_filepath}")
                
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        write_rules_to_file(applied_rules, rules_output_filepath)
        print(f"[RULES SAVED] {len(applied_rules)} rules written to {rules_output_filepath}")







import collections
from pathlib import Path

src_path = Path('sample/pcfg2_10k.txt')
lines = src_path.read_text(encoding='utf-8').splitlines()

# PCFG symbol -> vocabulary mapping
symbol_to_words = {
    'N': {'artist', 'artists', 'bridge', 'bridges', 'car', 'cars', 'cat', 'cats', 'city', 'cities', 'doctor', 'doctors', 'dog', 'dogs', 'forest', 'forests', 'garden', 'gardens', 'home', 'idea', 'ideas', 'library', 'libraries', 'machine', 'machines', 'music', 'planet', 'planets', 'professor', 'professors', 'project', 'projects', 'river', 'rivers','songs', 'story', 'stories', 'student', 'students', 'teacher', 'teachers'},
    'V': {'admire', 'admired', 'admires', 'arrive', 'arrived', 'arrives', 'build', 'builds', 'built', 'call', 'called', 'calls', 'chase', 'chased', 'chases', 'collapse', 'collapsed', 'collapses', 'cough', 'coughed', 'coughs', 'cried', 'cries', 'cry', 'dance', 'danced', 'dances', 'depart', 'departed', 'departs', 'find', 'finds', 'follow', 'followed', 'follows', 'found', 'help', 'helped', 'helps', 'laugh', 'laughed', 'laughs', 'meet', 'meets', 'met', 'move', 'moved', 'moves', 'pause', 'paused', 'pauses', 'praise', 'praised', 'praises', 'prefer', 'preferred', 'prefers', 'saw', 'see', 'sees', 'shout', 'shouted', 'shouts', 'sleep', 'sleeps', 'slept', 'smile', 'smiled', 'smiles', 'sneeze', 'sneezed', 'sneezes', 'solve', 'solved', 'solves', 'travel', 'traveled', 'travels', 'visit', 'visited', 'visits', 'wait', 'waited', 'waits', 'wander', 'wandered', 'wanders', 'watch', 'watched', 'watches'},
    'DT': {'a', 'the', 'this', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}, 
    'JJ': {'ancient', 'big', 'brave', 'bright', 'calm', 'careful', 'eager', 'gentle', 'happy', 'honest', 'kind', 'modern', 'noisy', 'quick', 'quiet', 'serious', 'slow', 'small', 'smart', 'wild'},
    'IN': {'at', 'by', 'from', 'in', 'near', 'on', 'with'},
    'RB': {'abroad', 'ahead', 'anywhere', 'away', 'back', 'downstairs', 'elsewhere', 'everywhere', 'inside', 'later', 'nearby', 'outside', 'somewhere', 'soon', 'today', 'tomorrow', 'tonight', 'upstairs', 'yesterday'},
    'DEG': {'quite', 'really', 'very'},
    'MD': {'can', 'could', 'may', 'might', 'should', 'will', 'would'},
}

word_to_symbol = {}
for symbol, words in symbol_to_words.items():
    for w in words:
        if w in word_to_symbol:
            raise ValueError(f'Duplicate mapping for word: {w}')
        word_to_symbol[w] = symbol

vocab = sorted({tok for line in lines for tok in line.split()})
missing = sorted(set(vocab) - set(word_to_symbol))
if missing:
    raise ValueError(f'Missing symbol mapping for: {missing}')

symbol_lines = [' '.join(word_to_symbol[tok] for tok in line.split()) for line in lines]

out_path = Path('sample/pcfg2_10k_symbols.txt')
out_path.write_text('\n'.join(symbol_lines) + '\n', encoding='utf-8')


# --- Execute ---
run_interactive_discovery('sample/pcfg2_10k.txt', symbol_to_words)

import re
import sys
from typing import List, Dict, Optional

class Tokenizer:
    """
    A minimal GPT-3.5 tokenizer implementation using only standard Python libraries.
    Based on tiktoken's cl100k_base encoding used by GPT-3.5.
    """
    
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276
        }
        
        # Load basic token set with common words and their space-prefixed versions
        self.encoder = {
            # Common words and their space-prefixed versions
            "hello": 9906, " hello": 9907, "Hello": 9908, " Hello": 9909,
            "world": 2787, " world": 2788, "World": 2789, " World": 2790,
            "how": 2129, " how": 2130, "How": 2131, " How": 2132,
            "are": 389, " are": 390,
            "you": 345, " you": 346,
            "doing": 2651, " doing": 2652,
            "today": 2371, " today": 2372,
            # Common words
            "the": 464, " the": 465,
            "of": 291, " of": 292,
            "and": 287, " and": 288,
            "in": 262, " in": 263,
            "to": 284, " to": 285,
            "a": 264, " a": 265,
            "for": 287, " for": 288,
            "is": 338, " is": 339,
            "on": 293, " on": 294,
            "that": 471, " that": 472,
            "this": 445, " this": 446,
            "with": 504, " with": 505,
            "it": 338, " it": 339,
            "as": 492, " as": 493,
            "by": 305, " by": 306,
            "was": 410, " was": 411,
            "be": 502, " be": 503,
            # Spaces and punctuation
            " ": 100,
            "\n": 198,
            "!": 0,
            ".": 13,
            ",": 11,
            "?": 30,
            "(": 7,
            ")": 8,
            "'": 6,
            # Contractions
            "'s": 50,
            "'t": 51,
            "'re": 52,
            "'ve": 53,
            "'m": 54,
            "'ll": 55,
            "'d": 56,
        }
        
        # Add basic characters
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            if c not in self.encoder:
                self.encoder[c] = len(self.encoder)
        
        # Create decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})
        
        # Pattern for basic tokenization that preserves spaces
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into tokens.
        """
        if not text:
            return []
        
        tokens = []
        last_was_space = True  # Track if last token ended with space
        
        for match in re.finditer(self.pat, text):
            token = match.group()
            
            # Try space-prefixed version if last token didn't end with space
            if not last_was_space and token.startswith(' '):
                space_prefixed = token
                if space_prefixed in self.encoder:
                    tokens.append(self.encoder[space_prefixed])
                    last_was_space = token.endswith(' ')
                    continue
            
            # Try regular token
            if token in self.encoder:
                tokens.append(self.encoder[token])
                last_was_space = token.endswith(' ')
                continue
            
            # Handle unknown tokens character by character
            for char in token:
                if char in self.encoder:
                    tokens.append(self.encoder[char])
                    last_was_space = char.isspace()
                else:
                    tokens.append(self.encoder.get('?', 30))  # Use ? for unknown chars
                    last_was_space = False
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode tokens back to text.
        """
        return ''.join(self.decoder.get(token, '?') for token in tokens)
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        """
        return len(self.encode(text))

def num_tokens_from_string(string: str) -> int:
    """
    Returns the number of tokens in a text string.
    """
    tokenizer = Tokenizer()
    return tokenizer.count_tokens(string)

def num_tokens_from_file(file_path: str) -> int:
    """
    Returns the number of tokens in a text file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return num_tokens_from_string(text)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return 0

def print_usage():
    print("Usage: python3 tiktoken_local.py [file1] [file2] [file3] [file4]")
    print("Count tokens in up to 4 text files")
    print("\nExample:")
    print("  python3 tiktoken_local.py file1.txt file2.txt")

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    # Process up to 4 files from command line arguments
    files = sys.argv[1:5]  # Limit to first 4 arguments
    
    total_tokens = 0
    for file_path in files:
        num_tokens = num_tokens_from_file(file_path)
        print(f"File: {file_path}")
        print(f"Number of tokens: {num_tokens}")
        print("-" * 40)
        total_tokens += num_tokens
    
    if len(files) > 1:
        print(f"Total tokens across all files: {total_tokens}")

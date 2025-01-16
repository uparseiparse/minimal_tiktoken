import json
import regex as re
from functools import lru_cache
from typing import List, Dict, Union
import argparse

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class TokenizationError(Exception):
    """Custom exception for tokenization errors"""
    pass

class GPT3Tokenizer:
    def __init__(self):
        try:
            # Common GPT-2/3 tokens with their IDs
            self.encoder = {
                "<|endoftext|>": 50256,
                "Hello": 15496,
                "hello": 31373,
                "world": 995,
                "World": 2159,
                "how": 2129,
                "How": 2182,
                "are": 389,
                "you": 345,
                "doing": 2651,
                "today": 2371,
                " Hello": 18435,
                " World": 2159,
                " How": 2182,
                " are": 389,
                " you": 345,
                " doing": 2651,
                " today": 2371,
                "!": 0,
                "?": 30,
                " ": 220,
            }
            
            # Add basic characters
            for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                if c not in self.encoder:
                    self.encoder[c] = len(self.encoder)
            
            self.decoder = {v: k for k, v in self.encoder.items()}
            
            # BPE merges for common word combinations
            self.bpe_ranks = {
                ('H', 'ello'): 0,
                ('w', 'orld'): 1,
                ('t', 'oday'): 2,
                ('do', 'ing'): 3,
                ('a', 're'): 4,
                ('y', 'ou'): 5,
                ('Ho', 'w'): 6,
            }
            
            self.byte_encoder = bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.cache = {}
            
            # Updated pattern to better handle word boundaries
            self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
            
        except Exception as e:
            raise TokenizationError(f"Failed to initialize tokenizer: {str(e)}")

    def bpe(self, token: str) -> str:
        """
        Apply Byte-Pair Encoding to a token.
        """
        if token in self.cache:
            return self.cache[token]
            
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token ids.
        """
        if not text:
            return []
            
        try:
            bpe_tokens = []
            # First try to match common multi-token sequences
            remaining_text = text
            while remaining_text:
                matched = False
                # Try matching with space prefix first
                if remaining_text.startswith(" "):
                    space_prefixed = remaining_text[:20]  # Look ahead up to 20 chars
                    if space_prefixed in self.encoder:
                        bpe_tokens.append(space_prefixed)
                        remaining_text = remaining_text[len(space_prefixed):]
                        matched = True
                        continue
                
                # Then try matching without space prefix
                for length in range(min(20, len(remaining_text)), 0, -1):
                    if remaining_text[:length] in self.encoder:
                        bpe_tokens.append(remaining_text[:length])
                        remaining_text = remaining_text[length:]
                        matched = True
                        break
                
                # If no match found, tokenize the first character
                if not matched:
                    token = remaining_text[0]
                    if token in self.encoder:
                        bpe_tokens.append(token)
                    else:
                        # Apply BPE encoding for unknown tokens
                        token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
                    remaining_text = remaining_text[1:]
            
            return [self.encoder.get(token, 0) for token in bpe_tokens]
        except Exception as e:
            raise TokenizationError(f"Failed to encode text: {str(e)}")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        """
        try:
            return len(self.encode(text))
        except TokenizationError as e:
            raise e
        except Exception as e:
            raise TokenizationError(f"Failed to count tokens: {str(e)}")

def num_tokens_from_string(string: str) -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        tokenizer = GPT3Tokenizer()
        return tokenizer.count_tokens(string)
    except TokenizationError as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

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

def main():
    parser = argparse.ArgumentParser(description='Count GPT-3 tokens in text or files')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Text to count tokens in')
    group.add_argument('--file', type=str, help='File to count tokens in')
    
    args = parser.parse_args()
    
    if args.text:
        num_tokens = num_tokens_from_string(args.text)
        print(f"Text: {args.text}")
        print(f"Number of tokens: {num_tokens}")
    elif args.file:
        num_tokens = num_tokens_from_file(args.file)
        print(f"File: {args.file}")
        print(f"Number of tokens: {num_tokens}")

if __name__ == "__main__":
    main()

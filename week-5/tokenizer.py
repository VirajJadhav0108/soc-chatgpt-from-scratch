"""
Week 5 — Character Level Tokenizer
SoC: Build Your Own ChatGPT from Scratch

This file creates a simple character-level tokenizer.
Every unique character in the dataset is assigned a unique integer.

Example:

Input:
hello

Vocabulary:
['e', 'h', 'l', 'o']

Encoding:
hello -> [1,0,2,2,3]

Decoding:
[1,0,2,2,3] -> hello
"""

import os


# -------------------------------------------------------------------
# Read Dataset
# -------------------------------------------------------------------

DATA_PATH = "tinyshakespeare.txt"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset '{DATA_PATH}' not found."
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total characters : {len(text)}")


# -------------------------------------------------------------------
# Build Vocabulary
# -------------------------------------------------------------------

chars = sorted(list(set(text)))

vocab_size = len(chars)

print(f"Unique characters : {vocab_size}")

print("\nVocabulary:")

print(chars)


# -------------------------------------------------------------------
# Character → Integer Mapping
# -------------------------------------------------------------------

char_to_idx = {}

for idx, ch in enumerate(chars):
    char_to_idx[ch] = idx


# -------------------------------------------------------------------
# Integer → Character Mapping
# -------------------------------------------------------------------

idx_to_char = {}

for idx, ch in enumerate(chars):
    idx_to_char[idx] = ch


print("\nCharacter Mapping (First 15)\n")

count = 0

for ch in chars:

    print(f"{repr(ch):>5} -> {char_to_idx[ch]}")

    count += 1

    if count == 15:
        break


# -------------------------------------------------------------------
# Encoding Function
# -------------------------------------------------------------------

def encode(text):

    """
    Convert a string into a list of integers.
    """

    encoded = []

    for ch in text:

        encoded.append(char_to_idx[ch])

    return encoded


# -------------------------------------------------------------------
# Decoding Function
# -------------------------------------------------------------------

def decode(tokens):

    """
    Convert integer tokens back into text.
    """

    decoded = ""

    for token in tokens:

        decoded += idx_to_char[token]

    return decoded


# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------

print("\n" + "=" * 60)
print("TOKENIZER TEST")
print("=" * 60)

sample = "To be, or not to be"

print("\nOriginal Text:")
print(sample)

encoded = encode(sample)

print("\nEncoded:")
print(encoded)

decoded = decode(encoded)

print("\nDecoded:")
print(decoded)


# -------------------------------------------------------------------
# Verify
# -------------------------------------------------------------------

print("\nVerification")

if sample == decoded:
    print("Tokenizer is working correctly.")
else:
    print("Something went wrong.")


# -------------------------------------------------------------------
# Encode Entire Dataset
# -------------------------------------------------------------------

dataset = encode(text)

print("\n" + "=" * 60)
print("ENCODED DATASET")
print("=" * 60)

print(f"Number of tokens : {len(dataset)}")

print("\nFirst 100 Tokens:")

print(dataset[:100])


# -------------------------------------------------------------------
# Key Takeaways
# -------------------------------------------------------------------

print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)

print("1. GPT models cannot understand raw text.")
print("2. Every character is converted into an integer token.")
print("3. The vocabulary contains every unique character.")
print("4. Encoding converts text → integers.")
print("5. Decoding converts integers → text.")
print("6. These integer tokens will be used to train the language model.")

"""
Week 1 - Pandas Basics
SoC: Build Your Own ChatGPT from Scratch
Covers: Series, DataFrame, data cleaning, aggregation — needed for dataset preprocessing
"""

import pandas as pd
import numpy as np

print("=" * 50)
print("Week 1 — Pandas Basics")
print("=" * 50)

# ── 1. Series & DataFrame Creation ────────────────────────────────────────────
print("\n1. Creating DataFrames")

# Simulating a small text dataset (like what we'd use for LLM training)
data = {
    "sentence": [
        "the cat sat on the mat",
        "neural networks learn representations",
        "transformers use attention mechanisms",
        "embeddings map words to vectors",
        "backpropagation computes gradients",
    ],
    "word_count": [6, 4, 4, 5, 3],
    "has_neural_term": [False, True, True, True, True],
    "char_count": [22, 35, 34, 32, 31],
}

df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDtypes:\n{df.dtypes}")

# ── 2. Indexing & Filtering ────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. Indexing & Filtering")

print("\nSentences with word_count > 4:")
print(df[df["word_count"] > 4]["sentence"])

print("\nNeural-term sentences:")
print(df[df["has_neural_term"]]["sentence"].values)

print("\nFirst sentence character count:", df.loc[0, "char_count"])

# ── 3. Basic Statistics ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. Descriptive Statistics")

print(df[["word_count", "char_count"]].describe())

print(f"\nMean word count: {df['word_count'].mean():.2f}")
print(f"Max char count: {df['char_count'].max()}")
print(f"Sentences with neural terms: {df['has_neural_term'].sum()} / {len(df)}")

# ── 4. Creating / Transforming Columns ────────────────────────────────────────
print("\n" + "=" * 50)
print("4. Feature Engineering (like we'd do before tokenization)")

df["avg_word_len"] = df["char_count"] / df["word_count"]
df["tokens_approx"] = df["word_count"] + 2   # +2 for BOS/EOS tokens
df["sentence_upper"] = df["sentence"].str.upper()

print(df[["sentence", "word_count", "avg_word_len", "tokens_approx"]])

# ── 5. Groupby & Aggregation ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("5. Groupby Aggregation")

df["category"] = ["basic", "ml", "ml", "ml", "ml"]
grouped = df.groupby("category")[["word_count", "char_count"]].mean()
print(grouped)

# ── 6. Handling Missing Data ──────────────────────────────────────────────────
print("\n" + "=" * 50)
print("6. Handling Missing Values (common in raw text datasets)")

df_missing = df.copy()
df_missing.loc[1, "char_count"] = np.nan
df_missing.loc[3, "word_count"] = np.nan

print(f"Missing values:\n{df_missing.isnull().sum()}")

df_filled = df_missing.fillna({
    "char_count": df_missing["char_count"].median(),
    "word_count": df_missing["word_count"].median(),
})
print(f"\nAfter filling nulls:\n{df_filled[['sentence','word_count','char_count']]}")

# ── 7. Vocabulary Building (mini tokenizer prep) ──────────────────────────────
print("\n" + "=" * 50)
print("7. Building a Vocabulary from Text Data")

corpus = " ".join(df["sentence"].tolist())
words = corpus.split()
word_series = pd.Series(words)
word_counts = word_series.value_counts()

print(f"Total words in corpus: {len(words)}")
print(f"Unique vocabulary size: {len(word_counts)}")
print(f"\nTop 10 words:\n{word_counts.head(10)}")

# Build word2idx (like a simple tokenizer)
vocab = ["<PAD>", "<BOS>", "<EOS>"] + word_counts.index.tolist()
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

sample = "transformers use attention"
tokens = [word2idx.get(w, 0) for w in sample.split()]
print(f"\nEncoded '{sample}': {tokens}")
print(f"Decoded back: {[idx2word[t] for t in tokens]}")

print("\n[Week 1 complete] Pandas basics covered.")

"""
Week 1 - NumPy Basics
SoC: Build Your Own ChatGPT from Scratch
Covers: array creation, operations, reshaping, broadcasting — fundamentals needed for neural nets
"""

import numpy as np

# ── 1. Array Creation ──────────────────────────────────────────────────────────
print("=" * 50)
print("1. Array Creation")
print("=" * 50)

a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 3))
c = np.ones((2, 4))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)
rand_mat = np.random.randn(3, 3)

print(f"1D array: {a}")
print(f"Zeros (3x3):\n{b}")
print(f"Ones (2x4):\n{c}")
print(f"Arange 0-10 step 2: {d}")
print(f"Linspace 0-1 (5 pts): {e}")
print(f"Random 3x3:\n{rand_mat}")

# ── 2. Array Operations ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("2. Element-wise Operations")
print("=" * 50)

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

print(f"x + y = {x + y}")
print(f"x * y = {x * y}")           # element-wise (NOT dot product)
print(f"x dot y = {np.dot(x, y)}")  # actual dot product
print(f"x ** 2 = {x ** 2}")
print(f"sqrt(x) = {np.sqrt(x)}")
print(f"exp(x) = {np.exp(x)}")

# ── 3. Reshaping & Axes ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("3. Reshaping — Critical for Tensors in Neural Networks")
print("=" * 50)

mat = np.arange(12)
print(f"Original (12,): {mat}")
reshaped = mat.reshape(3, 4)
print(f"Reshaped (3,4):\n{reshaped}")
print(f"Transposed (4,3):\n{reshaped.T}")
print(f"Flattened back: {reshaped.flatten()}")

# ── 4. Broadcasting ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("4. Broadcasting — How NumPy handles shape mismatches")
print("=" * 50)

A = np.array([[1, 2, 3],
              [4, 5, 6]])         # shape (2, 3)
bias = np.array([10, 20, 30])    # shape (3,)  → broadcasts over rows

result = A + bias
print(f"Matrix A (2x3):\n{A}")
print(f"Bias (3,): {bias}")
print(f"A + bias (broadcast):\n{result}")

# ── 5. Aggregations ────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("5. Aggregations — Sum, Mean, Std (used in layer norm)")
print("=" * 50)

data = np.random.randn(4, 5)
print(f"Matrix shape: {data.shape}")
print(f"Global mean: {data.mean():.4f}")
print(f"Column means: {data.mean(axis=0)}")
print(f"Row sums: {data.sum(axis=1)}")
print(f"Std dev: {data.std():.4f}")

# ── 6. Matrix Multiplication ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("6. Matrix Multiplication — Core of every linear layer")
print("=" * 50)

W = np.random.randn(4, 3)   # weight matrix
x_in = np.random.randn(3)   # input vector
out = W @ x_in              # matmul: (4,3) @ (3,) = (4,)
print(f"W shape: {W.shape}, x shape: {x_in.shape}")
print(f"W @ x shape: {out.shape}")
print(f"W @ x = {out}")

# Batch matmul: (batch, seq, d_model) @ (d_model, d_k)
batch, seq, d_model, d_k = 2, 5, 8, 4
X_batch = np.random.randn(batch, seq, d_model)
W_q = np.random.randn(d_model, d_k)
Q = X_batch @ W_q   # (2, 5, 4) — this is how Query projection works in attention
print(f"\nBatch Query projection: {X_batch.shape} @ {W_q.shape} = {Q.shape}")

# ── 7. Softmax (manual) ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("7. Softmax from Scratch — Used in attention mechanism")
print("=" * 50)

def softmax(x):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x)   # subtract max for numerical stability
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum()

logits = np.array([2.0, 1.0, 0.1, -1.0])
probs = softmax(logits)
print(f"Logits: {logits}")
print(f"Softmax: {probs}")
print(f"Sum of probs: {probs.sum():.6f}  (should be 1.0)")

print("\n[Week 1 complete] NumPy fundamentals covered.")

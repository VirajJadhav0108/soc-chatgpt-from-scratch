"""
Week 2 & 3 — Neural Networks from Scratch (NumPy only)
SoC: Build Your Own ChatGPT from Scratch
Based on: StatQuest Neural Networks playlist (videos 74–84)

Implements: Perceptron → MLP → Forward Pass → Backpropagation → Training Loop
"""

import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    """Numerically stable softmax — works on 2D arrays (batch, classes)."""
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_deriv(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_pred, y_true):
    """y_pred: (N, C) probabilities; y_true: (N,) integer class indices."""
    N = y_true.shape[0]
    clipped = np.clip(y_pred[np.arange(N), y_true], 1e-9, 1.0)
    return -np.log(clipped).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Layer — Dense (Fully Connected)
# ─────────────────────────────────────────────────────────────────────────────

class DenseLayer:
    """
    A single fully-connected layer: output = activation(X @ W + b)
    Stores gradients internally; updated by the optimizer.
    """

    def __init__(self, in_features, out_features, activation="relu"):
        # He initialization for ReLU, Xavier for tanh/sigmoid
        if activation == "relu":
            scale = np.sqrt(2.0 / in_features)
        else:
            scale = np.sqrt(1.0 / in_features)

        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))

        self.activation_name = activation
        self.activation = {"relu": relu, "sigmoid": sigmoid, "tanh": tanh, "linear": lambda x: x}[activation]
        self.activation_deriv = {"relu": relu_deriv, "sigmoid": sigmoid_deriv, "tanh": tanh_deriv, "linear": lambda x: np.ones_like(x)}[activation]

        # Cache for backprop
        self.X = None     # input
        self.Z = None     # pre-activation
        self.A = None     # post-activation

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, dA):
        """
        dA: gradient w.r.t. this layer's output (shape: batch x out_features)
        Returns dX: gradient to pass to the previous layer.
        """
        dZ = dA * self.activation_deriv(self.Z)   # element-wise chain rule
        self.dW = self.X.T @ dZ / self.X.shape[0]
        self.db = dZ.mean(axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dX


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MLP — Multi-Layer Perceptron
# ─────────────────────────────────────────────────────────────────────────────

class MLP:
    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        activations: list of activation names (len = len(layer_sizes) - 1)
        """
        assert len(activations) == len(layer_sizes) - 1
        self.layers = []
        for i in range(len(activations)):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dLoss):
        grad = dLoss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def predict(self, X):
        return self.forward(X)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Training — Regression Example (XOR)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 55)
print("DEMO 1: XOR Problem (Regression with MLP)")
print("=" * 55)

# XOR dataset — cannot be solved by a single perceptron
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

model_xor = MLP([2, 4, 1], ["relu", "sigmoid"])

losses_xor = []
lr = 0.1
for epoch in range(5000):
    # Forward
    y_pred = model_xor.forward(X_xor)
    loss = mse_loss(y_pred, y_xor)
    losses_xor.append(loss)

    # Backward
    dLoss = mse_loss_deriv(y_pred, y_xor)
    model_xor.backward(dLoss)
    model_xor.update(lr)

    if (epoch + 1) % 1000 == 0:
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.6f}")

preds_xor = model_xor.predict(X_xor)
print(f"\nXOR Predictions (rounded):")
for i, (x, y, p) in enumerate(zip(X_xor, y_xor, preds_xor)):
    print(f"  Input {x.astype(int)} → Target: {int(y[0])} | Pred: {p[0]:.4f} | Correct: {round(p[0]) == int(y[0])}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Training — Classification Example
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("DEMO 2: 2-Class Classification (synthetic circles)")
print("=" * 55)

# Generate synthetic 2-class data
N = 200
np.random.seed(0)
X_class = np.vstack([
    np.random.randn(N//2, 2) + np.array([2, 2]),
    np.random.randn(N//2, 2) + np.array([-2, -2]),
])
y_class = np.array([0]*(N//2) + [1]*(N//2))

model_cls = MLP([2, 8, 8, 2], ["relu", "relu", "linear"])

lr = 0.01
for epoch in range(2000):
    # Forward
    logits = model_cls.forward(X_class)
    probs = softmax(logits)
    loss = cross_entropy_loss(probs, y_class)

    # Backprop: dL/dlogits for cross-entropy + softmax combined
    dLogits = probs.copy()
    dLogits[np.arange(N), y_class] -= 1

    model_cls.backward(dLogits)
    model_cls.update(lr)

    if (epoch + 1) % 500 == 0:
        preds = np.argmax(softmax(model_cls.predict(X_class)), axis=1)
        acc = (preds == y_class).mean() * 100
        print(f"  Epoch {epoch+1:5d} | Loss: {loss:.4f} | Accuracy: {acc:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Key Concepts Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("KEY CONCEPTS RECAP")
print("=" * 55)
concepts = [
    ("Forward Pass",    "Input → Linear transform (W@X+b) → Activation → Output"),
    ("Loss",            "Measures how wrong predictions are (MSE, CrossEntropy)"),
    ("Backpropagation", "Chain rule: propagate dLoss/dOutput backwards layer-by-layer"),
    ("Gradient Descent","W = W - lr * dW  (nudge weights in direction that lowers loss)"),
    ("Activation Fn",   "Adds non-linearity; without it, MLP = single linear map"),
    ("He Init",         "scale = sqrt(2/fan_in); prevents vanishing/exploding gradients"),
]
for name, desc in concepts:
    print(f"  {name:20s}: {desc}")

print("\n[Weeks 2-3 complete] Neural networks from scratch implemented.")

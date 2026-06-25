"""
Week 4 — Backpropagation Deep Dive + Intro to CNNs
SoC: Build Your Own ChatGPT from Scratch
Based on: StatQuest playlist videos 86–91

Covers:
  - Computational graph & chain rule (why backprop works)
  - Gradient checking (verify numerical == analytical gradients)
  - Batch Normalization from scratch
  - Weight initialization strategies
  - Adam optimizer from scratch
  - Intro to vanishing/exploding gradients
"""

import numpy as np

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Computational Graph & Chain Rule
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Computational Graph Walkthrough")
print("=" * 60)

# Manual forward + backward for: L = (w*x + b - y)^2
# This is the simplest possible regression neuron.

x, y_true = 2.0, 5.0
w, b = 0.5, 1.0

# Forward pass — track every intermediate value
z = w * x + b       # linear: z = wx + b
loss = (z - y_true) ** 2   # MSE for 1 sample

print(f"w={w}, x={x}, b={b}")
print(f"z = w*x + b = {z}")
print(f"loss = (z - y)^2 = ({z} - {y_true})^2 = {loss}")

# Backward pass — apply chain rule manually
dL_dz   = 2 * (z - y_true)         # dL/dz
dL_dw   = dL_dz * x                # dL/dw = dL/dz * dz/dw = dL/dz * x
dL_db   = dL_dz * 1                # dL/db = dL/dz * dz/db = dL/dz * 1

print(f"\nGradients:")
print(f"  dL/dz = {dL_dz:.4f}")
print(f"  dL/dw = dL/dz * x = {dL_dw:.4f}")
print(f"  dL/db = dL/dz * 1 = {dL_db:.4f}")

# Gradient descent step
lr = 0.01
w_new = w - lr * dL_dw
b_new = b - lr * dL_db
print(f"\nAfter one GD step: w={w_new:.4f}, b={b_new:.4f}")
print(f"New prediction: {w_new * x + b_new:.4f}  (closer to {y_true})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Gradient Checking — Numerical vs Analytical
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Gradient Checking")
print("=" * 60)
print("Verifies backprop is correct by comparing to numerical gradient.")

def forward(params, X, y):
    """Simple 2-layer net forward pass returning loss."""
    W1, b1, W2, b2 = params
    h = np.tanh(X @ W1 + b1)
    out = h @ W2 + b2
    loss = ((out - y) ** 2).mean()
    return loss

def flatten_params(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

def unflatten_params(p, shapes):
    params = []
    idx = 0
    for s in shapes:
        size = np.prod(s)
        params.append(p[idx:idx+size].reshape(s))
        idx += size
    return params

# Small network for checking
n_in, n_h, n_out = 3, 4, 1
W1 = np.random.randn(n_in, n_h) * 0.1
b1 = np.zeros((1, n_h))
W2 = np.random.randn(n_h, n_out) * 0.1
b2 = np.zeros((1, n_out))

X_chk = np.random.randn(5, n_in)
y_chk = np.random.randn(5, n_out)

shapes = [(n_in, n_h), (1, n_h), (n_h, n_out), (1, n_out)]
p = flatten_params(W1, b1, W2, b2)

# Numerical gradient
eps = 1e-5
grad_numerical = np.zeros_like(p)
for i in range(len(p)):
    p_plus = p.copy();  p_plus[i] += eps
    p_minus = p.copy(); p_minus[i] -= eps
    f_plus  = forward(unflatten_params(p_plus,  shapes), X_chk, y_chk)
    f_minus = forward(unflatten_params(p_minus, shapes), X_chk, y_chk)
    grad_numerical[i] = (f_plus - f_minus) / (2 * eps)

# Analytical gradient (manual backprop)
W1_, b1_, W2_, b2_ = unflatten_params(p, shapes)
h = np.tanh(X_chk @ W1_ + b1_)
out_ = h @ W2_ + b2_
dout = 2 * (out_ - y_chk) / y_chk.shape[0]
dW2 = h.T @ dout / X_chk.shape[0]; db2 = dout.mean(axis=0, keepdims=True)
dh = dout @ W2_.T
dh_pre = (1 - h**2) * dh   # tanh derivative
dW1 = X_chk.T @ dh_pre / X_chk.shape[0]; db1 = dh_pre.mean(axis=0, keepdims=True)

grad_analytical = flatten_params(dW1, db1, dW2, db2)

# Compare
diff = np.abs(grad_numerical - grad_analytical).max()
rel_err = np.linalg.norm(grad_analytical - grad_numerical) / (np.linalg.norm(grad_analytical) + np.linalg.norm(grad_numerical))
print(f"Max absolute error:   {diff:.2e}  (should be < 1e-5)")
print(f"Relative error:       {rel_err:.2e}  (should be < 1e-7)")
print(f"Gradient check: {'✓ PASSED' if rel_err < 1e-5 else '✗ FAILED'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Batch Normalization
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Batch Normalization from Scratch")
print("=" * 60)
print("Normalizes activations within a mini-batch; stabilizes training.")

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones((1, num_features))   # learnable scale
        self.beta  = np.zeros((1, num_features))  # learnable shift
        self.eps = eps
        self.momentum = momentum
        # Running stats for inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var  = np.ones((1, num_features))
        # Cache
        self.x_norm = None; self.var = None; self.mean = None; self.x = None

    def forward(self, x, training=True):
        if training:
            self.mean = x.mean(axis=0, keepdims=True)
            self.var  = x.var(axis=0, keepdims=True)
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.x = x
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, d_out):
        N = self.x.shape[0]
        d_xnorm = d_out * self.gamma
        d_var  = (-0.5 * d_xnorm * (self.x - self.mean) * (self.var + self.eps)**(-1.5)).sum(axis=0, keepdims=True)
        d_mean = (-d_xnorm / np.sqrt(self.var + self.eps)).sum(axis=0, keepdims=True) + d_var * (-2 * (self.x - self.mean)).mean(axis=0, keepdims=True)
        dx = d_xnorm / np.sqrt(self.var + self.eps) + d_var * 2 * (self.x - self.mean) / N + d_mean / N
        self.dgamma = (d_out * self.x_norm).sum(axis=0, keepdims=True)
        self.dbeta  = d_out.sum(axis=0, keepdims=True)
        return dx

X_bn = np.random.randn(32, 8) * 5 + 3   # unnormalized activations
bn = BatchNorm(8)
X_normed = bn.forward(X_bn)
print(f"Before BN — mean: {X_bn.mean():.2f}, std: {X_bn.std():.2f}")
print(f"After  BN — mean: {X_normed.mean():.4f}, std: {X_normed.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Adam Optimizer from Scratch
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Adam Optimizer")
print("=" * 60)
print("Adam = Adaptive Moment Estimation (used in GPT training)")

class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}   # 1st moment (mean of gradients)
        self.v = {}   # 2nd moment (mean of squared gradients)
        self.t = 0    # time step

    def step(self, params_and_grads):
        """params_and_grads: list of (param_array, grad_array) tuples"""
        self.t += 1
        for i, (param, grad) in enumerate(params_and_grads):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Demo: Adam vs plain SGD on Rosenbrock function
def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1 - x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

p_sgd  = np.array([-1.0, 1.0])
p_adam = np.array([-1.0, 1.0])
adam = AdamOptimizer(lr=0.01)
lr_sgd = 0.001

losses_sgd, losses_adam = [], []
for step in range(500):
    g = rosenbrock_grad(*p_sgd)
    p_sgd -= lr_sgd * g
    losses_sgd.append(rosenbrock(*p_sgd))

    g = rosenbrock_grad(*p_adam)
    adam.step([(p_adam, g)])
    losses_adam.append(rosenbrock(*p_adam))

print(f"After 500 steps (Rosenbrock, optimum at (1,1)):")
print(f"  SGD  — final point: ({p_sgd[0]:.4f}, {p_sgd[1]:.4f}), loss: {losses_sgd[-1]:.4f}")
print(f"  Adam — final point: ({p_adam[0]:.4f}, {p_adam[1]:.4f}), loss: {losses_adam[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Vanishing / Exploding Gradients
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Vanishing & Exploding Gradients")
print("=" * 60)

def gradient_flow_demo(n_layers, init_scale, activation_deriv_val):
    """
    Shows how gradient magnitudes change through a deep network.
    activation_deriv_val: max derivative of activation (0.25 for sigmoid, 1.0 for ReLU)
    """
    grad = 1.0
    grads = [grad]
    for _ in range(n_layers):
        grad *= init_scale * activation_deriv_val
        grads.append(grad)
    return grads

print("Gradient at each layer (deeper → earlier), 10-layer net:")
print(f"\n  Sigmoid (deriv≈0.25), W~N(0,1)  — VANISHING:")
g = gradient_flow_demo(10, 1.0, 0.25)
for i, v in enumerate(g):
    bar = "█" * min(40, int(abs(v) * 40))
    print(f"    Layer {i:2d}: {v:.6f}  {bar}")

print(f"\n  ReLU, W~N(0,2) (He init)  — STABLE:")
g = gradient_flow_demo(10, np.sqrt(2), 1.0)
for i, v in enumerate(g):
    bar = "█" * min(40, int(abs(v) * 5))
    print(f"    Layer {i:2d}: {v:.4f}  {bar}")


print("\n" + "=" * 60)
print("KEY TAKEAWAYS — Week 4")
print("=" * 60)
takeaways = [
    "Backprop = chain rule applied to a computational graph",
    "Always gradient-check custom implementations",
    "Batch Norm stabilises training by keeping activations ~N(0,1)",
    "Adam adapts lr per-parameter using moment estimates",
    "Vanishing gradients killed deep sigmoid nets → ReLU + He init fixed it",
    "These same issues motivated LayerNorm + residual connections in Transformers",
]
for i, t in enumerate(takeaways, 1):
    print(f"  {i}. {t}")

print("\n[Week 4 complete] Backprop deep dive done.")

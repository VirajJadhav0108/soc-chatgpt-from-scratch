# soc-chatgpt-from-scratch
Seasons of Code | Building GPT from Scratch using Python, NumPy &amp; PyTorch
# 🧠 Build Your Own ChatGPT from Scratch — SoC 2025

> **IIT Bombay Seasons of Code (SoC) 2025**  
> Project: *Build Your Own ChatGPT from Scratch*  
> Mentors: Ahoy-Codey  
> Participant: Viraj | B.Tech Civil Engineering (Minor: ML/AI), IIT Bombay

---

## 📌 Project Overview

This repository tracks my progress through the SoC "Build Your Own ChatGPT from Scratch" project. The goal is to build a GPT-style language model from first principles — starting from Python/NumPy basics, moving through neural networks, and ultimately implementing a character-level transformer in PyTorch (Week 6).

**Resources:** [Ahoy-Codey/ChatGPT-from-Scratch-Resource](https://github.com/Ahoy-Codey/ChatGPT-from-Scratch-Resource)

---

## 📅 Progress Tracker

| Week | Topic | Status |
|------|-------|--------|
| 1 | Python, NumPy, Pandas, Matplotlib | ✅ Complete |
| 2 | Neural Networks (StatQuest 74–84) | ✅ Complete |
| 3 | Backpropagation & Training Loops | ✅ Complete |
| 4 | Advanced Backprop, Adam, BatchNorm | ✅ Complete |
| 5 | PyTorch Introduction | 🔄 In Progress |
| 6 | GPT from Scratch (Karpathy) | ⏳ Upcoming |
| 7-8 | Fine-tuning / Buffer Weeks | ⏳ Upcoming |

---

## 📁 Repository Structure

```
.
├── week1/
│   ├── numpy_basics.py          # Arrays, broadcasting, matmul, softmax
│   └── pandas_basics.py         # DataFrames, feature engineering, vocab building
│
├── week2_3/
│   └── neural_network_scratch.py  # MLP from scratch, XOR, classification
│
├── week4/
│   └── backprop_deep_dive.py    # Chain rule, grad check, BatchNorm, Adam
│
└── README.md
```

---

## 🗂️ Week-by-Week Notes

### Week 1 — Data Science Toolkit
- **NumPy:** Array operations, reshaping, broadcasting, matrix multiplication, manual softmax
- **Pandas:** DataFrames, filtering, groupby, handling missing values, building a toy vocabulary
- **Key insight:** Broadcasting + matmul in NumPy is exactly what happens inside every linear layer and attention head

### Week 2 & 3 — Neural Networks from Scratch
Implemented a fully working MLP (no frameworks) with:
- Forward pass: `X → Linear → Activation → Output`
- He / Xavier weight initialization
- Sigmoid, ReLU, Tanh activations + their derivatives
- MSE and Cross-Entropy loss
- Manual backpropagation via chain rule
- **Solved XOR** (the classic test for non-linear networks)
- 2-class classification on synthetic data reaching >95% accuracy

### Week 4 — Backpropagation Deep Dive
- **Computational graph walkthrough** — traced gradients for `L = (wx+b-y)²` by hand
- **Gradient checking** — verified analytical gradients match numerical gradients (relative error < 1e-7) ✓
- **Batch Normalization** — implemented forward + backward pass; showed it centres activations to ~N(0,1)
- **Adam Optimizer** — implemented from scratch with bias correction; benchmarked vs SGD on Rosenbrock function
- **Vanishing gradient analysis** — showed how sigmoid + random init kills gradients in deep nets, and why ReLU + He init fixes it (directly motivates LayerNorm + residuals in Transformers)

---

## 🔑 Key Concepts Mastered (Weeks 1–4)

| Concept | How it connects to GPT |
|---------|------------------------|
| Matrix multiplication | Every linear projection (Q, K, V matrices) |
| Softmax | Attention score normalisation |
| Backpropagation | How GPT learns from data |
| Adam optimizer | Default optimizer used to train GPT-2/3 |
| Batch Normalization | Precursor to Layer Normalization in Transformers |
| Vanishing gradients | Motivates residual connections in Transformer blocks |
| Vocabulary / tokenization | Foundation of text → token → embedding pipeline |

---

## 🛠️ Setup & Running

```bash
# Clone the repo
git clone https://github.com/<your-username>/soc-chatgpt-from-scratch.git
cd soc-chatgpt-from-scratch

# Install dependencies (just NumPy and Pandas for Weeks 1–4)
pip install numpy pandas

# Run any week's script
python week1/numpy_basics.py
python week1/pandas_basics.py
python week2_3/neural_network_scratch.py
python week4/backprop_deep_dive.py
```

---

## 🚧 Challenges Faced

1. **Time constraints** — Simultaneously running two internships (GiftsBazaar + Hey Recruiting) and a research internship (IIT Bombay Groundwater ML), which compressed study time significantly
2. **Gradient debugging** — Initial backprop implementation had sign errors in the MSE derivative; gradient checking caught it
3. **Numerical stability** — Softmax and cross-entropy required adding numerical stability tricks (subtracting max, clipping log args) to avoid NaN/Inf
4. **Adam bias correction** — Easy to forget the `(1 - β^t)` bias correction in early Adam steps

---

## 📚 References

- [StatQuest Neural Networks Playlist](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) (Videos 74–91)
- [Andrej Karpathy — makemore / nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Week 6 target)
- [SoC Resource Repo](https://github.com/Ahoy-Codey/ChatGPT-from-Scratch-Resource)

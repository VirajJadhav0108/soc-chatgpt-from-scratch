# 🧠 Build Your Own ChatGPT from Scratch — SoC

> **Seasons of Code (SoC)**
> Project: *Build Your Own ChatGPT from Scratch*

---

## 📌 Project Overview

This repository documents my journey through the Seasons of Code project **"Build Your Own ChatGPT from Scratch."** The objective is to understand and implement the core components of modern language models from first principles, beginning with Python and NumPy fundamentals, progressing through neural networks, and culminating in a character-level GPT built entirely in PyTorch.

Rather than relying on high-level libraries, this project focuses on understanding the mathematics and implementation behind every major component of a GPT-style model.

**Resources:** https://github.com/Ahoy-Codey/ChatGPT-from-Scratch-Resource

---

## 📅 Progress Tracker

| Week | Topic                                          | Status      |
| ---- | ---------------------------------------------- | ----------- |
| 1    | Python, NumPy, Pandas, Matplotlib              | ✅ Complete  |
| 2    | Neural Networks Fundamentals                   | ✅ Complete  |
| 3    | Backpropagation & Training Loops               | ✅ Complete  |
| 4    | Advanced Backpropagation & Optimization        | ✅ Complete  |
| 5    | PyTorch Fundamentals                           | ✅ Complete  |
| 6    | GPT from Scratch (Character-Level Transformer) | ✅ Complete  |

---

# 📁 Repository Structure

```text
.
├── Week1/
│   ├── numpy_basics.py
│   └── pandas_basics.py
│
├── Week2_3/
│   └── neural_network_scratch.py
│
├── Week4/
│   └── backprop_deep_dive.py
│
├── Week5_PyTorch/
│   ├── 01_tensors.py
│   ├── 02_autograd.py
│   ├── 03_nn_module.py
│   ├── 04_dataloader.py
│   ├── 05_training.py
│   └── requirements.txt
│
├── Week6_GPT/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── input.txt
│   └── checkpoints/
│
└── README.md
```

---

# 🗂️ Week-by-Week Notes

## Week 1 — Python & Data Science Fundamentals

* Python refresher
* NumPy array operations
* Matrix multiplication
* Broadcasting
* Pandas DataFrames
* Data preprocessing
* Data visualization using Matplotlib

**Key Takeaway:** Matrix operations and broadcasting form the mathematical foundation of neural networks and transformers.

---

## Week 2 & 3 — Neural Networks from Scratch

Implemented a complete multilayer perceptron without deep learning frameworks.

Covered:

* Forward propagation
* Backpropagation
* Chain Rule
* Gradient Descent
* Weight Initialization
* ReLU, Sigmoid and Tanh
* Binary Classification
* XOR Problem
* Cross Entropy Loss
* Mean Squared Error

**Outcome**

* Built a neural network entirely from scratch.
* Understood how gradients flow through computational graphs.

---

## Week 4 — Deep Learning Fundamentals

Implemented and explored

* Computational Graphs
* Gradient Checking
* Batch Normalization
* Adam Optimizer
* Vanishing & Exploding Gradients
* Numerical Stability
* Optimization Techniques

**Outcome**

Developed a deeper understanding of optimization methods that modern transformer architectures rely on.

---

## Week 5 — PyTorch Fundamentals

Implemented multiple PyTorch programs covering

* Tensor creation and manipulation
* Automatic differentiation (Autograd)
* Neural network construction using `torch.nn`
* Dataset and DataLoader APIs
* Complete model training pipeline
* Model saving and loading
* GPU acceleration

**Files**

* `01_tensors.py`
* `02_autograd.py`
* `03_nn_module.py`
* `04_dataloader.py`
* `05_training.py`

**Outcome**

Gained practical experience with PyTorch and prepared for implementing a transformer from scratch.

---

## Week 6 — GPT From Scratch

Built a character-level GPT language model in PyTorch.

Implemented

* Character-level tokenizer
* Vocabulary construction
* Token & positional embeddings
* Scaled Dot-Product Self-Attention
* Multi-Head Attention
* Feed Forward Networks
* Residual Connections
* Layer Normalization
* Transformer Blocks
* GPT Language Model
* Training Pipeline
* Text Generation
* Model Checkpointing

**Files**

* `config.py`
* `dataset.py`
* `model.py`
* `train.py`
* `generate.py`
* `utils.py`

The model is trained on the Tiny Shakespeare dataset and can generate coherent character-level English text after training.

---

# 🔑 Concepts Learned

| Concept                | Application in GPT                               |
| ---------------------- | ------------------------------------------------ |
| Matrix Multiplication  | Linear projections for Query, Key and Value      |
| Softmax                | Attention probability distribution               |
| Embeddings             | Convert tokens into dense vector representations |
| Backpropagation        | Model parameter optimization                     |
| Adam Optimizer         | Efficient transformer training                   |
| Layer Normalization    | Stable deep transformer training                 |
| Residual Connections   | Better gradient flow                             |
| Self-Attention         | Context-aware token representations              |
| Multi-Head Attention   | Parallel attention mechanisms                    |
| Positional Embeddings  | Encode sequence order                            |
| Transformer Blocks     | Core GPT architecture                            |
| Character Tokenization | Language modelling pipeline                      |

---

# 🛠️ Running the Project

Clone the repository

```bash
git clone https://github.com/VirajJadhav0108/soc-chatgpt-from-scratch.git
cd soc-chatgpt-from-scratch
```

Install dependencies

```bash
pip install -r Week5_PyTorch/requirements.txt
pip install -r Week6_GPT/requirements.txt
```

Train the GPT model

```bash
cd Week6_GPT
python train.py
```

Generate text

```bash
python generate.py
```

---

# 🚧 Challenges Faced

* Understanding gradient flow without relying on high-level frameworks.
* Debugging numerical instability during optimization.
* Implementing attention mechanisms from first principles.
* Managing tensor dimensions throughout transformer layers.
* Building an end-to-end GPT pipeline including tokenization, training and inference.

---

# 📚 References

* SoC Resource Repository: https://github.com/Ahoy-Codey/ChatGPT-from-Scratch-Resource
* Andrej Karpathy — Let's Build GPT: https://www.youtube.com/watch?v=kCc8FmEb1nY
* PyTorch Documentation: https://pytorch.org/docs/stable/
* NumPy Documentation: https://numpy.org/doc/
* Pandas Documentation: https://pandas.pydata.org/docs/

---

## 🎯 Final Outcome

By the end of Week 6, this project progresses from basic Python programming to a complete character-level GPT implementation in PyTorch. It provides a practical understanding of how modern transformer-based language models are built, trained, and used for text generation from first principles.

import os
import torch
import torch.optim as optim

from config import *
from dataset import get_batch, estimate_loss, vocab_size
from model import GPTLanguageModel

os.makedirs("checkpoints", exist_ok=True)

model = GPTLanguageModel(vocab_size).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print(f"Using device: {device}")
print(f"Vocabulary Size: {vocab_size}")

for iteration in range(max_iters):

    if iteration % eval_interval == 0 or iteration == max_iters - 1:

        losses = estimate_loss(model)

        print(
            f"Step {iteration:5d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": iteration
            },
            checkpoint_path
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

torch.save(model.state_dict(), "checkpoints/final_model.pt")

print("\nTraining Complete.")
print("Final model saved to checkpoints/final_model.pt")

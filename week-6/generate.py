import torch

from config import *
from dataset import decode, stoi, vocab_size
from model import GPTLanguageModel

model = GPTLanguageModel(vocab_size).to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

start_text = input("Enter prompt: ")

if start_text.strip() == "":
    start_text = "\n"

context = torch.tensor(
    [[stoi.get(ch, 0) for ch in start_text]],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=500
)

output = decode(
    generated[0].tolist()
)

print("\n" + "=" * 60)
print(output)
print("=" * 60)

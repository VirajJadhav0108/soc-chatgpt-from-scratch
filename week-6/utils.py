import os
import random
import numpy as np
import torch

from config import seed


def set_seed():

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def save_checkpoint(model, optimizer, step, path):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        path
    )


def load_checkpoint(model, optimizer, path, device):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint["step"]


@torch.no_grad()
def generate_text(
    model,
    stoi,
    itos,
    prompt,
    max_new_tokens,
    device
):

    model.eval()

    context = torch.tensor(
        [[stoi.get(ch, 0) for ch in prompt]],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        context,
        max_new_tokens=max_new_tokens
    )[0].tolist()

    return "".join(
        itos[index]
        for index in output
    )

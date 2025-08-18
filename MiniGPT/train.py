from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from .config import MiniGPTConfig
from .model import MiniGPTModel
from .tokenizer import TikTokenizer
from .data import batch_loader

def train_model(
    raw_dataset: str | list[str],
    out_dir: str = "checkpoints/minigpt",
    config: MiniGPTConfig = MiniGPTConfig(),
    epochs: int = 1000,
    seq_len: int = 64,
    batch_size: int = 16,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sample_every: int = 100,
    sample_prompt: str = "The Emperor",
):
    tokenizer = TikTokenizer("gpt2")
    model = MiniGPTModel(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    it = trange(1, epochs + 1, desc="training")
    for epoch in it:
        model.train()
        x_batch, y_batch = batch_loader(raw_dataset, tokenizer, T=seq_len, B=batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)  # [B, T, V]
        loss = criterion(logits.view(-1, config.vocab_size), y_batch.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        it.set_postfix(loss=float(loss.item()))

        if sample_every and (epoch % sample_every == 0):
            from .generate import generate
            txt = generate(model, sample_prompt, tokenizer, max_tokens=50, temperature=1.0, device=device)
            print("\n--- sample ---\n", txt[:300], "\n--------------\n")

    # save
    out = Path(out_dir)
    model.save_pretrained(out)
    print(f"Saved to: {out.resolve()}")

    return model

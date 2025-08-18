from __future__ import annotations
import torch
from .tokenizer import TikTokenizer

@torch.no_grad()
def generate(
    model,
    start_text: str,
    tokenizer: TikTokenizer,
    max_tokens: int = 50,
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    
    
    model.eval()
    x = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        logits = model(x)             # [1, T, V]
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs  = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        x = torch.cat([x, next_token], dim=1)

        if next_token.item() == tokenizer.eot_token_id:
            break

    return tokenizer.decode(x[0].tolist())

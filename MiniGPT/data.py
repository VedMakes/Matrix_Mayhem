from __future__ import annotations
import torch
from torch import Tensor
from typing import Iterable
from .tokenizer import TikTokenizer

def build_ids(
    data: str | Iterable[str],
    tokenizer: TikTokenizer,
    add_eot: bool = True,
) -> Tensor:
    """
    Returns a flat 1D LongTensor of token ids.
    Appends <|endoftext|> between items when add_eot=True.
    """
    if isinstance(data, str):
        txts = [data]
    else:
        txts = list(data)

    buf: list[int] = []
    for s in txts:
        buf.extend(tokenizer.encode(s))
        if add_eot:
            buf.append(tokenizer.eot_token_id)
    return torch.tensor(buf, dtype=torch.long)

@torch.no_grad()
def batch_loader(
    raw_dataset: str | Iterable[str],
    tokenizer: TikTokenizer,
    T: int = 64,
    B: int = 8,
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    
    """
    Makes random next-token batches.
    Returns x,y each [B, T] on device.
    """
    
    ids = build_ids(raw_dataset, tokenizer=tokenizer, add_eot=True)
    N = ids.numel()
    if N <= T + 1:
        raise ValueError(f"Need > {T+1} tokens, got {N}.")

    # sample B starting positions
    i = torch.randint(0, N - T - 1, (B,))
    x = torch.stack([ids[j:j+T]     for j in i], dim=0)
    y = torch.stack([ids[j+1:j+T+1] for j in i], dim=0)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

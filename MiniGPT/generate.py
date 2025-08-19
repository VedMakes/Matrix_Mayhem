from __future__ import annotations
import torch
from .tokenizer import TikTokenizer

@torch.no_grad()
def generate(
    model,
    start_text: str,
    tokenizer: TikTokenizer,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int= 15, 
    top_p : float= 0.9, 
    repetition_penalty = 1.5,
    device: str = "cuda",
) -> str:
    model.eval()
    
    # Encode starting text
    x = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)  # [1, seq_len]
    
    for _ in range(max_tokens):
        logits = model(x)  # [1, seq_len, vocab_size]
        logits = logits[:, -1, :] / temperature   # last token’s logits, scaled

        for token_id in set(x[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty
        
        # --- Top-K filtering ---
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))  # safety
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)  # cutoff threshold
            logits = torch.where(logits < min_val, torch.full_like(logits, -float("Inf")), logits)
        
        # --- Top-P (nucleus) filtering ---
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            
            # Mask out tokens above nucleus probability
            mask = cumulative_probs > top_p
            
            # Shift mask right to keep at least one token
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            
            sorted_logits[mask] = -float("Inf")
            # Re-map back to original indices
            logits = torch.full_like(logits, -float("Inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)
        
        # Turn logits into probabilities
        probs = torch.softmax(logits, dim=-1)    
        
        # Sample from distribution
        next_token = torch.multinomial(probs, num_samples=1)  
        
        x = torch.cat([x, next_token], dim=1)  # append to sequence

        # Stop if we hit <|endoftext|>
        if next_token.item() == tokenizer.eot_token:
            break

    # Decode back to text
    return tokenizer.decode(x[0].tolist())

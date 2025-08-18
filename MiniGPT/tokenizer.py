from __future__ import annotations
import tiktoken as tk

class TikTokenizer:
    
    """
    Thin wrapper around tiktoken GPT-2 BPE.
    """
    
    def __init__(self, name: str = "gpt2"):
        self._enc = tk.get_encoding(name)
        # EOT token id for GPT-2 is 50256
        self.eot_token_id = 50256
        self.allowed_special = {"<|endoftext|>"}

    def encode(self, s: str) -> list[int]:
        return self._enc.encode(s, allowed_special=self.allowed_special)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

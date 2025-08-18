from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class MiniGPTConfig:
    vocab_size: int = 50257
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    num_layers: int = 4
    max_len: int = 256
    dropout: float = 0.1

    @staticmethod
    def from_json(path: str | Path) -> "MiniGPTConfig":
        with open(path, "r") as f:
            return MiniGPTConfig(**json.load(f))

    def to_json(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

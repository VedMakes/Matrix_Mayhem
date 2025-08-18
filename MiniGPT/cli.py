from __future__ import annotations
import argparse
from pathlib import Path
import torch
from .config import MiniGPTConfig
from .model import MiniGPTModel
from .tokenizer import TikTokenizer
from .train import train_model
from .generate import generate

def main():
    parser = argparse.ArgumentParser(prog="minigpt", description="Mini GPT CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a mini GPT on a text file")
    p_train.add_argument("--data", type=str, required=True, help="Path to a text file (or folder with .txt)")
    p_train.add_argument("--out_dir", type=str, default="checkpoints/minigpt")
    p_train.add_argument("--vocab_size", type=int, default=50257)
    p_train.add_argument("--d_model", type=int, default=256)
    p_train.add_argument("--num_heads", type=int, default=8)
    p_train.add_argument("--d_ff", type=int, default=1024)
    p_train.add_argument("--num_layers", type=int, default=4)
    p_train.add_argument("--max_len", type=int, default=256)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--epochs", type=int, default=1000)
    p_train.add_argument("--seq_len", type=int, default=64)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--sample_every", type=int, default=100)
    p_train.add_argument("--sample_prompt", type=str, default="The Emperor")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text with a trained checkpoint")
    p_gen.add_argument("--ckpt", type=str, required=True, help="Path to folder with config.json & pytorch_model.bin")
    p_gen.add_argument("--prompt", type=str, required=True)
    p_gen.add_argument("--max_tokens", type=int, default=100)
    p_gen.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "train":
        # load data (single text file or concat all *.txt in folder)
        data_path = Path(args.data)
        if data_path.is_dir():
            parts = []
            for p in sorted(data_path.glob("*.txt")):
                parts.append(p.read_text(encoding="utf-8"))
            raw_dataset = "\n".join(parts)
        else:
            raw_dataset = data_path.read_text(encoding="utf-8")

        cfg = MiniGPTConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout=args.dropout,
        )

        train_model(
            raw_dataset=raw_dataset,
            out_dir=args.out_dir,
            config=cfg,
            epochs=args.epochs,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            sample_every=args.sample_every,
            sample_prompt=args.sample_prompt,
        )

    elif args.cmd == "generate":
        tok = TikTokenizer("gpt2")
        model = MiniGPTModel.from_pretrained(args.ckpt).to(device)
        text = generate(
            model, args.prompt, tok,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device
        )
        print(text)

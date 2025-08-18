# Matrix_Mayhem
Notebooks and modular codes for the IIIT-BH 2025 "Matrix Mayhem" 

## Mini-GPT Trainer
A lightweight GPT-style transformer implemented in PyTorch for training on Warhammer-themed text datasets. Includes both a modular Python script for CLI training and a Jupyter notebook version for interactive experimentation.

### Features
- GPT-style transformer built from scratch in PyTorch
- Multi-head attention with residual connections and layer norm
- Feed-forward layers with GELU nonlinearity and dropout
- Custom tokenizer using GPT-2 encoding (tiktoken)
- Text generation with temperature-controlled sampling
- Easy to train on your own text datasets

---

### Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Notebook Usage](#notebook-usage)  
4. [Script Usage](#script-usage)  
5. [Hyperparameters](#hyperparameters)  
6. [License](#license)  

---

### Requirements

- Python 3.9+  
- PyTorch 2.x  
- tiktoken  
- Optional: GPU with CUDA for faster training 

### Notebook Usage

1) Open mini_gpt_notebook.ipynb in Jupyter Notebook or JupyterLab.
2) Execute cells sequentially:
  - Tokenizer setup (enc, dec)
  - Model components (Embedding, MHA, FeedForward, Transformer)
  - Training loop
  - Text generation
3) Load your dataset as a string and pass it to the training loop:


```python
with open("raw_dataset.txt") as f:
    raw_dataset = f.read()

train(model, dataset, epochs=5, batch_size=8, seq_len=64, device="cuda")

sample_text = generate(model, start_text="The Emperor", max_tokens=100)
print(sample_text)
```

### Default Hyperparameters

| Parameter      | Default Value | Description                                           |
|----------------|---------------|-------------------------------------------------------|
| `d_model`      | 256           | Embedding dimension / model hidden size               |
| `num_heads`    | 8             | Number of attention heads                             |
| `head_dim`     | 32            | Dimension per attention head (`d_model / num_heads`)  |
| `d_ff`         | 1024          | Hidden size of feed-forward network                   |
| `num_layers`   | 4             | Number of Transformer blocks                          |
| `max_len`      | 256           | Maximum context length / positional embedding size    |
| `vocab_size`   | 50257         | GPT-2 tokenizer vocabulary size                       |
| `dropout`      | 0.1           | Dropout rate in feed-forward layers                   |
| `seq_len`      | 64            | Sequence length for training batches                  |
| `batch_size`   | 8             | Number of sequences per batch                         |
| `learning_rate`| 3e-4          | Adam optimizer learning rate                          |
| `temperature`  | 1.0           | Sampling temperature for text generation              |

### License
MIT License – free to use, modify, and share.

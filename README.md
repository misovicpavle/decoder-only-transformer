# Small Decoder-Only Transformers (From Scratch): 4 Tokenizers, 2 Architectures, 2 Languages

I built this project to really understand what's happening inside a decoder-only transformer instead of only using high-level libraries.

It started as a simple comparison between a character-level model and a SentencePiece subword model. Since then it has grown into a full experiment matrix:

Two languages: English and Serbian (Cyrillic).  
Four tokenizers: char-level, SentencePiece 10k, HuggingFace BPE 10k and tiktoken o200k_base.  
Two architectures: my own hand-written transformer block ("old") and PyTorch's `nn.TransformerEncoder` with a causal mask ("nn").

That's 16 runs in total (2 × 4 × 2), each with its own log.

All of them were trained in Google Colab on a T4 GPU.

---

## Why I made this

I wanted to answer a few practical questions for myself:

How different do training dynamics look when tokenization changes but the core architecture stays the same?  
Does a tokenizer trained on my own corpus beat a huge pretrained one like tiktoken?  
Does a model behave differently on Serbian than on English?  
Is my hand-written transformer block actually as good as PyTorch's built-in one?  
How much do small details like weight tying and AdamW vs Adam matter on a model this small?

This repo is my hands-on benchmark and learning record.

---

## What's in this project

Notebook implementations for every combination of:

- Language: ENG / SRB
- Tokenizer: char / SentencePiece / HF BPE / tiktoken
- Architecture: old (hand-written `TransformerBlock`) / nn (`nn.TransformerEncoder`)

There's also a presentation (`transformer-experiments.pptx`) with all the loss curves, the comparisons and the conclusions.

The "old" models use my custom decoder-only pipeline in PyTorch:
- Token embedding
- Learned positional embedding
- Stacked masked self-attention + feed-forward transformer blocks
- Residual connections + layer norm + dropout
- Linear LM head for next-token prediction

The "nn" models swap the hand-written blocks for `nn.TransformerEncoder` and use `generate_square_subsequent_mask(T)` as the causal mask. That might sound strange for a decoder, but `nn.TransformerDecoder` expects a second input (memory from an encoder for cross-attention), which a GPT-style model doesn't have. An encoder with a causal mask computes exactly what a decoder-only transformer needs: token t sees tokens 1…t and never t+1…T.

Training setup includes:
- Chunk shuffle before split
- Train/val split (10% of the corpus for validation)
- Gradient clipping
- Checkpoint save and resume
- Periodic train/val loss logging

---

## Tokenization setups

### Character-based model
- Vocabulary is built from unique characters in the corpus (~100 symbols).
- Encoding and decoding are direct char-to-id and id-to-char mappings.
- Advantage: Very transparent and easy to reason about.
- Limitation: Longer token sequences for the same text.

### SentencePiece (unigram, 10k)
- Trained on my own corpus, so every token actually appears in the text.
- Starts from whole words and splits them into frequent subwords.
- Advantage: Small vocabulary → small embedding (2.6 M params) and a model of ~10 M.
- Limitation: Extra tokenizer training step.

### HuggingFace BPE (byte-level, 10k)
- Byte-level BPE trained on my corpus with the HF `tokenizers` library.
- Same vocabulary size as SentencePiece, so the two compare directly.
- Advantage: Turned out to be the best word-level tokenizer on English.

### tiktoken (o200k_base, ~200k)
- Pretrained byte-level BPE built on general web data, not trained by me.
- Handles any UTF-8 text, Cyrillic included, with no `<unk>`.
- Limitation: The vocabulary is fixed at 200k, so the embedding table alone is 200,000 × 256 ≈ 51 M parameters. Most of those tokens never appear in my corpus.

---

## Model/training details

Common core settings used in the notebooks:
- Embedding dimension: 256
- Attention heads: 4
- Transformer blocks: 6
- Context window (`block_size`): 64
- Batch size: 20
- Loss: cross-entropy
- Optimizer: Adam in the older runs, AdamW in the newer ones
- Gradient clipping at 1.0
- Dropout in transformer blocks
- Weight tying (`lm_head.weight = embedding.weight`, `bias=False`) in most of the newer runs
- Lowered LR (lr·0.33) for the char runs that used to diverge

Corpora:
- ENG: ≈ 12.6 M characters (Shakespeare + KJV Bible + a bit of modern text)
- SRB: ≈ 10.1 M characters of Serbian prose in Cyrillic

Model size depends almost entirely on the tokenizer:
- char: ~4.8 M params
- SentencePiece / HF BPE 10k: ~9.9 M without tying, ~7.3 M with it
- tiktoken 200k: ~56 M with tying (it would be ~107 M without)

All 6 transformer layers together are only ~4.7 M params, so on a model this small the tokenizer is the biggest single decision about size.

---

## Results

Metric: minimum per-token cross-entropy on the validation set (1st epoch of each run). Perplexity = e^loss.

| Model | old arch | nn arch | ppl (best) |
|---|---|---|---|
| ENG · char | 1.312 | **1.159** | 3.2 |
| ENG · SentencePiece 10k | 4.317 | 4.247 | 69.9 |
| ENG · HF BPE 10k | **3.880** | 3.881 | 48.4 |
| ENG · tiktoken o200k | 4.587 | 4.285 | 72.6 |
| SRB · char | 1.578 | **1.422** | 4.2 |
| SRB · SentencePiece 10k | 4.844 | 4.761 | 116.9 |
| SRB · HF BPE 10k | 4.750 | 4.607 | 100.2 |
| SRB · tiktoken o200k | **3.892** | 4.012 | 49.0 |

Char and word-level losses are not comparable. A char model chooses among ~100 symbols, while a word-level model chooses among 10k–200k tokens. The comparison only makes sense within the same tokenizer.

---

## What I observed

The tokenizer matters more than anything else. The same model ranges from 10 M to 56 M parameters and from ~4 to ~115 perplexity purely because of how the text is tokenized.

More practical observations:
- On English, my own 10k tokenizers (HF BPE 3.880, SentencePiece 4.317) beat tiktoken (4.587). A vocabulary trained on the corpus wins over sheer size.
- On Serbian it's the other way around. tiktoken gets the best word-level loss (3.892) despite having 5.7× more parameters, thanks to shorter sequences per word and weight tying.
- `nn.TransformerEncoder` beats my hand-written block in 7 of 8 pairs at the same number of iterations (average −0.138), ties in 1 and is never worse. The fused attention kernels and well-tested LayerNorm/residual placement make a difference. That said, only the two tiktoken pairs change the architecture alone, so the sample is small.
- Weight tying is basically a free win. It halves the tiktoken model and acts as regularization.
- The two AdamW runs point in opposite directions, but in both of them the architecture changed too, so they don't prove anything. I still use AdamW because decoupled weight decay is the correct way to regularize, and it's the standard for transformers.
- Without an LR schedule, the longer runs diverge after hitting their minimum. With the lowered LR, the char models stay stable until the end.

This project helped me understand how tokenization choice changes the full modeling pipeline, not just preprocessing.

---

## Hardware

Training was run on **Google Colab T4 GPU**.

---

## How to run

1. Open the notebook for the language/tokenizer/architecture combination you want in Colab or local Jupyter.
2. Put the training corpus for that language in the working directory.
3. Run all cells in order.
4. Optional: Resume from `model_checkpoint.pt` if available.
5. Use the generation cell and test prompts interactively.

Tokenizer notes:
- SentencePiece model files (`.model` and `.vocab`) and the HF BPE tokenizer JSON are generated during training.
- tiktoken needs no training, just `pip install tiktoken`.

Checkpoints aren't in the repo because they're too big for GitHub.

---

## What's next

- A proper LR schedule and early stopping on val loss, so I always keep the checkpoint with the lowest validation loss
- Changing one thing at a time (optimizer, tying, bias) so each effect can be measured on its own
- Side-by-side generation quality checks, not just loss numbers
- Token-normalized metrics so different tokenizers can be compared more fairly

---

## Notes

This repo is intentionally from-scratch and educational, not a production LLM training framework.

The comparisons are indicative, not rigorous. Each run shows only its first epoch, and some settings changed together between runs. The presentation goes into more detail on every chart.

---

## Author

Built by **Pavle Mišović** as a practical deep-dive into decoder-only transformers, tokenization tradeoffs, and low-level PyTorch training behavior.
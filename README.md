# Decoder-Only Transformer (From Scratch): Char-Level vs Subword Tokenization

I built this project to really understand what’s happening inside a decoder-only transformer instead of only using high-level libraries.

The main idea was simple: Train two versions of the same model family and compare behavior in practice.

One version uses character-level tokenization.  
One version uses subword tokenization with SentencePiece (unigram).

Both were trained in Google Colab on a T4 GPU.

---

## Why I made this

I wanted to answer a few practical questions for myself:

How different do training dynamics look when tokenization changes but the core architecture is almost the same?  
How much does sequence compression from subword tokenization matter in real runs?  
What tradeoffs do I get in loss behavior, generation style, and implementation complexity?

This repo is my hands-on benchmark and learning record.

---

## What’s in this project

Two notebook implementations:

- `char_based_decoder_Final.ipynb`
- `word_based_decoder_Final.ipynb` (SentencePiece subword version)

Both models use a custom decoder-only transformer pipeline in PyTorch:
- Token embedding
- Learned positional embedding
- Stacked masked self-attention + feed-forward transformer blocks
- Residual connections + layer norm + dropout
- Linear LM head for next-token prediction

Training setup includes:
- Chunk shuffle before split
- Train/val/test split
- Gradient clipping
- Checkpoint save and resume
- Periodic train/val loss logging

---

## Tokenization setups

### Character-based model
- Vocabulary is built from unique characters in the corpus.
- Encoding and decoding are direct char-to-id and id-to-char mappings.
- Advantage: Very transparent and easy to reason about.
- Limitation: Longer token sequences for the same text.

### Subword model (SentencePiece unigram)
- SentencePiece is trained on the corpus with vocab size 10,000.
- Text is encoded into subword units instead of raw characters.
- Advantage: Better sequence compression and usually better context efficiency per token.
- Limitation: Extra tokenizer training step and larger vocabulary handling.

---

## Model/training details

Common core settings used in the notebooks:
- Embedding dimension: 256
- Attention heads: 4
- Transformer blocks: 6
- Context window (`block_size`): 64
- Batch size: 20
- Optimizer: Adam with weight decay
- Gradient clipping at 1.0
- Dropout in transformer blocks
- LR reduction after epoch transition in the current training loop

---

## What I observed

From my runs, the character model reaches much lower numerical cross-entropy than the subword model, but this is expected because the token spaces are different and loss values are not directly apples-to-apples across tokenizations.

More practical observations:
- Subword tokenization compresses text into fewer tokens and is generally more context-efficient.
- Character tokenization is easier to debug and gives very fine-grained control.
- Both models are sensitive to training duration and LR schedule.
- Long training without careful scheduling can drift or overfit patterns depending on the corpus slice order.

This project helped me understand how tokenization choice changes the full modeling pipeline, not just preprocessing.

---

## Hardware

Training was run on **Google Colab T4 GPU**.

---

## How to run

1. Open either notebook in Colab or local Jupyter.
2. Put `training-data-expanded_3.txt` in the working directory.
3. Run all cells in order.
4. Optional: Resume from `model_checkpoint.pt` if available.
5. Use the generation cell and test prompts interactively.

For subword notebook:
- SentencePiece model files (`.model` and `.vocab`) are generated during training.

---

## What’s next

Very soon I will add a TikTokenizer-based version and compare it directly against the two current tokenizers (character-level and SentencePiece subword) under the same decoder-only setup.

I also plan to standardize the comparison even more with:
- Token-normalized metrics
- Matched training budgets
- Side-by-side generation quality checks

---

## Notes

This repo is intentionally from-scratch and educational, not a production LLM training framework.

If I continue this further, the next upgrades I’d make are:
- Cleaner experiment tracking
- Better LR schedule and early stopping
- Perplexity and token-normalized comparison metrics
- Grouped attention/runtime profiling
- Stronger evaluation prompts and side-by-side generation scoring

---

## Author

Built by **Pavle Mišović** as a practical deep-dive into decoder-only transformers, tokenization tradeoffs, and low-level PyTorch training behavior.

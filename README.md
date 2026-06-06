# Token by Token: Neural Machine Translation from Gated Cells to Global Attention

Conceptual pseudocode repository for a neural machine translation system built from first principles — implementing a hand-rolled LSTM cell, a full Seq2Seq encoder-decoder with cosine-similarity attention, and a Transformer with multi-head self-attention, all without high-level recurrent wrappers. Each architecture directly addresses a failure mode of the previous one, tracing the field's progression from fixed-bottleneck recurrence to parallel global attention. The Transformer variant achieved a 45% reduction in validation perplexity over the vanilla RNN baseline.

---

## Repository Structure

```
NMT_pseudocode/
├── README.md
├── pseudocode/
│   ├── lstm_cell.py        — Hand-rolled LSTM cell: gating arithmetic and memory highway
│   └── architectures.py    — Seq2Seq encoder/decoder, attention, and Transformer
└── diagrams/
    ├── diagram_system_architecture.html
    ├── diagram_lstm_cell.html
    ├── diagram_attention_mechanism.html
    ├── diagram_transformer_architecture.html
    └── diagram_results.html
```

---

## System Architecture

Three architectures were built sequentially, each addressing a specific limitation of the previous:

**RNN Seq2Seq** — The baseline encoder reads the full source sequence and compresses it into a single fixed-size hidden vector (the bottleneck). The decoder is initialised from this vector and generates target tokens one at a time. Performance degrades as source sequences grow longer because all information must fit into one vector.

**LSTM Seq2Seq + Cosine Attention** — The encoder is upgraded to an LSTM, which uses four learnable gates (forget, input, cell, output) to selectively retain and erase information across timesteps, mitigating vanishing gradients. The decoder is enhanced with cosine-similarity attention: at each decode step it queries all encoder hidden states, computes a direction-aware similarity score for each, and constructs a fresh weighted context vector. This breaks the bottleneck — the decoder sees a dynamic, step-specific summary of the source rather than a single compressed vector.

**Transformer** — Replaces recurrence entirely with multi-head self-attention. The full source sequence is processed in one parallel pass: every position attends to every other simultaneously. A position-wise feedforward sublayer adds nonlinear capacity. Residual connections and layer normalisation stabilise training. Without recurrence there is no bottleneck and no sequential dependency — the architecture scales more effectively to longer sequences.

---

## Key Algorithms

### LSTM Gating
Four gates each combine the current input and the previous hidden state through separate affine projections and activations. The forget and input gates together determine what is erased from and written to the cell state (the persistent memory). The output gate determines what fraction of the cell state is exposed as the hidden state at this step. Weights initialised with a variance-preserving scheme; biases initialised to zero.

### Cosine-Similarity Attention
At each decode step, the decoder's current hidden state acts as a query. Cosine similarity (direction-only, magnitude-invariant) is computed between the query and each of the encoder's T hidden states, producing T alignment scores. Softmax normalises these into attention weights. A batched weighted sum over encoder outputs yields the context vector. Using cosine rather than dot-product similarity prevents magnitude-dominated alignment distributions early in training.

### Multi-Head Self-Attention
Each head independently projects the input into lower-dimensional query, key, and value subspaces, computes scaled dot-product attention (scale factor = 1 / √key_dim prevents softmax saturation), and produces a head-specific attended representation. All head outputs are concatenated and linearly projected back to model dimension. A residual connection adds the original input before layer normalisation. Multiple heads allow the model to attend to different relationship types simultaneously.

### Position-wise Feedforward
A two-layer MLP applied identically at every sequence position. An expansion layer followed by ReLU nonlinearity provides representational capacity; a contraction layer returns to model dimension. A residual connection and layer normalisation follow. Without this sublayer, the Transformer would be a purely linear function of its inputs after attention.

---

## Results

| Architecture | Validation Perplexity | Δ vs RNN Baseline |
|---|---|---|
| RNN Seq2Seq | ~27 | — |
| LSTM Seq2Seq | ~22 | −18.5% |
| LSTM + Cosine Attention | ~19 | −29.6% |
| Transformer (tuned) | **14.8** | **−45.2%** |

Each model was trained for the same number of epochs under comparable compute. Perplexity measures how confidently the model assigns probability to the correct next token — lower is better.

---

## How to Run (Conceptual)

The pseudocode files are not executable. To build and train these architectures:

1. **Prepare data** — Tokenise source and target language corpora; build vocabularies; pad sequences to a fixed length.
2. **Instantiate models** — Initialise `RecurrentEncoder` + `RecurrentDecoder` → `Seq2SeqModel` for the recurrent variants; initialise `TransformerTranslator` for the self-attention variant.
3. **Train** — Optimise cross-entropy loss between model log-probabilities and ground-truth target tokens. Train with a scheduler that reduces learning rate on validation perplexity plateau.
4. **Evaluate** — Compute perplexity on a held-out validation set each epoch. Select the checkpoint with the lowest validation perplexity.

---

## Academic Integrity Note

This repository contains **conceptual pseudocode only** — not a working implementation. All specific hyperparameter values, exact tensor dimensions, and framework-specific implementation details have been intentionally omitted. The diagrams illustrate architectural concepts at a high level. This material is shared as a portfolio artifact demonstrating design understanding; it is not suitable as a basis for any academic submission.

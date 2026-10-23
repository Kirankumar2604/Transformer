# Analysis of What We Built: Encoder-Only, Decoder-Only, or Both?

## 1. Overview

Across this series of implementations, we built the core components of the Transformer architecture from scratch using pure Python and NumPy-style operations:

1. **[Tokenization](tokenization/tokenization.md)** – Splitting the input text into discrete tokens (`text.split()`).
2. **[Token Embeddings](embeddings/embeddings.md)** – Converting discrete token IDs into dense numerical vectors.
3. **[Positional Encoding](Positional-encoding/Positional-encoding.md)** – Injecting sequence order information into token embeddings.
4. **[Self-Attention](self-attention/self-attention.md)** – Computing Queries, Keys, and Values ($Q, K, V$) and measuring token relationships.
5. **[Multi-Head Attention](multi-head-self-attention/multi-head-attention.md)** – Splitting representations across multiple attention heads to capture diverse relational subspaces.
6. **[Residual Connections & Layer Normalization](Residual-Connections-&-Layer-Normalization/Residual-Connections-&-Layer-Normalization.md)** – Stabilizing gradients via Add & Norm operations.
7. **[Feed-Forward Network](Feed-Forward-Network/Feed-Forward-Network.md)** – Applying two linear transformations with a non-linear activation (ReLU) per token.
8. **[Output Layer](Output-Layer/Output-Layer.md)** – Projecting hidden states to vocabulary logits, applying softmax, and selecting the next token for text generation.

---

## 2. The Core Question

> **Is the model we implemented an Encoder-Only, Decoder-Only, or Both (Encoder-Decoder) architecture?**

### The Verdict

* **The Output Layer itself is Decoder-style (Decoder-only):** Its objective is next-token prediction and autoregressive text generation.
* **The Hidden / Attention Layers are Encoder-style:** The self-attention mechanism we implemented is bidirectional (unmasked), allowing all tokens to attend to all other tokens.
* **The entire program is a Simplified Hybrid / Demonstration Model:** It uses an **Encoder-style representation backbone** combined with a **Decoder-style next-token generation head**.
* **It is NOT Both (Encoder-Decoder):** We did not implement two separate stacks, nor did we implement **Cross-Attention**.

---

## 3. Why the Output Layer is Decoder-Style

In [Output-Layer.ipynb](Output-Layer/Output-Layer.ipynb), the output layer was implemented as follows:

1. **Extract the Last Token's Representation:**
   ```python
   # Represents the final hidden state of the last token in the prompt
   hidden = [0.5, 0.8, 0.3, 0.9]
   ```
2. **Linear Output Projection:**
   Maps the hidden vector to vocabulary logits using an output weight matrix `W_output` and bias:

   $$
   \text{logits} = h \cdot W_{\text{output}} + b
   $$

3. **Softmax Normalization:**
   Converts raw logits into a valid probability distribution over candidate tokens:

   $$
   P(\text{token}_i) = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}
   $$

4. **Greedy Token Selection:**
   Picks the candidate with the highest predicted probability:
   ```python
   best_index = probabilities.index(max(probabilities))
   predicted_token = candidate_tokens[best_index]
   ```
5. **Autoregressive Text Generation:**
   Appends the predicted token to extend the sentence:
   ```python
   generated_text = text + predicted_token
   ```

In standard Transformer literature (such as GPT-style models), this process of predicting the next token and appending it to continue generation is the signature role of a **Decoder**.

---

## 4. Why the Attention Mechanism We Built is Encoder-Style

In [self-attention.ipynb](self-attention/self-attention.ipynb) and [multi-head-self-attention.ipynb](multi-head-self-attention/multi-head-self-attention.ipynb):

* **Bidirectional (Unmasked) Attention:** When computing attention scores:

  $$
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  $$

  every token attends to **every other token** in the sequence—past, present, and future tokens simultaneously.
* **Lack of Causal Masking:** A true autoregressive **Decoder** requires a causal mask (a lower-triangular matrix setting future positions to $-\infty$) to prevent tokens from looking ahead at future answers during training and inference.
* Because no mask was applied, the internal feature extraction we implemented corresponds directly to a Transformer **Encoder** (like BERT).

---

## 5. Why It Is NOT "Both" (Encoder-Decoder)

As detailed in [encoder-decoder.md](encoder-decoder.md), a true **Encoder-Decoder** model (like the original Vaswani et al. architecture, T5, or BART) requires:

1. **Two Distinct Stacks:**
   * An **Encoder Stack** that reads and encodes the full source sequence bidirectionally.
   * A **Decoder Stack** that takes target tokens, applies masked self-attention, and generates output tokens sequentially.
2. **Cross-Attention Mechanism:**
   * A dedicated multi-head attention block inside the Decoder where:
     * **Queries ($Q$)** come from the Decoder (target sequence).
     * **Keys ($K$) and Values ($V$)** come from the Encoder outputs (source sequence).

Since our program consists of a single linear pipeline without a secondary stack or cross-attention, it is **not an Encoder-Decoder model**.

---

## 6. Comprehensive Architecture Comparison

| Feature | What We Built in Code | Encoder-Only (e.g., BERT) | Decoder-Only (e.g., GPT) | Encoder-Decoder (e.g., T5 / Vaswani) |
|---|---|---|---|---|
| **Number of Stacks** | **1 Single Stack** | 1 Stack (Encoder) | 1 Stack (Decoder) | **2 Stacks** (Encoder + Decoder) |
| **Self-Attention Masking** | **Unmasked** (Bidirectional) | **Unmasked** (Bidirectional) | **Masked** (Causal look-ahead mask) | Masked in Decoder; Unmasked in Encoder |
| **Cross-Attention** | **None** | None | None | **Yes** (Decoder attends to Encoder outputs) |
| **Primary Output Task** | **Next-Token Prediction & Text Generation** | Classification, NER, Masked Token Prediction | **Next-Token Prediction & Autoregressive Generation** | Sequence-to-Sequence (Translation, Summarization) |
| **Output Layer Location** | Top of the single stack | Top of Encoder stack | Top of Decoder stack | Top of Decoder stack |
| **Classification** | **Simplified Hybrid** (Encoder body + Decoder head) | Pure Encoder | Pure Decoder | Pure Encoder-Decoder |

---

## 7. How to Convert What We Built into Pure Architectures

If you want to transition what we wrote into one of the standard pure architectures:

1. **To make it a true Decoder-Only model (like GPT):**
   * Add a causal mask matrix to `self-attention` and `multi-head-attention` by masking all upper-diagonal values ($i < j$) with $-\infty$ before applying softmax.
   * Keep the current output layer as-is for next-token generation.
2. **To make it a true Encoder-Only model (like BERT):** 
   * Keep the current unmasked bidirectional self-attention as-is.
   * Replace the next-token generation output layer with a classification head (e.g., sentiment analysis) or masked token predictor.
3. **To make it an Encoder-Decoder model (like T5 / Vaswani):**
   * Use our current encoder stack to encode the input sentence.
   * Create a second stack (the Decoder) with causal masked self-attention, add a Cross-Attention block connecting to the Encoder outputs, and place the output layer at the end of the Decoder.

---

## Related Topics

- [Life Cycle of Transformers](life-cycle-of-tansformers.md)
- [Encoder and Decoder Architecture](encoder-decoder.md)
- [Output Layer](Output-Layer/Output-Layer.md)
- [Feed-Forward Network](Feed-Forward-Network/Feed-Forward-Network.md)
- [Residual Connections & Layer Normalization](Residual-Connections-&-Layer-Normalization/Residual-Connections-&-Layer-Normalization.md)
- [Multi-Head Attention](multi-head-self-attention/multi-head-attention.md)
- [Self-Attention](self-attention/self-attention.md)
- [Positional Encoding](Positional-encoding/Positional-encoding.md)
- [Embeddings](embeddings/embeddings.md)
- [Tokenization](tokenization/tokenization.md)

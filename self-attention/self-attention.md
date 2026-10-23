
# Self-Attention Mechanism in Transformers

## 1. Introduction

Self-Attention is one of the most important components of the Transformer architecture. It allows a model to understand the relationships between different words or tokens in a sequence.

Using self-attention, each token can focus on other relevant tokens to understand the context and meaning of the input.

**Example:**

Consider the sentence:

"The animal didn't cross the road because it was tired."

To understand the word "it", the model needs to identify that it refers to "the animal".

Self-attention helps the model learn relationships between words using information from other tokens in the sentence.

## 2. Why Is Self-Attention Needed?

Traditional sequence models, such as RNNs and LSTMs, process tokens sequentially.

Transformers use self-attention to process relationships between tokens without requiring the same sequential processing approach.

### Main purposes

- Understand relationships between words.
- Capture contextual information.
- Identify relevant tokens in a sequence.
- Help resolve references between words.
- Allow tokens to interact directly with other tokens.
- Support parallel processing of input tokens during training.

## 3. How Self-Attention Works

Self-attention uses three important vectors:

- **Query (Q):** Represents what a token is looking for.
- **Key (K):** Represents information used to determine how relevant a token is to a query.
- **Value (V):** Contains the information that can be combined based on attention weights.

These vectors are calculated from the input embeddings using learned weight matrices.

### Step-by-step process

1. **Input Embeddings:** Convert tokens into numerical vectors.
2. **Create Q, K, and V:** Project each input vector into query, key, and value vectors.
3. **Calculate Attention Scores:** Compare queries with keys to measure relevance.
4. **Scale the Scores:** Divide the scores by the square root of the key dimension.
5. **Apply Softmax:** Convert scores into attention weights.
6. **Combine Values:** Multiply the attention weights by the value vectors and sum the results.

The output is a contextual representation of each token.

## 4. Scaled Dot-Product Attention Formula

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

- $Q$ = Query matrix.
- $K$ = Key matrix.
- $V$ = Value matrix.
- $K^T$ = Transpose of the key matrix.
- $d_k$ = Dimension of each key vector.
- $\text{softmax}$ = Function that converts scores into normalized weights.

### Why do we divide by sqrt(d_k)?

When key dimensions are large, dot-product scores can become large in magnitude. This can make the softmax distribution very peaked and lead to small gradients.

Dividing by `sqrt(d_k)` helps control the scale of the scores.

## 5. Understanding Query, Key, and Value

Think of searching for information in a library.

- **Query:** The question you want to answer.
- **Key:** The information used to determine whether a document is relevant to your question.
- **Value:** The actual information retrieved from that document.

In self-attention, every token generates its own query, key, and value vectors.

The query of one token is compared with the keys of all tokens it is allowed to attend to. The resulting weights determine how their value vectors contribute to the output.

## 6. Example of Self-Attention

Consider the sentence:

"The cat sat on the mat."

Suppose the model is processing the word "sat".

It may assign different attention weights to other tokens:

| Token | Illustrative attention weight |
|---|---:|
| The | 0.05 |
| cat | 0.35 |
| sat | 0.25 |
| on | 0.10 |
| the | 0.05 |
| mat | 0.20 |

These values are only an illustration, not actual model output.

The token "sat" may attend strongly to "cat" and "mat" because they provide useful context.

The weighted combination of the value vectors produces a contextual representation of "sat".

## 7. Self-Attention vs. Multi-Head Attention

### Self-Attention

A single attention operation computes attention weights and combines value vectors.

### Multi-Head Attention

Multi-head attention performs multiple attention operations in parallel using different learned projections.

Each head can learn to focus on different types of relationships.

For example:

- One head may focus on nearby words.
- Another may focus on subject-verb relationships.
- Another may learn to associate pronouns with earlier nouns.

These are illustrative possibilities; individual heads do not always have such clean interpretations.

The outputs of the heads are concatenated and projected into a combined representation.

### Multi-Head Attention Formula

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

Each head is computed as:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Where:

- $h$ = Number of attention heads.
- $W_i^Q$ = Query projection for head $i$.
- $W_i^K$ = Key projection for head $i$.
- $W_i^V$ = Value projection for head $i$.
- $W^O$ = Output projection matrix.
- $\text{Concat}$ = Concatenation of the head outputs.

## 8. Types of Attention in Transformers

### A. Self-Attention

Queries, keys, and values come from the same input sequence.

It allows tokens to exchange information with other tokens in that sequence, subject to any attention mask.

### B. Causal Self-Attention

Causal attention prevents a token from attending to future tokens.

It is commonly used in autoregressive language models that generate text one token at a time.

For example, when predicting the next token, the model can use preceding tokens but not the actual future tokens.

### C. Cross-Attention

Queries come from one sequence, while keys and values come from another sequence.

For example, in an encoder-decoder Transformer, the decoder can use cross-attention to access representations produced by the encoder.

## 9. Advantages of Self-Attention

- Captures long-range relationships between tokens.
- Allows direct interactions between tokens.
- Supports parallel processing during training.
- Produces context-dependent token representations.
- Works effectively with multi-head attention.
- Forms a central part of many language and vision Transformer models.

## 10. Limitations of Self-Attention

- Standard full self-attention has quadratic time and attention-matrix memory costs with respect to sequence length.
- Processing very long sequences can be computationally expensive.
- Attention weights do not automatically provide a complete explanation of a model's reasoning.
- The model still needs suitable training data and learned parameters to develop useful representations.

## 11. Self-Attention in the Transformer Architecture

In a typical Transformer block, self-attention is followed by additional processing.

A simplified sequence is:

1. Input token representations, including positional information where applicable.
2. Multi-head self-attention.
3. Residual connection and layer normalization, depending on the architecture.
4. Feed-forward neural network.
5. Another residual connection and normalization step.

Modern Transformer implementations may use different normalization placement and other architectural variations.

## 12. Applications of Self-Attention

Self-attention is used in many Transformer-based applications:

- Large Language Models (LLMs)
- Machine Translation
- Text Summarization
- Question Answering
- Chatbots and conversational AI
- Document Understanding
- Vision Transformers
- Speech and multimodal models

## 13. Conclusion

Self-attention allows each token to gather relevant information from other tokens and build a context-aware representation.

The mechanism uses query, key, and value vectors, scaled dot-product scores, softmax weights, and weighted combinations of values.

Multi-head attention extends this idea by allowing several attention operations to learn different relationships in parallel.

Self-attention is one of the foundational mechanisms behind modern Transformer architectures.

---

**Previous Topic:** [Positional Encoding](../Positional-encoding/Positional-encoding.md)

**Next Topic:** [Multi-Head Attention](../multi-head-self-attention/multi-head-attention.md)

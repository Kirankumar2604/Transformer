
# Multi-Head Attention in Transformers

## 1. Introduction

Multi-Head Attention is one of the most important components of the Transformer architecture. It extends the Self-Attention mechanism by allowing the model to perform multiple attention operations in parallel.

Each attention head uses its own learned Query, Key, and Value projections. This allows different heads to learn different relationships between tokens.

**Example:**

Consider the sentence:

"The student gave the teacher a book because she needed it."

Different attention heads may learn relationships involving:
- The relationship between "student" and "gave".
- The relationship between "teacher" and "she".
- The relationship between "book" and "it".

These are illustrative examples. Actual attention heads may learn more complex patterns.

## 2. Why Is Multi-Head Attention Needed?

A single attention operation can learn useful relationships between tokens, but multi-head attention gives the model multiple learned attention spaces.

### Main purposes

- Capture different relationships between tokens.
- Learn different types of contextual information.
- Improve the model's ability to represent complex sequences.
- Allow attention heads to operate in parallel.
- Combine information from multiple attention patterns.

## 3. How Multi-Head Attention Works

Multi-head attention follows these steps:

1. Receive input token representations.
2. Create separate Query, Key, and Value projections for each head.
3. Calculate attention scores for each head.
4. Apply softmax to obtain attention weights.
5. Use the weights to combine the Value vectors.
6. Concatenate the outputs from all heads.
7. Apply a final learned linear projection.

The final output combines information learned by the different attention heads.

## 4. Query, Key, and Value

Multi-head attention uses the same basic concepts as self-attention.

- **Query (Q):** Represents what each token is looking for.
- **Key (K):** Helps determine how relevant each token is to a query.
- **Value (V):** Contains the information combined according to attention weights.

Each head has its own learned projection matrices.

Therefore, different heads can calculate different attention patterns from the same input.

## 5. Multi-Head Attention Formula

The mathematical formula is:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)W^O
$$

Each individual head is calculated as:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Scaled dot-product attention is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

- $Q$ = Query matrix.
- $K$ = Key matrix.
- $V$ = Value matrix.
- $h$ = Number of attention heads.
- $W_i^Q$ = Query projection matrix for head $i$.
- $W_i^K$ = Key projection matrix for head $i$.
- $W_i^V$ = Value projection matrix for head $i$.
- $W^O$ = Output projection matrix.
- $d_k$ = Dimension of the key vectors in each head.
- $\text{Concat}$ = Concatenation of all head outputs.

## 6. Understanding Multi-Head Attention with an Example

Consider the sentence:

"The cat sat on the mat."

Imagine a Transformer with four attention heads.

Each head processes the input using different learned projections.

| Attention Head | Possible learned pattern |
|---|---|
| Head 1 | Relationships between nouns and verbs |
| Head 2 | Relationships between nearby tokens |
| Head 3 | Relationships between objects and actions |
| Head 4 | Broader contextual relationships |

These are simplified examples to illustrate the concept. Attention heads are not guaranteed to specialize in these exact patterns.

After processing the sequence, the model concatenates the outputs of all four heads and applies an output projection.

The resulting vectors combine information from all the heads.

## 7. Multi-Head Attention Architecture

A simplified representation of the process is:

```text
                Input Embeddings
                       |
          +------------+------------+
          |            |            |
       Head 1        Head 2       Head 3 ... Head h
          |            |            |
      Attention     Attention    Attention
          |            |            |
          +------------+------------+
                       |
                  Concatenate
                       |
               Output Projection
                       |
                Final Output
```

Each head performs its attention operation independently before the outputs are combined.

## 8. Single-Head vs. Multi-Head Attention

| Feature | Single-Head Attention | Multi-Head Attention |
|---|---|---|
| Attention operations | One | Multiple |
| Projection spaces | One set | Separate learned projections per head |
| Relationship patterns | One attention pattern per layer and query | Multiple attention patterns |
| Output | One attention result | Combined results from all heads |
| Parallel processing | Yes | Yes, across heads |
| Complexity | Depends on dimensions and sequence length | Depends on head count and total model dimensions |

Multi-head attention does not automatically guarantee better results in every situation. Its effectiveness depends on the architecture, model size, training, and task.

## 9. Why Are Multiple Heads Useful?

Suppose the model processes the sentence:

"John gave Mary a book because she needed it."

To understand this sentence, the model may need to learn relationships between:

- "John" and "gave".
- "Mary" and "she".
- "book" and "it".

Multiple heads provide different learned representations that can help capture these relationships simultaneously.

The combined output gives later Transformer layers a richer representation of the input.

## 10. Number of Attention Heads

The number of attention heads is a model design choice.

For example, a Transformer might use:

- 4 attention heads.
- 8 attention heads.
- 12 attention heads.
- 16 attention heads.

In the standard design, the model dimension is divided across the heads.

For example:

$$
d_{\text{model}} = 512, \quad h = 8
$$

$$
d_k = \frac{d_{\text{model}}}{h} = \frac{512}{8} = 64
$$

Each head uses a key/query dimension of 64 in this example.

In the original Transformer design, the value dimension per head is also 64.

Some modern architectures use different arrangements, including grouped-query attention and multi-query attention.

## 11. Advantages of Multi-Head Attention

- Learns multiple attention patterns.
- Captures different relationships between tokens.
- Combines information from several representation spaces.
- Supports parallel computation across heads.
- Provides a flexible mechanism for contextual representation learning.
- Forms a core component of many Transformer architectures.

## 12. Limitations of Multi-Head Attention

- Full attention can be expensive for long sequences.
- Additional heads can increase computation and memory requirements, depending on the architecture.
- Some heads may learn overlapping or redundant patterns.
- More heads do not always improve model performance.
- Attention patterns alone do not fully explain a model's reasoning.

## 13. Applications of Multi-Head Attention

Multi-head attention is used in many Transformer-based systems, including:

- Large Language Models (LLMs).
- Machine Translation.
- Text Summarization.
- Question Answering.
- Chatbots and conversational AI.
- Vision Transformers.
- Speech Recognition.
- Multimodal AI systems.

## 14. Multi-Head Attention in the Transformer

In a typical Transformer block, multi-head self-attention is followed by a feed-forward neural network.

A simplified sequence is:

1. Input embeddings and positional information, where applicable.
2. Multi-head self-attention.
3. Residual connection and normalization.
4. Feed-forward neural network.
5. Another residual connection and normalization.

The exact order of normalization and other components varies between Transformer architectures.

## 15. Conclusion

Multi-Head Attention extends self-attention by performing multiple learned attention operations in parallel.

Each head uses its own Query, Key, and Value projections. Their outputs are concatenated and passed through a final projection to create a combined representation.

This allows Transformers to learn different relationships between tokens and use that information to understand language and other sequential data.

Multi-head attention is a foundational component of the original Transformer architecture and many modern AI models.

---

**Previous Topic:** [Self-Attention](../self-attention/self-attention.md)

**Next Topic:** [Residual Connections & Layer Normalization](../Residual-Connections-&-Layer-Normalization/Residual-Connections-&-Layer-Normalization.md)
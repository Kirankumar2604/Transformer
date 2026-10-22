
# Positional Encoding in Transformers

## 1. Introduction

Positional Encoding is a technique used in Transformer models to provide information about the position of each word or token in a sequence.

Unlike RNNs and LSTMs, Transformers process all tokens in parallel. Therefore, they need positional information to understand the order of words in a sentence.

**Example:**

- "The dog chased the cat."
- "The cat chased the dog."

Both sentences contain similar words, but their meanings differ because the word order changes. Positional encoding helps Transformers understand these positions.

## 2. Why Is Positional Encoding Needed?

Transformers use a mechanism called Self-Attention to understand relationships between tokens.

However, self-attention alone does not inherently encode token order.

Positional encoding adds position information to token embeddings so the model can distinguish between tokens at different positions.

### Main purposes

- Provides information about token positions.
- Helps the model understand word order.
- Helps distinguish tokens appearing at different positions.
- Supports learning relationships between tokens across a sequence.

## 3. How Positional Encoding Works

The Transformer first converts each token into an embedding vector.

Next, it generates a positional encoding vector for each token position.

The positional encoding is added to the token embedding.

**Formula:**

$$\text{Input Representation} = \text{Token Embedding} + \text{Positional Encoding}$$

The resulting vectors are passed to the Transformer layers.

### Processing steps

1. Convert tokens into token embeddings.
2. Generate positional encodings for each position.
3. Add positional encodings to token embeddings.
4. Pass the combined representations into the Transformer.
5. Use self-attention to learn relationships between tokens.

## 4. Types of Positional Encoding

### A. Sinusoidal Positional Encoding

The original Transformer architecture introduced sinusoidal positional encoding.

It uses sine and cosine functions at different frequencies to represent token positions.

The formulas are:

For even dimensions ($2i$):

$$\text{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

For odd dimensions ($2i+1$):

$$\text{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

Where:

- $\text{pos}$ = Position of the token in the sequence.
- $i$ = Dimension index ($0 \le i < d_{\text{model}} / 2$).
- $d_{\text{model}}$ = Embedding dimension.
- $\text{PE}$ = Positional encoding value.

### Advantages

- Does not require training additional positional parameters.
- Produces a unique positional pattern for each position within the usual operating range.
- Can be calculated for sequence lengths beyond those seen during training, although this does not guarantee good generalization.

### B. Learned Positional Encoding

In learned positional encoding, the model learns positional vectors during training.

Each position is associated with a trainable embedding.

### Advantages

- Learns positional representations suited to the training task.
- Can work effectively within the trained position range.

### Limitations

- Usually requires learned parameters for the supported positions.
- May not generalize well to positions beyond the trained range.

### C. Relative Positional Encoding

Relative positional encoding represents the relative distance or relationship between tokens rather than relying only on their absolute positions.

For example, it can help the model represent whether one token appears two positions before another.

### Advantages

- Captures relative relationships between tokens.
- Can be useful for modeling sequences of different lengths.

### D. Rotary Positional Embedding (RoPE)

Rotary Positional Embedding incorporates positional information by rotating query and key representations in attention.

It is used in several modern Transformer models.

### Advantages

- Encodes position through attention representations.
- Helps attention capture relative positional relationships.
- Is widely used in modern large language models.

## 5. Positional Encoding Example

Consider the sentence:

"The cat sleeps"

The tokens have the following positions:

| Token | Position |
|---|---:|
| The | 0 |
| cat | 1 |
| sleeps | 2 |

Each token receives an embedding vector, and each position receives positional information.

The model combines the two representations:

$$\text{Final Vector} = \text{Token Embedding} + \text{Positional Encoding}$$

As a result, the representation of "cat" at position 1 differs from how the same token would be represented at another position, assuming its token embedding remains the same.

## 6. Positional Encoding vs. Token Embedding

| Feature | Token Embedding | Positional Encoding |
|---|---|---|
| Purpose | Represents token meaning and learned features | Represents token position |
| Input | Token identity | Token position |
| Information | Semantic and lexical features | Order and positional information |
| Example | Representation of "cat" | Representation of position 1 |
| Combination | Added to positional information in the original Transformer | Added to token embeddings in the original Transformer |

## 7. Where Is Positional Encoding Used?

Positional information is important in Transformer-based systems such as:

- Natural Language Processing (NLP)
- Machine Translation
- Text Summarization
- Question Answering
- Large Language Models (LLMs)
- Vision Transformers (ViTs)
- Speech and sequence-processing systems

Different Transformer architectures use different ways to represent position. Not every model uses the original sinusoidal formula.

## 8. Advantages of Positional Encoding

- Gives Transformers information about token order.
- Complements token embeddings.
- Helps self-attention interpret sequence structure.
- Supports sequence modeling in language and other domains.
- Offers several design choices, including sinusoidal, learned, relative, and rotary methods.

## 9. Limitations

- Different positional methods have different generalization properties.
- Learned absolute embeddings may be limited by their trained position range.
- Long sequences can introduce challenges even when a positional method can represent their positions.
- Positional information alone does not guarantee that a model understands sentence meaning correctly.

## 10. Conclusion

Positional encoding is an important component of Transformer design because self-attention does not inherently represent token order.

By incorporating positional information into token representations or attention computations, Transformers can learn relationships between tokens while processing sequences efficiently.

Common approaches include sinusoidal positional encoding, learned positional embeddings, relative positional encoding, and Rotary Positional Embedding (RoPE).

Understanding positional encoding is essential for learning how Transformer architectures and modern language models process sequential information.

---

**Previous Topic:** [Token Embedding](../embeddings/embeddings.md)

**Next Topic:** [Self-Attention Mechanism](../self-attention/self-attention.md)

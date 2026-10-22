
# Feed-Forward Neural Network in Transformers

## 1. Introduction

A Feed-Forward Neural Network (FFN) is an important component of the Transformer architecture. It processes each token representation independently using learned linear transformations and a nonlinear activation function.

In a standard Transformer block, the FFN follows the attention sublayer and transforms the contextual representations produced by attention.

**In simple words:** Attention helps tokens gather information from other tokens, while the feed-forward network transforms the information within each token representation.

## 2. Why Is a Feed-Forward Network Needed?

Self-attention allows tokens to exchange information and build contextual representations. However, attention alone is not the entire Transformer computation.

The feed-forward network applies additional learned transformations to these representations.

### Main purposes

- Transform token representations.
- Learn nonlinear relationships between features.
- Increase the model's representational capacity.
- Process each token's representation independently within the FFN sublayer.
- Help the model learn complex patterns in language and other data.

## 3. How a Feed-Forward Network Works

A standard Transformer FFN contains two linear transformations with a nonlinear activation function between them.

### Processing steps

1. Receive a token representation from the attention sublayer.
2. Apply the first linear transformation.
3. Apply a nonlinear activation function.
4. Apply the second linear transformation.
5. Pass the result to the surrounding residual and normalization components.

The first linear layer typically expands the feature dimension, and the second projects it back to the model dimension.

## 4. Feed-Forward Network Formula

The standard FFN can be represented as:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Where:

- $x$ = Input token representation.
- $W_1$ = First weight matrix.
- $b_1$ = First bias vector.
- $W_2$ = Second weight matrix.
- $b_2$ = Second bias vector.
- $\max(0, z)$ = ReLU activation applied element-wise.

More generally:

$$\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2$$

Different Transformer models can use different activation functions, such as GELU or variants of gated activations.

## 5. Understanding the Two Linear Layers

### A. First Linear Layer

The first linear layer transforms the input from the model dimension into a larger hidden dimension.

For example:

$$d_{\text{model}} = 512, \quad d_{\text{ff}} = 2048$$

The transformation expands the representation from 512 features to 2048 features.

This provides an expanded feature space in which the network can learn useful intermediate representations.

### B. Activation Function

After the first linear transformation, a nonlinear activation function is applied.

The original Transformer uses ReLU.

ReLU is defined as:

$$\text{ReLU}(z) = \max(0, z)$$

Examples:

- $\text{ReLU}(-3) = 0$
- $\text{ReLU}(0) = 0$
- $\text{ReLU}(4) = 4$

The activation function introduces nonlinearity, allowing the network to learn more complex transformations than a stack of linear layers alone.

### C. Second Linear Layer

The second linear layer projects the expanded hidden representation back to the model dimension.

For example:

`2048 features -> 512 features`

This allows the FFN output to have the same dimension as its input, making it compatible with the residual connection.

## 6. Feed-Forward Network Example

Suppose a Transformer uses the following dimensions:

| Component | Dimension |
|---|---:|
| Input representation | 512 |
| Hidden representation | 2048 |
| Output representation | 512 |

The processing flow is:

```text
Input Token Representation
       (512 features)
             |
       Linear Layer 1
             |
      (2048 features)
             |
       ReLU Activation
             |
      (2048 features)
             |
       Linear Layer 2
             |
       (512 features)
             |
        FFN Output
```

These dimensions are an illustrative example. Actual models may use different dimensions and activation functions.

## 7. How the FFN Processes a Sequence

Suppose the input sentence is:

"The cat sat down."

The Transformer represents the tokens as vectors.

The attention sublayer first allows each token to gather contextual information from other tokens, subject to the model's attention rules.

The FFN then applies the same learned transformation separately to each token vector.

Conceptually:

```text
Token 1 Representation ──> FFN ──> Output 1
Token 2 Representation ──> FFN ──> Output 2
Token 3 Representation ──> FFN ──> Output 3
Token 4 Representation ──> FFN ──> Output 4
```

The same FFN weights are shared across token positions within a standard Transformer layer.

The FFN does not directly mix information between token positions. That interaction primarily happens in the attention sublayer.

## 8. Feed-Forward Network vs. Self-Attention

| Feature | Self-Attention | Feed-Forward Network |
|---|---|---|
| Main purpose | Exchange information between tokens | Transform features within each token |
| Token interaction | Tokens can attend to other permitted tokens | Each position is processed independently |
| Core operations | Attention scores and weighted values | Linear layers and nonlinear activation |
| Parameters | Query, Key, Value, and output projections | Input and output projection weights |
| Processing | Depends on relationships between tokens | Uses shared weights at every position |
| Output | Contextual token representations | Further transformed token representations |

Both components work together to build powerful representations.

## 9. Role in the Transformer Block

A simplified Transformer block contains:

1. Multi-Head Attention.
2. A residual connection and normalization.
3. Feed-Forward Network.
4. Another residual connection and normalization.

The exact normalization placement varies by architecture.

In a Post-LayerNorm design, the process can be expressed as:

$$x_1 = \text{LayerNorm}(x + \text{MultiHeadAttention}(x))$$

$$x_2 = \text{LayerNorm}(x_1 + \text{FFN}(x_1))$$

Here, $x_2$ is the output of the Transformer block.

## 10. Common Activation Functions

### A. ReLU

ReLU returns zero for negative inputs and preserves positive inputs.

It was used in the original Transformer FFN.

### B. GELU

GELU is a smooth activation function commonly used in Transformer models.

It scales inputs according to a smooth function rather than setting every negative input to zero.

### C. SwiGLU

SwiGLU is a gated activation design used in several modern language models.

It combines multiple learned projections with a gating mechanism and is not simply the same two-layer ReLU FFN used in the original Transformer.

Different activations and FFN designs can affect model capacity, computation, and performance.

## 11. Advantages of Feed-Forward Networks

- Learn nonlinear transformations of token features.
- Increase the representational capacity of Transformers.
- Complement the information mixing performed by attention.
- Apply the same learned transformation across token positions.
- Can be computed efficiently across tokens in parallel.

## 12. Limitations

- FFN layers contain many parameters in typical Transformer architectures.
- Their computation can be expensive for large models and long sequences.
- A standard FFN does not directly exchange information between token positions.
- Increasing the hidden dimension can increase memory use and computation.

## 13. Applications

Feed-forward networks are used within Transformer architectures for:

- Large Language Models (LLMs).
- Machine Translation.
- Text Summarization.
- Question Answering.
- Chatbots.
- Vision Transformers.
- Speech Processing.
- Multimodal AI.

## 14. Conclusion

The Feed-Forward Neural Network is a fundamental component of the Transformer architecture.

It applies learned linear transformations and nonlinear activation functions to each token representation independently. While attention gathers contextual information from other tokens, the FFN transforms the features within each token representation.

Together, attention and feed-forward layers help Transformers learn complex patterns and representations.

---

**Previous Topic:** [Residual Connections and Layer Normalization](Residual-Connections-&-Layer-Normalization.md)

**Next Topic:** [Encoder and Decoder Architecture](Encoder-and-Decoder.md)


# Output Layer in Transformers

## 1. Introduction

The Output Layer is the component of a Transformer model that converts its internal representations into predictions suitable for the task.

In a language model, the output layer commonly converts the final hidden representation of each token into a score for every token in the model's vocabulary.

These scores are called **logits**. A softmax function can convert logits into probabilities.

**In simple words:** The output layer helps the model decide which token is most likely to come next.

## 2. Why Is the Output Layer Needed?

Transformer layers produce hidden representations, which are numerical vectors containing learned information about the input.

These representations must be converted into task-specific outputs.

For a language model, the output layer helps calculate scores for possible next tokens.

### Main purposes

- Convert hidden representations into vocabulary scores.
- Produce logits for possible output tokens.
- Support next-token prediction in autoregressive language models.
- Provide predictions for training and inference.
- Connect the Transformer representation to the model's output task.

## 3. How the Output Layer Works

A typical language-model output pipeline follows these steps:

1. The Transformer produces a final hidden representation.
2. A linear projection converts that representation into vocabulary logits.
3. Softmax can convert the logits into probabilities.
4. A decoding strategy selects or samples the next token.
5. The selected token can be added to the input sequence for the next generation step.

The precise output pipeline depends on the model architecture and task.

## 4. Output Layer Formula

The linear output projection can be written as:

$$\text{logits} = h W_{\text{out}} + b_{\text{out}}$$

Where:

- $h$ = Final hidden representation.
- $W_{\text{out}}$ = Output projection matrix.
- $b_{\text{out}}$ = Output bias, if used.
- $\text{logits}$ = Scores for the possible output tokens.

To convert logits into probabilities, softmax is applied:

$$P(\text{token}_i) = \frac{\exp(\text{logit}_i)}{\sum_j \exp(\text{logit}_j)}$$

Where:

- $P(\text{token}_i)$ = Probability assigned to token $i$.
- $\exp$ = Exponential function.
- $\text{logit}_i$ = Score for token $i$.
- The denominator sums the exponentials of all vocabulary logits.

In practical implementations, numerically stable softmax calculations are used.

## 5. Example of Next-Token Prediction

Suppose a language model receives the sentence:

"The sun rises in the"

The model might assign the following illustrative probabilities to possible next tokens:

| Possible token | Illustrative probability |
|---|---:|
| morning | 0.60 |
| east | 0.25 |
| sky | 0.10 |
| water | 0.05 |

These values are examples, not real model predictions.

The model could select "morning" because it has the highest probability. Alternatively, a sampling strategy might select another token.

The selected token is then used as part of the context for subsequent generation.

## 6. Understanding Logits and Probabilities

### A. Logits

Logits are the raw numerical scores produced by the output projection.

They can be positive, negative, or zero, and they do not need to sum to one.

A larger logit generally corresponds to a larger softmax probability relative to the other logits.

### B. Softmax

Softmax converts a set of logits into a probability distribution.

For example, suppose the logits are:

`[2.0, 1.0, 0.1]`

After applying softmax, the approximate probabilities are:

`[0.659, 0.242, 0.099]`

The probabilities sum to approximately 1.

### C. Token Selection

During text generation, the model uses its output distribution to choose the next token.

Common approaches include:

- **Greedy decoding:** Select the token with the highest score or probability.
- **Sampling:** Randomly select a token according to a probability distribution.
- **Top-k sampling:** Sample from a restricted set of the highest-scoring tokens.
- **Top-p sampling:** Sample from a set of tokens whose cumulative probability reaches a specified threshold.

The decoding strategy affects the generated text.

## 7. Vocabulary and Output Dimensions

A language model has a tokenizer that maps text to token IDs.

Suppose a model has:

- Hidden dimension: `768`
- Vocabulary size: `30,000`

The output projection typically maps:

`768 features -> 30,000 logits`

Each logit corresponds to a token ID in the model's vocabulary.

The vocabulary size and hidden dimension vary across models.

## 8. Output Layer in Text Generation

The output layer is important in autoregressive language generation.

A simplified generation process is:

```text
Input Text
    |
Tokenizer
    |
Token Embeddings
    |
Transformer Layers
    |
Final Hidden Representation
    |
Linear Output Projection
    |
Vocabulary Logits
    |
Softmax / Decoding
    |
Selected Next Token
    |
Updated Sequence
```

The model repeats the generation process to produce additional tokens until it reaches a stopping condition or generation limit.

Some models use caching to avoid recomputing all previous token representations at every generation step.

## 9. Output Layer During Training

During training, the model learns to predict target tokens from the available context.

For an autoregressive language model:

1. The input sequence is tokenized.
2. The Transformer produces hidden representations.
3. The output projection generates vocabulary logits.
4. The predicted distribution is compared with the correct next token.
5. A loss function, commonly cross-entropy, measures prediction error.
6. Backpropagation updates the model parameters.

The model learns to assign higher probabilities to appropriate target tokens.

## 10. Cross-Entropy Loss

Cross-entropy loss is commonly used to train language models.

For a single target token, it can be written as:

$$\text{Loss} = -\log(P(\text{correct\_token}))$$

Where:

- $P(\text{correct\_token})$ = Predicted probability assigned to the correct token.
- $\log$ = Natural logarithm.

Assigning a low probability to the correct token produces a larger loss. Assigning a higher probability produces a smaller loss.

During training, the loss is generally calculated across many target tokens.

## 11. Output Layer vs. Hidden Layers

| Feature | Transformer Hidden Layers | Output Layer |
|---|---|---|
| Main purpose | Build contextual representations | Convert representations into predictions |
| Output | Hidden feature vectors | Task-specific scores or outputs |
| Language model output | Intermediate token representations | Vocabulary logits |
| Typical operation | Attention, feed-forward transformations, and related operations | Often a linear projection |
| Role in training | Learn useful representations | Produce predictions used to calculate loss |

The output layer depends on the representations learned by the preceding Transformer layers.

## 12. Weight Tying

Some language models share the output projection weights with the token embedding matrix.

This technique is called **weight tying**.

Weight tying can reduce the number of independent parameters and connect input token representations with output token predictions.

However, not all Transformer models use weight tying.

## 13. Other Applications of Output Layers

Output layers are not limited to language generation. Depending on the task, they may produce:

- Class probabilities for text classification.
- Token labels for named entity recognition.
- Vocabulary scores for machine translation.
- Predictions for image classification.
- Outputs for speech recognition.
- Predictions for multimodal tasks.

The output layer's design depends on the model's objective.

## 14. Advantages

- Converts learned representations into usable predictions.
- Supports next-token prediction.
- Works with probability-based decoding methods.
- Provides predictions that can be evaluated with suitable loss functions.
- Can be adapted to many different machine-learning tasks.

## 15. Limitations

- Large vocabularies can make the output projection computationally expensive.
- The output distribution depends on the quality of the learned representations.
- High predicted probability does not guarantee factual correctness.
- Decoding settings can affect output quality and diversity.
- The output layer alone cannot guarantee that generated text is accurate or reliable.

## 16. Conclusion

The Output Layer connects the internal representations of a Transformer to its final prediction task.

In language models, it commonly maps the final hidden representation to vocabulary logits. Softmax can convert those logits into probabilities, and a decoding strategy can select the next token.

During training, these predictions are compared with target tokens using a loss function such as cross-entropy.

The output layer is therefore a key component of Transformer-based language generation and other prediction systems.

---

**Previous Topic:** [Feed-Forward Network](../Feed-Forward-Network/Feed-Forward-Network.md)

**Next Topic:** [Encoder Decoder](../encoder-decoder.md)
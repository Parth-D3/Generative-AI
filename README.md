# Generative-AI

<ol>
  <li><b>Text-Summarizer:</b> A deep learning approach to summarizing text. Model used [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6).</li> 
<br><br>Applications of Text Summarizer

<ol>
<li>News Aggregation</li>
   - Automatically summarize news articles to provide concise headlines and key points for readers who want quick updates.

<li>Academic Research</li>
   - Help researchers quickly grasp the essence of lengthy journal articles, research papers, and literature reviews.

<li>Customer Support</li>
   - Summarize long email threads, chat logs, or support tickets to help agents respond more efficiently.

<li>Legal Document Review</li>
   - Extract key information from contracts, case law, and other legal texts to save time in legal analysis.

<li>Meeting Minutes Generation</li>
   - Convert long meeting transcripts or audio recordings into structured and brief summaries for easier review and sharing.
</ol>
  
<li> Maths & Boolean Reasoning GPT </li>
A pair of decoder-only GPT-style transformer language models built from scratch in PyTorch that learn to evaluate arithmetic expressions (+, -, *, /, including compound parenthesized expressions) and boolean logic expressions (AND, OR, XOR, NOT) via next-token prediction. <br>
Overview<br>
Two models, two domains: one trained on arithmetic (e.g. (6*5)/7=), one on boolean logic (e.g. (True XOR False) AND True =).
Architecture: multi-head self-attention with causal masking, token + position embeddings, feed-forward layers, residual connections, LayerNorm, and dropout.<br>
Task-specific tokenization: a digit-aware tokenizer for multi-digit numbers in the maths model, and a regex word-level tokenizer for the boolean model.
Training: cross-entropy loss with the AdamW optimizer, a 90/10 train–validation split, best-validation-loss checkpointing, and tracked train/val loss curves.
Evaluation: per-operator and overall exact-match accuracy on held-out expressions.<br>
Configuration<br>
## Config

| Hyperparameter | Maths GPT | Boolean GPT |
|---|---|---|
| Embedding dim (`n_embd`) | 384 | 128 |
| Heads (`n_head`) | 6 | 4 |
| Layers (`n_layer`) | 6 | 4 |
| Block size | 12 | 12 |
| Batch size | 48 | 60 |
| Dropout | 0.4 | 0.2 |
| Learning rate | 3e-4 | 3e-4 |
| Optimizer | AdamW | AdamW |
| Iterations | 8,000 | 5,000 |
| Train/val split | 90/10 | 90/10 |

</ol>

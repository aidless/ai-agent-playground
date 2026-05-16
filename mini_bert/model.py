"""Mini-BERT — A from-scratch implementation for learning how Transformers work.

Every line annotated with tensor shapes. Cross-reference with:
  E:/transformers-main/src/transformers/models/bert/modeling_bert.py

Core BERT pipeline:
  token ids → Embedding → 12× TransformerBlock → Pooler → Classifier

References:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Embedding Layer — Turns token IDs into vectors
#    Compare: transformers/models/bert/modeling_bert.py → BertEmbeddings
# =============================================================================

class BertEmbedding(nn.Module):
    """Word + Position + TokenType → single embedding vector.

    BERT uses THREE embeddings added together:
      word_embedding(token) + position_embedding(pos) + segment_embedding(seg)
    """

    def __init__(self, vocab_size=30522, hidden_size=768,
                 max_position=512, type_vocab_size=2,
                 layer_norm_eps=1e-12, dropout=0.1, pad_token_id=0):
        super().__init__()
        # (vocab_size, hidden_size) — maps each word to a vector
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        # (max_position, hidden_size) — encodes "where in the sequence"
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        # (type_vocab_size, hidden_size) — encodes "sentence A vs sentence B"
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # Register position_ids as a buffer — not a parameter, but part of state
        self.register_buffer(
            "position_ids",
            torch.arange(max_position).expand((1, -1)),
        )

    def forward(self, input_ids, token_type_ids=None):
        """
        input_ids:       (batch, seq_len)  e.g. (32, 128)
        token_type_ids:  (batch, seq_len)  e.g. (32, 128)

        Returns: (batch, seq_len, hidden_size)  e.g. (32, 128, 768)
        """
        batch_size, seq_length = input_ids.shape  # (32, 128)

        # Position IDs: just [0, 1, 2, ..., seq_len-1]
        position_ids = self.position_ids[:, :seq_length]  # (1, 128)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (batch_size, seq_length), dtype=torch.long, device=input_ids.device
            )

        # Three embeddings, all shape → (batch, seq_len, hidden_size)
        word_emb = self.word_embeddings(input_ids)              # (32, 128, 768)
        pos_emb = self.position_embeddings(position_ids)        # (1,   128, 768)
        seg_emb = self.token_type_embeddings(token_type_ids)    # (32, 128, 768)

        # Add them together — the "embedding cocktail"
        embeddings = word_emb + pos_emb + seg_emb  # (32, 128, 768)

        embeddings = self.LayerNorm(embeddings)     # Normalize
        embeddings = self.dropout(embeddings)       # Regularize
        return embeddings


# =============================================================================
# 2. Multi-Head Self-Attention — The core of Transformer
#    Compare: transformers/models/bert/modeling_bert.py → BertSelfAttention
#
#    Formula: Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
#
#    Why multi-head? One head might attend to "verbs", another to "subjects".
#    Why √d_k?     Keeps dot products from exploding at large dimensions.
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product attention with multiple heads."""

    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 768 / 12 = 64
        self.scaling = self.head_dim ** -0.5       # 1 / √64 = 0.125

        # Q, K, V projections — 3 separate linear layers
        # Each maps hidden_size → hidden_size (768 → 768)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states:   (batch, seq_len, hidden_size)  e.g. (32, 128, 768)
        attention_mask:  (batch, 1, 1, seq_len)         e.g. (32, 1, 1, 128)
                        — large negative where padded, 0 elsewhere

        Returns: (batch, seq_len, hidden_size)
        """
        bsz, seq_len, _ = hidden_states.shape  # (32, 128, 768)

        # ---- Step 1: Project to Q, K, V ----
        # Each: (batch, seq_len, hidden_size)
        Q = self.query(hidden_states)  # (32, 128, 768)
        K = self.key(hidden_states)    # (32, 128, 768)
        V = self.value(hidden_states)  # (32, 128, 768)

        # ---- Step 2: Split into multiple heads ----
        # Reshape: (batch, seq_len, hidden) → (batch, seq_len, n_heads, head_dim)
        # Then transpose: → (batch, n_heads, seq_len, head_dim)
        Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Each now: (32, 12, 128, 64)

        # ---- Step 3: Compute attention scores: Q · K^T / √d ----
        # Q: (32, 12, 128, 64)  K^T: (32, 12, 64, 128)  →  (32, 12, 128, 128)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        # attn_scores[i][h][q][k] = how much token q attends to token k

        # ---- Step 4: Apply mask (hide padding tokens) ----
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            # Masked positions get large negative → softmax → near zero

        # ---- Step 5: Softmax → attention weights ----
        attn_weights = F.softmax(attn_scores, dim=-1)  # (32, 12, 128, 128)
        attn_weights = self.dropout(attn_weights)

        # ---- Step 6: Weighted sum of values ----
        # attn_weights: (32, 12, 128, 128)  V: (32, 12, 128, 64)  →  (32, 12, 128, 64)
        attn_output = torch.matmul(attn_weights, V)

        # ---- Step 7: Merge heads ----
        # (32, 12, 128, 64) → (32, 128, 12, 64) → (32, 128, 768)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)

        return attn_output  # (32, 128, 768)


# =============================================================================
# 3. Feed-Forward Network — Position-wise, same weights for each position
#    Compare: transformers/models/bert/modeling_bert.py → BertIntermediate + BertOutput
#
#    Formula: FFN(x) = GELU(x · W1 + b1) · W2 + b2
#    Why?     Attention mixes between positions. FFN processes each position
#             independently, adding non-linearity and capacity.
#    Size:    768 → 3072 → 768 (4x expansion in the middle)
# =============================================================================

class FeedForward(nn.Module):
    """Two linear layers with GELU activation. 768 → 3072 → 768."""

    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)   # 768 → 3072
        self.dense2 = nn.Linear(intermediate_size, hidden_size)   # 3072 → 768
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, hidden_size)  e.g. (32, 128, 768)
        Returns:       (batch, seq_len, hidden_size)
        """
        # GELU is a smooth ReLU: GELU(x) = x · Φ(x)
        # Better than ReLU for transformers — used in BERT, GPT-2/3
        h = self.dense1(hidden_states)      # (32, 128, 3072)
        h = F.gelu(h)                       # GELU activation
        h = self.dense2(h)                  # (32, 128, 768)
        h = self.dropout(h)
        return h


# =============================================================================
# 4. Transformer Block — Attention + FFN, with residual connections
#    Compare: transformers/models/bert/modeling_bert.py → BertLayer
#
#    Structure:
#      x → Attention(LayerNorm(x)) + x   ← residual + pre-norm
#      x → FFN(LayerNorm(x)) + x         ← residual + pre-norm
#
#    Why residuals?  Lets gradients flow directly through 12+ layers.
#    Why LayerNorm?  Stabilizes training, reduces sensitivity to initialization.
#    Why PRE-norm?   BERT uses post-norm originally, but pre-norm is more stable.
#                    (The HF implementation uses post-norm to match original BERT)
# =============================================================================

class TransformerBlock(nn.Module):
    """One BERT layer: Self-Attention + Feed-Forward, each with residual."""

    def __init__(self, hidden_size=768, num_heads=12,
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)  # Before attention
        self.ln2 = nn.LayerNorm(hidden_size)  # Before FFN

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states:   (batch, seq_len, hidden_size)
        attention_mask:  (batch, 1, 1, seq_len)
        Returns:         (batch, seq_len, hidden_size)
        """
        # ---- Sub-layer 1: Self-Attention with residual ----
        normed = self.ln1(hidden_states)                       # Pre-norm
        attn_out = self.attention(normed, attention_mask)      # Attend
        hidden_states = hidden_states + attn_out               # Residual: x + Attention(x)

        # ---- Sub-layer 2: Feed-Forward with residual ----
        normed = self.ln2(hidden_states)                       # Pre-norm
        ffn_out = self.ffn(normed)                             # Process
        hidden_states = hidden_states + ffn_out                # Residual: x + FFN(x)

        return hidden_states


# =============================================================================
# 5. BERT Encoder — Stack of N transformer blocks
#    Compare: transformers/models/bert/modeling_bert.py → BertEncoder
# =============================================================================

class BertEncoder(nn.Module):
    """Stack of TransformerBlock layers."""

    def __init__(self, num_layers=12, hidden_size=768, num_heads=12,
                 intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states:  (batch, seq_len, hidden_size)
        Returns:        (batch, seq_len, hidden_size)
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# =============================================================================
# 6. Pooler — Extract a single vector for the whole sequence
#    Compare: transformers/models/bert/modeling_bert.py → BertPooler
#
#    Takes the FIRST token ([CLS]) → dense → tanh
#    [CLS] is special: BERT is trained so this token captures "sentence meaning"
# =============================================================================

class BertPooler(nn.Module):
    """First token ([CLS]) → dense → tanh → single vector for classification."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        """
        hidden_states: (batch, seq_len, hidden_size)
        Returns:       (batch, hidden_size) — one vector per sample
        """
        first_token = hidden_states[:, 0]  # Take [CLS] token  (batch, 768)
        pooled = self.dense(first_token)
        pooled = torch.tanh(pooled)
        return pooled


# =============================================================================
# 7. Full BERT Model — Embedding → Encoder → Pooler → Classifier
#    Compare: transformers/models/bert/modeling_bert.py → BertModel + BertForSequenceClassification
# =============================================================================

class MiniBertForClassification(nn.Module):
    """Complete BERT for text classification.

    Config presets:
      - bert-tiny:  hidden=128, layers=2, heads=2  (for quick testing)
      - bert-mini:  hidden=256, layers=4, heads=4  (for CPU training)
      - bert-base:  hidden=768, layers=12, heads=12 (the original, needs GPU)
    """

    def __init__(self, num_labels=4, hidden_size=256, num_layers=4,
                 num_heads=4, intermediate_size=1024, vocab_size=30522,
                 max_position=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.embeddings = BertEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position=max_position,
            dropout=dropout,
        )
        self.encoder = BertEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.pooler = BertPooler(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        input_ids:       (batch, seq_len)
        attention_mask:  (batch, seq_len)  — 1 for real tokens, 0 for padding
        labels:          (batch,)          — optional, for computing loss

        Returns: dict with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # ---- Create attention mask for the attention layers ----
        # Transformers use additive masks: 0 for attend, large negative for ignore
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        # (batch, seq_len) → (batch, 1, 1, seq_len)
        extended_mask = attention_mask[:, None, None, :]         # (32, 1, 1, 128)
        extended_mask = (1.0 - extended_mask) * -10000.0         # pad → -10000

        # ---- Pipeline ----
        # 1. Embed
        emb = self.embeddings(input_ids, token_type_ids)         # (32, 128, 256)

        # 2. Encode through transformer layers
        encoded = self.encoder(emb, extended_mask)               # (32, 128, 256)

        # 3. Pool [CLS] token
        pooled = self.pooler(encoded)                            # (32, 256)

        # 4. Classify
        logits = self.classifier(pooled)                         # (32, 4)

        # 5. Compute loss if training
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "pooled": pooled}


# =============================================================================
# Quick test: verify shapes are correct
# =============================================================================

if __name__ == "__main__":
    print("Testing Mini-BERT forward pass...\n")

    # Create a tiny BERT for testing
    model = MiniBertForClassification(
        num_labels=4,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
        intermediate_size=512,
    )

    # Fake batch: 4 sentences, max 16 tokens each
    input_ids = torch.randint(0, 30522, (4, 16))
    attention_mask = torch.ones(4, 16)
    labels = torch.randint(0, 4, (4,))

    output = model(input_ids, attention_mask, labels=labels)

    print(f"  Input:        {input_ids.shape}")        # (4, 16)
    print(f"  Embedding:    (4, 16, {model.hidden_size})")
    print(f"  Encoded:      (4, 16, {model.hidden_size})")
    print(f"  Pooled:       (4, {model.hidden_size})")
    print(f"  Logits:       {output['logits'].shape}")  # (4, 4)
    print(f"  Loss:         {output['loss']:.4f}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:   {n_params:,}")

    print("\n[OK] Forward pass successful!")

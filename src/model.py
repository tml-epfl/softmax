import torch
import torch.nn as nn
from src.attention import ATTENTION_FUNCTIONS


class TransformerBlock(nn.Module):
    """Single transformer block with configurable components and attention type."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        use_mlp: bool = True,
        use_layernorm: bool = True,
        attention_type: str = "softmax",
    ):
        super().__init__()
        self.use_mlp = use_mlp
        self.use_layernorm = use_layernorm
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.attention_type = attention_type
        self.attention_fn = ATTENTION_FUNCTIONS[attention_type]

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # LayerNorm (pre-norm style)
        if use_layernorm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim) if use_mlp else None

        # MLP
        if use_mlp:
            mlp_hidden = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, embed_dim),
            )

        # MLP if we use MLP attention
        if attention_type in ["mlp_norm", "mlp_unnorm"]:
            self.attn_mlp = nn.Sequential(
                nn.Linear(self.head_dim, int(mlp_ratio * self.head_dim)),
                nn.GELU(),
                nn.Linear(int(mlp_ratio * self.head_dim), self.head_dim),
            )
        else:
            self.attn_mlp = None

    def forward(self, x, mask=None):
        """Forward pass with optional attention mask."""
        batch_size, seq_len, _ = x.shape

        # Pre-norm attention
        residual = x
        if self.use_layernorm:
            x = self.norm1(x)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.attention_fn(
            q, k, v,
            mlp=self.attn_mlp,
            mask=mask,
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        x = self.o_proj(attn_output)

        x = x + residual

        # Pre-norm MLP
        if self.use_mlp:
            residual = x
            if self.use_layernorm:
                x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual

        return x, attn_weights


class TransformerLM(nn.Module):
    """Transformer for causal (or not) language modeling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: int,
        num_layers: int = 1,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        use_mlp: bool = True,
        use_layernorm: bool = True,
        attention_type: str = "softmax",
        max_seq_length: int = 512,
        is_causal: bool = True,
        positional_type: str = "additive", # one of "add", "concat", "no"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.is_causal = is_causal
        self.positional_type = positional_type

        # Adjust embedding dimensions based on positional embedding mode
        if positional_type == "concat":
            # When concatenating, each embedding is half the final dimension
            assert embed_dim % 2 == 0, "embed_dim must be even when concat_positional=True"
            self.token_embed_dim = embed_dim // 2
            self.pos_embed_dim = embed_dim // 2
            self.final_embed_dim = embed_dim
            self.pos_embedding = nn.Embedding(max_seq_length, self.pos_embed_dim)
        elif positional_type == "add":
            # When adding, both embeddings have the full dimension
            self.token_embed_dim = embed_dim
            self.pos_embed_dim = embed_dim
            self.final_embed_dim = embed_dim
            self.pos_embedding = nn.Embedding(max_seq_length, self.pos_embed_dim)
        elif positional_type == "no":
            self.token_embed_dim = embed_dim
            self.final_embed_dim = embed_dim
        else:
            raise ValueError(f"Invalid positional type: {positional_type}")

        self.embedding = nn.Embedding(vocab_size, self.token_embed_dim)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.final_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_mlp=use_mlp,
                use_layernorm=use_layernorm,
                attention_type=attention_type,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.final_embed_dim) if use_layernorm else nn.Identity()
        self.output_proj = nn.Linear(self.final_embed_dim, output_dim)
    
    def forward(self, input_ids):
        """Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
        
        Returns:
            logits: (batch, seq_len, vocab_size) - predictions for next token at each position
            all_attn_weights: List of attention weights from each layer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal mask if needed
        if self.is_causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        else:
            mask = None
        
        # Token embeddings
        token_emb = self.embedding(input_ids)  # (batch, seq_len, token_embed_dim)
        
        # Positional embeddings
        if self.positional_type != "no":
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_embedding(positions)  # (batch, seq_len, pos_embed_dim)

            # Combine token and positional embeddings
            if self.positional_type == "concat":
                # Concatenate along the embedding dimension
                x = torch.cat([token_emb, pos_emb], dim=-1)  # (batch, seq_len, embed_dim)
            elif self.positional_type == "add":
                # Add positional embeddings (standard approach)
                x = token_emb + pos_emb  # (batch, seq_len, embed_dim)
        else:
            x = token_emb
        
        # Apply transformer layers
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            all_attn_weights.append(attn_weights)
        
        x = self.final_norm(x)

        # Predict the target at all positions
        logits = self.output_proj(x)  # (batch, seq_len, output_dim)
        
        return logits, all_attn_weights

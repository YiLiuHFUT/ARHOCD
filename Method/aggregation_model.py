import torch
import torch.nn as nn


class CrossAttentionWeightGenerator(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=2, num_model=5, num_sample=6):
        super().__init__()
        # -------------------- Embedding layers --------------------
        # Embed model predictions and sample predictions separately
        self.model_embed = nn.Sequential(
            nn.Linear(num_model, hidden_dim),
            nn.ReLU()
        )
        self.sample_embed = nn.Sequential(
            nn.Linear(num_sample, hidden_dim),
            nn.ReLU()
        )

        # -------------------- Attention modules --------------------
        self.model_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.sample_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # -------------------- Layer normalization --------------------
        self.norm_model = nn.LayerNorm(hidden_dim)
        self.norm_sample = nn.LayerNorm(hidden_dim)

        # -------------------- Output weight generation --------------------
        self.weight_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """ x: batch×M×N """
        batch_size, M, N = x.shape

        # 1. Embed the model and sample dimensions separately
        model_emb = self.model_embed(x)  # [batch, M, hidden_dim]
        sample_emb = self.sample_embed(x.permute(0, 2, 1))  # [batch, N, hidden_dim]

        # 2. Apply self-attention across models
        model_attn_out, _ = self.model_attention(
            model_emb, model_emb, model_emb
        )
        model_attn_out = self.norm_model(model_attn_out + model_emb)

        # 3. Apply self-attention across samples
        sample_attn_out, _ = self.sample_attention(
            sample_emb, sample_emb, sample_emb
        )
        sample_attn_out = self.norm_sample(sample_attn_out + sample_emb)

        # 4. Align and concatenate model and sample attention outputs
        model_attn_out = model_attn_out.unsqueeze(2).expand(-1, -1, N, -1)  # [batch, M, N, d]
        sample_attn_out = sample_attn_out.unsqueeze(1).expand(-1, M, -1, -1)  # [batch, M, N, d]
        combined = torch.cat([model_attn_out, sample_attn_out], dim=-1)  # [batch, M, N, 2d]

        # 5. Generate normalized weight matrix
        weights = self.weight_layer(combined).squeeze(-1)  # [batch, M, N]
        weights = torch.softmax(weights.view(batch_size, -1), dim=1).view(batch_size, M, N)
        return weights


class MLPWeightGenerator(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size, M, N = x.shape
        x = x.unsqueeze(-1)
        weights = self.mlp(x).squeeze(-1)
        weights = torch.softmax(weights.view(batch_size, -1), dim=1).view(batch_size, M, N)
        return weights

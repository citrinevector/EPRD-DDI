"""Reciprocal cross-view fusion between the motif sequence and the visual pair representation."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Multi-head cross-attention, softmax(Q K^T / sqrt(d_k)) V, with learned projections."""

    def __init__(self, query_dim, kv_dim, embed_dim=512, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(query_dim, embed_dim)
        self.W_k = nn.Linear(kv_dim, embed_dim)
        self.W_v = nn.Linear(kv_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query: [batch, query_len, query_dim]
            key, value: [batch, key_len, kv_dim]
            attention_mask: optional [batch, query_len, key_len]; 0 marks excluded keys
        Returns:
            output: [batch, query_len, embed_dim]
            attention_weights: [batch, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        output = self.W_o(context)
        return output, attention_weights


class BidirectionalCrossAttention(nn.Module):
    """One reciprocal fusion layer.

    Visual-to-motif path: every motif position (width D) queries the visual pair
    vector (width 2D, a single key and value).
    Motif-to-visual path: the visual pair vector queries all 2L motif positions;
    padded positions are not masked here. The D-dimensional context is projected to 2D.
    Each update is added residually, followed by layer normalization and a residual
    feed-forward block.
    """

    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(BidirectionalCrossAttention, self).__init__()
        self.embed_dim = embed_dim

        # Visual-to-motif path
        self.cross_attn_v2m = CrossAttention(
            query_dim=embed_dim, kv_dim=2 * embed_dim,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm_motif = nn.LayerNorm(embed_dim)
        self.ffn_motif = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.norm_motif_ffn = nn.LayerNorm(embed_dim)

        # Motif-to-visual path
        self.cross_attn_m2v = CrossAttention(
            query_dim=2 * embed_dim, kv_dim=embed_dim,
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.output_proj_m2v = nn.Linear(embed_dim, 2 * embed_dim)
        self.norm_vision = nn.LayerNorm(2 * embed_dim)
        self.ffn_vision = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, 2 * embed_dim), nn.Dropout(dropout)
        )
        self.norm_vision_ffn = nn.LayerNorm(2 * embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Z_motif, I_xy):
        """
        Args:
            Z_motif: [batch, 2L, D] contextual motif sequence of the ordered pair
            I_xy: [batch, 2D] visual pair representation
        Returns:
            z_first: [batch, D] first motif position after fusion
            I_fused: [batch, 2D] refined visual pair representation
            info: attention weights and the full fused motif sequence
        """
        I_xy_seq = I_xy.unsqueeze(1)  # [B, 1, 2D]
        info = {}

        v2m_out, attn_v2m = self.cross_attn_v2m(query=Z_motif, key=I_xy_seq, value=I_xy_seq)
        Z_fused = self.norm_motif(Z_motif + self.dropout(v2m_out))
        Z_fused = self.norm_motif_ffn(Z_fused + self.ffn_motif(Z_fused))
        info['attn_weights_v2m'] = attn_v2m

        m2v_out, attn_m2v = self.cross_attn_m2v(query=I_xy_seq, key=Z_motif, value=Z_motif)
        m2v_out = self.output_proj_m2v(m2v_out)  # [B, 1, 2D]
        I_fused = self.norm_vision(I_xy_seq + self.dropout(m2v_out))
        I_fused = self.norm_vision_ffn(I_fused + self.ffn_vision(I_fused))
        I_fused = I_fused.squeeze(1)  # [B, 2D]
        info['attn_weights_m2v'] = attn_m2v

        z_first = Z_fused[:, 0, :]
        info['Z_fused_full'] = Z_fused
        return z_first, I_fused, info


class MultiLayerBidirectionalCrossAttention(nn.Module):
    """Stack of reciprocal fusion layers; each layer receives the full fused motif sequence of the previous one."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=1, dropout=0.1):
        super(MultiLayerBidirectionalCrossAttention, self).__init__()
        self.layers = nn.ModuleList([
            BidirectionalCrossAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, Z_motif, I_xy):
        all_info = []
        current_Z = Z_motif
        current_I = I_xy
        for i, layer in enumerate(self.layers):
            if i > 0:
                current_Z = all_info[-1]['Z_fused_full']
            z_first, I_fused, info = layer(current_Z, current_I)
            current_I = I_fused
            all_info.append(info)
        return z_first, I_fused, all_info


class FusionModule(nn.Module):
    """Reciprocal cross-view fusion; returns h_fuse = [first motif position; refined visual pair] in R^{3D}."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=1, dropout=0.1):
        super(FusionModule, self).__init__()
        self.cross_attention = MultiLayerBidirectionalCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.embed_dim = embed_dim

    def forward(self, Z_motif, I_xy):
        z_first, I_fused, all_info = self.cross_attention(Z_motif, I_xy)
        h_fuse = torch.cat([z_first, I_fused], dim=-1)  # [B, 3D]
        info = {'motif_first': z_first, 'visual_pair': I_fused, 'h_fuse': h_fuse, 'cross_attention_info': all_info}
        return h_fuse, info

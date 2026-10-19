"""EPRD-DDI: explicit pair relations and dynamic prototype regularization for DDI event prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cross_attention import FusionModule


class ResNetVisionEncoder(nn.Module):
    """Shared ResNet-18 image encoder; the stem and residual stages 1-3 are frozen, stage 4 is trained."""

    def __init__(self, base_model, hidden_dim=512, unfreeze_layer4=True):
        super(ResNetVisionEncoder, self).__init__()
        children = list(base_model.children())
        self.frozen_layers = nn.Sequential(*children[:7])
        self.layer4 = children[7]
        self.avgpool = children[8]

        for param in self.frozen_layers.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = unfreeze_layer4

        resnet_out_dim = 512
        self.proj = nn.Linear(resnet_out_dim, hidden_dim) if resnet_out_dim != hidden_dim else None

    def forward(self, x1, x2):
        feat1 = self.avgpool(self.layer4(self.frozen_layers(x1))).flatten(1)
        feat2 = self.avgpool(self.layer4(self.frozen_layers(x2))).flatten(1)
        if self.proj is not None:
            feat1 = self.proj(feat1)
            feat2 = self.proj(feat2)
        visual_pair = torch.cat([feat1, feat2], dim=-1)  # [B, 2D], in pair order
        return visual_pair, {'feat_A': feat1, 'feat_B': feat2}


class MotifSequenceEncoder(nn.Module):
    """Shared Transformer encoder for the motif-token sequence of one drug."""

    def __init__(self, vocab_size, hidden_dim, max_seq_len, num_layers=6, nhead=8, dropout=0.1):
        super(MotifSequenceEncoder, self).__init__()
        # Token IDs: 0 = PAD, 1..vocab_size = motifs, vocab_size + 1 = SEP.
        self.content_embedding = nn.Embedding(vocab_size + 2, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, motif_seq, attention_mask=None):
        batch_size, seq_len = motif_seq.shape
        content_embed = self.content_embedding(motif_seq)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=motif_seq.device)
        position_ids = torch.clamp(position_ids, max=self.max_seq_len - 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embed = self.position_embedding(position_ids)
        motif_embed = self.dropout(content_embed + position_embed)
        # Padded positions are excluded as keys in self-attention.
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded_seq = self.transformer_encoder(motif_embed, src_key_padding_mask=src_key_padding_mask)
        # The first position is an ordinary motif token (no prepended [CLS]); its state summarizes the drug.
        first_token = encoded_seq[:, 0, :]
        return encoded_seq, first_token


class PrototypeAuxLoss(nn.Module):
    """Training-only prototype loss.

    One prototype per event class is kept in a non-trainable buffer. The rows are
    initialized orthogonally, updated by normalized momentum aggregation of the
    detached pair representations of each mini-batch, and compared with the
    normalized representations by temperature-scaled cosine similarity.
    """

    def __init__(self, input_dim, num_classes, temperature=0.1, momentum=0.99):
        super(PrototypeAuxLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        prototypes = torch.empty(num_classes, input_dim)
        nn.init.orthogonal_(prototypes)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        self.register_buffer('prototypes', prototypes)

    @torch.no_grad()
    def _update_prototypes(self, features, labels):
        features = F.normalize(features.detach(), p=2, dim=-1)
        for k in range(self.num_classes):
            mask = (labels == k)
            if mask.sum() > 0:
                class_mean = F.normalize(features[mask].mean(dim=0), p=2, dim=-1)
                self.prototypes[k] = F.normalize(
                    self.momentum * self.prototypes[k] + (1 - self.momentum) * class_mean,
                    p=2, dim=-1
                )

    def forward(self, features, labels):
        # The bank is updated before the loss, so the current batch contributes a detached term.
        if self.training:
            self._update_prototypes(features, labels)
        z = F.normalize(features, p=2, dim=-1)
        proto = F.normalize(self.prototypes, p=2, dim=-1)
        proto_logits = torch.matmul(z, proto.T) / self.temperature
        return F.cross_entropy(proto_logits, labels.long())


class EPRD_DDI(nn.Module):
    """Dual-view drug encoders, reciprocal cross-view fusion, explicit pair features,
    an MLP classifier, and a training-only prototype loss."""

    def __init__(
        self,
        num_classes,
        vocab_size,
        hidden_dim=512,
        max_motif_len=8,
        cross_attn_layers=1,
        temperature=0.1,
        prototype_momentum=0.99,
        proto_weight=0.1,
        dropout=0.1,
        base_model=None,
    ):
        super(EPRD_DDI, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_motif_len = max_motif_len
        self.proto_weight = proto_weight

        # Classifier input: [h_fuse (3D); h_prod (D); h_diff (D)].
        classifier_dim = 5 * hidden_dim

        self.motif_encoder = MotifSequenceEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max(max_motif_len, 200),
            num_layers=6, nhead=8, dropout=dropout
        )
        self.vision_encoder = ResNetVisionEncoder(base_model, hidden_dim=hidden_dim, unfreeze_layer4=True)
        self.fusion_module = FusionModule(
            embed_dim=hidden_dim, num_heads=8,
            num_layers=cross_attn_layers, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.prototype_aux = PrototypeAuxLoss(
            input_dim=classifier_dim,
            num_classes=num_classes,
            temperature=temperature,
            momentum=prototype_momentum
        )

    def forward(self, x1, x2, motif_seq, attention_mask=None, labels=None, return_loss=False):
        info = {}
        ml = self.max_motif_len
        expected_seq_len = 2 * ml + 1
        if motif_seq.size(1) != expected_seq_len:
            raise ValueError(
                f"motif_seq length mismatch: got {motif_seq.size(1)}, "
                f"expected {expected_seq_len} for max_motif_len={ml}"
            )
        if attention_mask is not None and attention_mask.shape != motif_seq.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} "
                f"does not match motif_seq shape {motif_seq.shape}"
            )

        # Image branch: one vector per drug and the ordered visual pair [v_i; v_j].
        visual_pair, vision_info = self.vision_encoder(x1, x2)
        vision_A = vision_info['feat_A']  # [B, D]
        vision_B = vision_info['feat_B']  # [B, D]

        # Motif branch: split the stored pair sequence at the separator and encode each drug.
        motif_A = motif_seq[:, :ml]
        motif_B = motif_seq[:, ml + 1:]
        mask_A = attention_mask[:, :ml] if attention_mask is not None else None
        mask_B = attention_mask[:, ml + 1:] if attention_mask is not None else None
        encoded_A, m_A = self.motif_encoder(motif_A, mask_A)
        encoded_B, m_B = self.motif_encoder(motif_B, mask_B)
        encoded_motif = torch.cat([encoded_A, encoded_B], dim=1)  # [B, 2L, D]

        # Explicit pair features of the drug-level summaries u = m + v.
        drug_A = m_A + vision_A
        drug_B = m_B + vision_B
        h_prod = drug_A * drug_B
        h_diff = torch.abs(drug_A - drug_B)

        # Reciprocal cross-view fusion (order-dependent).
        h_fuse, fusion_info = self.fusion_module(encoded_motif, visual_pair)
        info['fusion_info'] = fusion_info

        classifier_input = torch.cat([h_fuse, h_prod, h_diff], dim=-1)  # [B, 5D]
        logits = self.classifier(classifier_input)

        # Training objective: cross-entropy + proto_weight * prototype loss. Inference uses the logits only.
        if return_loss and labels is not None:
            ce_loss = self.ce_loss(logits, labels.long())
            proto_loss = self.prototype_aux(classifier_input, labels)
            total_loss = ce_loss + self.proto_weight * proto_loss
            info['proto_loss'] = proto_loss.item()
            info['loss'] = total_loss
            info['ce_loss'] = ce_loss.item()

        return logits, classifier_input, info

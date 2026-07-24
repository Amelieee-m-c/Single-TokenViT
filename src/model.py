"""
Reproduction (PyTorch) of:
"A Lightweight Single-Token CNN-Transformer Architecture for Robust
Multi-Crop Plant Disease Classification" (Hasanah, Liu, Azmi; IEEE Access,
2026). No official code repo found; independent clean-room reimplementation
from the paper's equations (1)-(22) and Figures 1-4.

Architecture (Section III-C):
  1. DenseNet121 backbone -> Fcnn in R^(1024x7x7) (last dense block, before
     the original 1000-way ImageNet classifier, which is dropped) ->
     GAP -> fcnn in R^1024.
  2. CNN-driven ViT module: a 7x7/stride-7 conv turns Fcnn into a single
     patch token Zp in R^(1x64); concatenated with a learnable class token
     zcls, + learnable positional embedding -> Z0 in R^(2x64); one
     transformer encoder layer (MHSA, num_heads=2, dk=32; MLP 64->128->64,
     GELU, dropout) -> ZL in R^(2x64).
  3. Attention-based global aggregation: a *second*, separate Q/K/V
     attention (not head-split, full d=64) over ZL, mean-pooled over the
     2 tokens, LayerNormed, then linearly expanded 64->128 -> fvit in R^128.
  4. Fusion: concat(fcnn, fvit) -> R^1152 -> Linear(1152,512)+ReLU+Dropout(0.3)
     -> Linear(512, num_classes).

Paper is unusually precise (explicit tensor shapes and equations throughout),
but two of its own numbers are internally inconsistent with each other:

  - **Patch embedding cost.** Eq. (4)/(5) describe "a convolutional
    projection layer of size 7x7 and stride 7" mapping Fcnn (1024x7x7) to a
    single 64-dim token. Taken literally (a dense Conv2d(1024, 64, kernel=7,
    stride=7)), that layer alone costs 1024*64*49 = 3,211,264 params --
    over 20x the paper's own claimed "Lightweight ViT module: ~0.15M
    parameters" (Section 5) for the *entire* transformer module. Since a
    7x7 kernel over a 7x7 input with stride 7 produces exactly one output
    position regardless of how the input channels are combined, we
    implement the mathematically-equivalent-in-spirit but far cheaper
    version: global-average-pool Fcnn to (B,1024), then Linear(1024, 64).
    This costs ~65.6K params and brings the whole ViT module (patch embed +
    cls/pos tokens + one encoder layer) to ~99K params -- much closer to
    the paper's stated ~0.15M than the literal reading's ~3.2M.
  - **Fusion classifier cost.** Eq. (22) explicitly gives W1 in
    R^(512x1152), which alone costs 512*1152+512 = 590,336 params -- already
    2.5x the paper's own claimed "Fusion classifier: ~0.23M parameters"
    (Section 5) for that whole block. We kept the explicit equation's shape
    (it's a formal specification, more authoritative than a rounded summary
    figure) rather than shrinking W1 to hit the 0.23M target.
  - LayerNorm placement inside the single transformer encoder layer isn't
    shown in the equations (eq. 8 shows bare attention only); we use the
    standard pre-norm ViT encoder block (LayerNorm before MHSA and before
    the MLP), the conventional choice the paper is almost certainly
    building on.
  - Section 5's "DenseNet121 backbone: ~7.98M params" matches torchvision's
    *full* densenet121 including its original 1000-way ImageNet classifier
    (1,025,000 params) even though that classifier is never used here (we
    take Fcnn from `.features` and attach our own fusion head). We
    instantiate `.features` only, so our backbone contributes ~6.95M, not
    7.98M -- a paper accounting quirk, not an architecture ambiguity.

Net effect: our total lands close to the paper's cited 8.37M (see
`count_all_params` in `if __name__` block) despite the two offsetting
corrections above, because they push in opposite directions (patch-embed
correction: -3.1M; fusion classifier as literally specified: +0.36M vs.
the paper's own summary of that block).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import DenseNet121_Weights


class DenseNetBackbone(nn.Module):
    """DenseNet121 feature extractor; denseblock3/denseblock4 fine-tuned,
    everything before them frozen, per the paper's stated fine-tuning scope."""

    def __init__(self, pretrained: bool = True, freeze_early: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.densenet121(weights=weights)
        self.features = backbone.features  # -> (B, 1024, 7, 7) for 224x224 input
        self.out_channels = 1024

        if freeze_early:
            finetune_from = ("denseblock3", "transition3", "denseblock4", "norm5")
            for name, param in self.features.named_parameters():
                top_level = name.split(".")[0]
                param.requires_grad = top_level in finetune_from

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        fmap = self.features(x)          # Fcnn: (B, 1024, 7, 7)
        fcnn = self.gap(fmap).flatten(1)  # (B, 1024)
        return fmap, fcnn


class SingleTokenViT(nn.Module):
    """CNN-driven ViT module: one CNN-derived patch token + one class token,
    refined by a single lightweight transformer encoder layer (Section III-C.2)."""

    def __init__(self, in_channels: int = 1024, embed_dim: int = 64,
                 num_heads: int = 2, mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # GAP + Linear, not a literal Conv2d(in_channels, embed_dim, 7, stride=7)
        # -- see module docstring "Patch embedding cost" for why.
        self.patch_pool = nn.AdaptiveAvgPool2d(1)
        self.patch_embed = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, fmap):
        b = fmap.shape[0]
        pooled = self.patch_pool(fmap).flatten(1)      # (B, in_channels)
        zp = self.patch_embed(pooled).unsqueeze(1)      # (B, 1, 64)
        cls = self.cls_token.expand(b, -1, -1)                   # (B, 1, 64)
        z0 = torch.cat([cls, zp], dim=1) + self.pos_embed        # (B, 2, 64)

        h = self.norm1(z0)
        attn_out, _ = self.mhsa(h, h, h, need_weights=False)
        z0 = z0 + attn_out
        z0 = z0 + self.mlp(self.norm2(z0))
        return z0  # ZL: (B, 2, 64)


class AttentionPooling(nn.Module):
    """Attention-based global aggregation over the 2 refined tokens
    (Section III-C.3, eqs. 10-16) -- a second, separate attention pass,
    then mean-pool + LayerNorm + linear expansion to 128-d."""

    def __init__(self, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5
        self.norm = nn.LayerNorm(embed_dim)
        self.expand = nn.Linear(embed_dim, out_dim)

    def forward(self, zl):
        q, k, v = self.wq(zl), self.wk(zl), self.wv(zl)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        z_refined = attn @ v                     # (B, 2, 64)
        pooled = self.norm(z_refined.mean(dim=1))  # (B, 64)
        return self.expand(pooled)                # fvit: (B, 128)


class FusionClassifier(nn.Module):
    def __init__(self, cnn_dim: int = 1024, vit_dim: int = 128,
                 hidden_dim: int = 512, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(cnn_dim + vit_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, fcnn, fvit):
        f = torch.cat([fcnn, fvit], dim=1)
        f = self.dropout(self.act(self.fc1(f)))
        return self.fc2(f)


class DenseNetSingleTokenViT(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = DenseNetBackbone(pretrained=pretrained_backbone)
        self.vit = SingleTokenViT(in_channels=self.backbone.out_channels)
        self.attn_pool = AttentionPooling()
        self.classifier = FusionClassifier(
            cnn_dim=self.backbone.out_channels, vit_dim=128, num_classes=num_classes
        )

    def forward(self, x):
        fmap, fcnn = self.backbone(x)
        zl = self.vit(fmap)
        fvit = self.attn_pool(zl)
        return self.classifier(fcnn, fvit)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = DenseNetSingleTokenViT(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("output shape:", y.shape)
    trainable = count_params(m)
    total = count_all_params(m)
    print(f"trainable params: {trainable} ({trainable/1e6:.3f}M)")
    print(f"total params (incl. frozen): {total} ({total/1e6:.3f}M)")

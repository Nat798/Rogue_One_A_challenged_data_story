# model_dino.py
# Architecture SegDinoRegressorV2 — DINOv3 ViT-B/16.
# Stratégie ExPLoRA : LoRA r=32 sur blocs 0-9, dégel complet blocs 10-11,
# LayerNorm dégelées partout. Têtes : CrossAttention + AttentionMapEncoder
# + TransformerFusionDecoder → régression du taux d'occlusion faciale.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

# ─────────────────────────────────────────────────────────────────────────────
# Constantes ViT-B/16 DINOv3
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID                   = "facebook/dinov3-vitb16-pretrain-lvd1689m"
EMBED_DIM                  = 768
N_BLOCKS                   = 12
N_HEADS                    = 12
N_PATCHES_DEFAULT          = 196
N_REGISTERS                = 4
INTERMEDIATE_BLOCK_DEFAULT = 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. LoRA
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """W_eff = W_pretrained + (alpha/r) * B @ A. W gelé, seuls lora_A/lora_B entraînés."""
    def __init__(self, linear: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = (nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                       if linear.bias is not None else None)
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scale  = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    def extra_repr(self) -> str:
        r = self.lora_A.shape[0]
        return f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, r={r}, scale={self.scale:.3f}"


def inject_lora(module, r=16, alpha=32.0, target_names=("query", "key", "value")) -> int:
    """Remplace récursivement les nn.Linear cibles par LoRALinear. Retourne le nombre remplacé."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, r=r, alpha=alpha, target_names=target_names)
    return count


def find_linear_names(module: nn.Module, prefix: str = "") -> None:
    """Utilitaire debug : affiche tous les nn.Linear du module avec leur chemin."""
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            print(f"  Linear '{name}'  chemin='{full}'  ({child.in_features}→{child.out_features})")
        else:
            find_linear_names(child, full)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CrossAttentionHead
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionHead(nn.Module):
    """Query learnable → cross-attention sur les patch tokens spatiaux."""
    def __init__(self, embed_dim=768, n_heads=12, attn_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, attn_dim)
        self.query      = nn.Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        self.attn       = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(attn_dim)
        self.proj       = nn.Sequential(nn.Linear(attn_dim, embed_dim), nn.GELU())

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        tokens  = self.input_proj(patch_tokens)
        B       = tokens.size(0)
        q       = self.query.expand(B, -1, -1)
        out, _  = self.attn(query=q, key=tokens, value=tokens)
        return self.proj(self.norm(out.squeeze(1)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. AttentionMapEncoder
# ─────────────────────────────────────────────────────────────────────────────

class AttentionMapEncoder(nn.Module):
    """Encode les cartes d'attention CLS→patches avec pondération apprise par tête."""
    def __init__(self, n_patches=196, n_heads=6, out_dim=128):
        super().__init__()
        self.head_weights = nn.Parameter(torch.ones(n_heads) / n_heads)
        self.encoder = nn.Sequential(
            nn.LayerNorm(n_patches),
            nn.Linear(n_patches, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )

    def forward(self, attn_maps: torch.Tensor) -> torch.Tensor:
        w        = F.softmax(self.head_weights, dim=0)
        saliency = (attn_maps * w[None, :, None]).sum(dim=1)
        return self.encoder(saliency)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TransformerFusionDecoder
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFusionDecoder(nn.Module):
    """Self-attention entre les 4 streams (CLS, spatial, texture, saliency) → régression."""
    def __init__(self, embed_dim=768, saliency_dim=128, d_model=256, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj_cls      = nn.Linear(embed_dim,    d_model)
        self.proj_spatial  = nn.Linear(embed_dim,    d_model)
        self.proj_texture  = nn.Linear(embed_dim,    d_model)
        self.proj_saliency = nn.Linear(saliency_dim, d_model)
        self.stream_embed  = nn.Parameter(torch.randn(4, d_model) * 0.02)
        encoder_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor   = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32),     nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(self, cls_feat, spatial_feat, texture_feat, saliency_feat) -> torch.Tensor:
        t0     = self.proj_cls(cls_feat)
        t1     = self.proj_spatial(spatial_feat)
        t2     = self.proj_texture(texture_feat)
        t3     = self.proj_saliency(saliency_feat)
        tokens = torch.stack([t0, t1, t2, t3], dim=1) + self.stream_embed.unsqueeze(0)
        out    = self.transformer(tokens)
        return self.regressor(out[:, 0, :]).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OcclusionLoss
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionLoss(nn.Module):
    """Loss identique à la métrique du challenge : w_i = 1/30 + GT_i, Score = (Err_F+Err_M)/2 + |Err_F-Err_M|."""
    def __init__(self, gender_penalty: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.gender_penalty = gender_penalty
        self.eps            = eps

    def _weighted_err(self, pred, target):
        w = (1.0 / 30.0) + target
        return (w * (pred - target) ** 2).sum() / w.sum().clamp(min=self.eps)

    @staticmethod
    def _is_female(g) -> bool:
        return str(g).strip().lower() in ("0", "0.0", "f", "female")

    def forward(self, pred, target, gender):
        mask_f     = torch.tensor([self._is_female(g) for g in gender], dtype=torch.bool, device=pred.device)
        mask_m     = ~mask_f
        err_global = self._weighted_err(pred, target)
        err_f      = self._weighted_err(pred[mask_f], target[mask_f]) if mask_f.sum() > 0 else err_global
        err_m      = self._weighted_err(pred[mask_m], target[mask_m]) if mask_m.sum() > 0 else err_global
        return (err_f + err_m) / 2.0 + self.gender_penalty * torch.abs(err_f - err_m)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Modèle principal
# ─────────────────────────────────────────────────────────────────────────────

class SegDinoRegressorV2(nn.Module):
    """
    DINOv3 ViT-B/16 — ExPLoRA : LoRA r=32 blocs 0-9, dégel complet blocs 10-11,
    LayerNorm dégelées partout. Têtes entraînables : CrossAttention + AttentionMap + FusionDecoder.
    """
    def __init__(
        self,
        use_lora=True, lora_r=32, lora_alpha=64.0,
        n_unfrozen_blocks=2, intermediate_block_idx=INTERMEDIATE_BLOCK_DEFAULT,
        attn_block_idx=-1, n_patches=N_PATCHES_DEFAULT,
        dropout=0.2, decoder_weights_path=None,
    ):
        super().__init__()
        self.intermediate_block_idx = intermediate_block_idx
        self.attn_block_idx         = attn_block_idx
        self.n_patches              = n_patches

        self.backbone  = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, attn_implementation="eager")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        blocks   = self._get_blocks()
        n_frozen = N_BLOCKS - n_unfrozen_blocks

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if use_lora:
            target_names = ("q_proj", "k_proj", "v_proj")
            n_replaced   = 0
            for block in blocks[:n_frozen]:
                n_replaced += inject_lora(block, r=lora_r, alpha=lora_alpha, target_names=target_names)
            if n_replaced == 0:
                print("⚠ 0 couches LoRA remplacées.")
                find_linear_names(blocks[0])
            else:
                print(f"✓ LoRA r={lora_r} α={lora_alpha} — blocs 0–{n_frozen-1} : {n_replaced} couches")

        for block in blocks[n_frozen:]:
            for p in block.parameters():
                p.requires_grad_(True)
        print(f"✓ Full fine-tuning — blocs {n_frozen}–{N_BLOCKS-1}")

        n_ln, ln_params = 0, 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.LayerNorm):
                for p in module.parameters():
                    p.requires_grad_(True)
                    ln_params += p.numel()
                n_ln += 1
        print(f"✓ LayerNorm dégelées : {n_ln} modules ({ln_params:,} params)")

        self.cross_attn_head  = CrossAttentionHead(embed_dim=EMBED_DIM, n_heads=N_HEADS, attn_dim=256, dropout=dropout)
        self.attn_map_encoder = AttentionMapEncoder(n_patches=n_patches, n_heads=N_HEADS, out_dim=128)
        self.decoder          = TransformerFusionDecoder(embed_dim=EMBED_DIM, saliency_dim=128,
                                                         d_model=256, n_heads=4, n_layers=2, dropout=dropout)

        if decoder_weights_path is not None:
            ckpt  = torch.load(decoder_weights_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            self.decoder.load_state_dict(state, strict=False)
            print(f"✓ Poids decoder chargés : {decoder_weights_path}")

        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing activé (VRAM -40%)")

        self._print_param_stats()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_blocks(self) -> list:
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
            return list(self.backbone.model.layer)
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return list(self.backbone.encoder.layer)
        raise AttributeError(f"Structure backbone non reconnue : {[n for n, _ in self.backbone.named_children()]}")

    def _print_param_stats(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_p    = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and ("lora_A" in n or "lora_B" in n))
        ln_p      = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and "backbone" in n and "lora_A" not in n and "lora_B" not in n and p.dim() == 1)
        head_p    = sum(p.numel() for n, p in self.named_parameters() if p.requires_grad and "backbone" not in n)
        print(f"Params total : {total:,}  |  entraînables : {trainable:,} ({100*trainable/total:.1f}%)")
        print(f"  LoRA : {lora_p:,}  LayerNorm : {ln_p:,}  Têtes : {head_p:,}")

    def get_trainable_param_groups(self, lr_lora=5e-6, lr_backbone=1e-5, lr_decoder=1e-4) -> list:
        """4 groupes : LoRA → lr_lora, LayerNorm → lr_lora, blocs dégelés → lr_backbone, têtes → lr_decoder."""
        lora_params     = [p for n, p in self.named_parameters() if p.requires_grad and "backbone" in n and ("lora_A" in n or "lora_B" in n)]
        ln_params       = [p for n, p in self.named_parameters() if p.requires_grad and "backbone" in n and "lora_A" not in n and "lora_B" not in n and p.dim() == 1]
        unfrozen_params = [p for n, p in self.named_parameters() if p.requires_grad and "backbone" in n and "lora_A" not in n and "lora_B" not in n and p.dim() > 1]
        head_params     = [p for n, p in self.named_parameters() if p.requires_grad and "backbone" not in n]
        return [
            {"params": lora_params,     "lr": lr_lora,     "weight_decay": 0.0},
            {"params": ln_params,       "lr": lr_lora,     "weight_decay": 0.01},
            {"params": unfrozen_params, "lr": lr_backbone, "weight_decay": 0.01},
            {"params": head_params,     "lr": lr_decoder,  "weight_decay": 0.01},
        ]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs       = self.backbone(pixel_values=pixel_values, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states
        attentions    = outputs.attentions
        last_hs       = hidden_states[-1]

        cls_feat     = last_hs[:, 0, :]
        patch_tokens = last_hs[:, 1 + N_REGISTERS:, :]
        spatial_feat = self.cross_attn_head(patch_tokens)
        mid_hs       = hidden_states[self.intermediate_block_idx + 1]
        texture_feat = mid_hs[:, 1 + N_REGISTERS:, :].mean(dim=1)

        if attentions is None or len(attentions) == 0:
            saliency_feat = torch.zeros(last_hs.size(0), 128, device=last_hs.device)
        else:
            attn_idx      = self.attn_block_idx % len(attentions)
            cls_attn      = attentions[attn_idx][:, :, 0, 1 + N_REGISTERS:]
            cls_attn      = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            saliency_feat = self.attn_map_encoder(cls_attn)

        return self.decoder(cls_feat, spatial_feat, texture_feat, saliency_feat).clamp(0.0, 1.0)

    @torch.no_grad()
    def get_saliency_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs  = self.backbone(pixel_values=pixel_values, output_attentions=True)
        attn_idx = self.attn_block_idx % len(outputs.attentions)
        cls_attn = outputs.attentions[attn_idx][:, :, 0, 1 + N_REGISTERS:]
        cls_attn = cls_attn / cls_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        cls_attn = cls_attn.mean(dim=1)
        h = w    = int(cls_attn.shape[-1] ** 0.5)
        return cls_attn.reshape(-1, h, w)

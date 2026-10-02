
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Small utilities
# ============================================================


def parse_adapter_layers(adapter_layers: Union[str, Sequence[int], None], num_layers: int) -> List[int]:
    """
    Parse adapter layer specification.

    Supported values:
      - "none" or "": no adapter layers
      - "all": all layers
      - "last": only final layer
      - "last2": final two layers
      - comma-separated 0-based indices, e.g. "1,2"
      - a list/tuple of integer layer indices
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    if adapter_layers is None:
        return [num_layers - 1]

    if isinstance(adapter_layers, (list, tuple)):
        layers = [int(x) for x in adapter_layers]
    else:
        spec = str(adapter_layers).strip().lower()
        if spec in ["", "none", "false", "0"]:
            layers = []
        elif spec == "all":
            layers = list(range(num_layers))
        elif spec == "last":
            layers = [num_layers - 1]
        elif spec == "last2":
            layers = list(range(max(0, num_layers - 2), num_layers))
        else:
            # 0-based comma-separated indices, e.g. "0,2"
            layers = [int(x.strip()) for x in spec.split(",") if x.strip() != ""]

    # Validate and unique-sort.
    cleaned = sorted(set(layers))
    for idx in cleaned:
        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"Invalid adapter layer index {idx}. Valid range is 0 to {num_layers - 1}. "
                "Use 'last', 'last2', 'all', 'none', or comma-separated 0-based indices."
            )
    return cleaned


# ============================================================
# Base blocks from V1
# ============================================================


class TransformerStack(nn.Module):
    def __init__(self, embed_dim: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.encoder(x))


class AttentionPool1D(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        hidden = max(1, embed_dim // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        weights = torch.softmax(self.score(x), dim=1)  # [B, L, 1]
        pooled = torch.sum(weights * x, dim=1)         # [B, D]
        if return_weights:
            return pooled, weights.squeeze(-1)         # [B, D], [B, L]
        return pooled


# ============================================================
# V2: Clinical-routed low-rank adapter components
# ============================================================


class LowRankAdapter(nn.Module):
    """Small bottleneck adapter: D -> rank -> D."""

    def __init__(self, embed_dim: int, rank: int, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter_rank must be positive")
        self.down = nn.Linear(embed_dim, rank)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.up = nn.Linear(rank, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.up(self.dropout1(self.act(self.down(x)))))


class ClinicalRoutedLowRankAdapter(nn.Module):
    """
    K low-rank adapters mixed by a clinical router.

    Inputs:
      x: [B, L, D]
      clinical_context: [B, D]

    Outputs:
      routed_delta: [B, L, D]
      aux:
        router_alpha: [B, K]
        expert_outputs_pooled: [B, K, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 3,
        rank: int = 8,
        dropout: float = 0.1,
        router_tau: float = 1.0,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_adapter_experts must be positive")
        if router_tau <= 0:
            raise ValueError("router_tau must be positive")

        self.embed_dim = int(embed_dim)
        self.num_experts = int(num_experts)
        self.rank = int(rank)
        self.router_tau = float(router_tau)

        self.router = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_experts),
        )
        self.experts = nn.ModuleList([
            LowRankAdapter(embed_dim=embed_dim, rank=rank, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        clinical_context: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        router_logits = self.router(clinical_context)  # [B, K]
        alpha = torch.softmax(router_logits / self.router_tau, dim=-1)

        # [B, K, L, D]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        routed_delta = torch.sum(alpha[:, :, None, None] * expert_outputs, dim=1)

        aux: Dict[str, torch.Tensor] = {}
        if return_aux:
            aux["router_logits"] = router_logits
            aux["router_alpha"] = alpha
            aux["expert_outputs_pooled"] = expert_outputs.mean(dim=2)  # [B, K, D]
        return routed_delta, aux


class ClinicalRoutedTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with optional clinical-routed adapter inserted
    inside the block after the shared self-attention + FFN update.

    This follows a norm-first Transformer style similar to PyTorch's
    TransformerEncoderLayer(norm_first=True), then adds:
        x = adapter_norm(x + adapter_scale * routed_adapter(x, clinical_context))
    when use_adapter=True.
    """

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        dropout: float,
        use_adapter: bool = False,
        num_adapter_experts: int = 3,
        adapter_rank: int = 8,
        adapter_scale: float = 0.1,
        adapter_scale_learnable: bool = False,
        router_tau: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.use_adapter = bool(use_adapter)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        if self.use_adapter:
            self.routed_adapter = ClinicalRoutedLowRankAdapter(
                embed_dim=embed_dim,
                num_experts=num_adapter_experts,
                rank=adapter_rank,
                dropout=dropout,
                router_tau=router_tau,
            )
            self.adapter_norm = nn.LayerNorm(embed_dim)
            if adapter_scale_learnable:
                self.adapter_scale = nn.Parameter(torch.tensor(float(adapter_scale)))
            else:
                self.register_buffer("adapter_scale", torch.tensor(float(adapter_scale)))
        else:
            self.routed_adapter = None
            self.adapter_norm = None
            self.register_buffer("adapter_scale", torch.tensor(0.0))

    @staticmethod
    def _mha_forward(
        mha: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            return mha(
                query=query,
                key=key,
                value=value,
                need_weights=need_weights,
                average_attn_weights=False,
            )
        except TypeError:
            return mha(query=query, key=key, value=value, need_weights=need_weights)

    def forward(
        self,
        x: torch.Tensor,
        clinical_context: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        aux: Dict[str, Any] = {}

        # Norm-first self-attention.
        x_norm = self.norm1(x)
        attn_out, attn_weights = self._mha_forward(
            self.self_attn,
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=return_attention,
        )
        x = x + self.dropout_attn(attn_out)
        if return_attention:
            aux["self_attention"] = attn_weights

        # Norm-first FFN.
        x = x + self.dropout_ffn(self.ffn(self.norm2(x)))

        # Clinical-routed adapter inside the Transformer block.
        if self.use_adapter:
            if clinical_context is None:
                raise ValueError("clinical_context is required when use_adapter=True")
            assert self.routed_adapter is not None
            routed_delta, adapter_aux = self.routed_adapter(
                x,
                clinical_context=clinical_context,
                return_aux=return_aux,
            )
            x = self.adapter_norm(x + self.adapter_scale * routed_delta)  # type: ignore[operator]
            if return_aux:
                aux.update(adapter_aux)

        return x, aux


class ClinicalRoutedTransformerStack(nn.Module):
    """Stack of ClinicalRoutedTransformerEncoderLayer with configurable adapter layers."""

    def __init__(
        self,
        embed_dim: int,
        heads: int,
        num_layers: int,
        dropout: float,
        use_routed_adapters: bool = True,
        adapter_layers: Union[str, Sequence[int], None] = "last",
        num_adapter_experts: int = 3,
        adapter_rank: int = 8,
        adapter_scale: float = 0.1,
        adapter_scale_learnable: bool = False,
        router_tau: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.adapter_layers = parse_adapter_layers(adapter_layers, num_layers)
        if not use_routed_adapters:
            self.adapter_layers = []
        self.adapter_layers_set = set(self.adapter_layers)

        self.layers = nn.ModuleList([
            ClinicalRoutedTransformerEncoderLayer(
                embed_dim=embed_dim,
                heads=heads,
                dropout=dropout,
                use_adapter=(i in self.adapter_layers_set),
                num_adapter_experts=num_adapter_experts,
                adapter_rank=adapter_rank,
                adapter_scale=adapter_scale,
                adapter_scale_learnable=adapter_scale_learnable,
                router_tau=router_tau,
            )
            for i in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        clinical_context: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        aux: Dict[str, Any] = {
            "router_alpha_layers": [],
            "router_logits_layers": [],
            "adapter_expert_outputs": [],
            "adapter_layer_indices": list(self.adapter_layers),
        }
        self_attention_layers: List[Any] = []

        for i, layer in enumerate(self.layers):
            x, layer_aux = layer(
                x,
                clinical_context=clinical_context,
                return_aux=return_aux,
                return_attention=return_attention,
            )
            if return_aux and "router_alpha" in layer_aux:
                aux["router_alpha_layers"].append(layer_aux["router_alpha"])
                aux["router_logits_layers"].append(layer_aux["router_logits"])
                aux["adapter_expert_outputs"].append(layer_aux["expert_outputs_pooled"])
            if return_attention and "self_attention" in layer_aux:
                self_attention_layers.append(layer_aux["self_attention"])

        x = self.out_norm(x)
        if return_aux or return_attention:
            if return_attention:
                aux["methyl_self_attention_layers"] = self_attention_layers
            return x, aux
        return x


# ============================================================
# MeCAlign V2
# ============================================================


class MeCAlignV2(nn.Module):
    """
    MeCAlign V2: Clinical-Routed Low-Rank Adapter MeCAlign.

    Important returned aux fields:
        cpg_gate: [B, num_probes]
        router_alpha_layers: list of [B, K]
        adapter_expert_outputs: list of [B, K, D]
        methyl_pool, clinical_pool, alignment_pool: [B, D]
        methyl_align_z, clinical_align_z: [B, D]
    """

    def __init__(
        self,
        num_region_global: int = 5,
        num_clin_cont: int = 22,
        num_clin_cat: int = 7,
        cat_cardinalities: Optional[Sequence[int]] = None,
        embed_dim: int = 128,
        layers_per_block: int = 2,
        heads: int = 1,
        dropout: float = 0.2,
        probe_fusion_type: str = "film",
        methy_transform: str = "none",
        use_value_mlp: bool = False,
        num_probes: Optional[int] = None,
        p_clin_mask: float = 0.0,
        p_methyl_mask: float = 0.0,
        use_probe_identity_embedding: bool = True,
        clinical_feature_dropout: float = 0.0,
        use_clin_cont_feature_tokens: bool = True,
        use_region_tokens: bool = True,
        disable_region_film: bool = False,
        use_latent_reencoding: bool = True,
        num_latent_tokens: int = 200,
        use_cpg_gate: bool = True,
        gate_tau: float = 1.0,
        use_alignment_tokens: bool = True,
        num_alignment_tokens: int = 8,
        # V2 knobs
        use_routed_adapters: bool = True,
        adapter_layers: Union[str, Sequence[int], None] = "last",
        num_adapter_experts: int = 3,
        adapter_rank: int = 8,
        adapter_scale: float = 0.1,
        adapter_scale_learnable: bool = False,
        router_tau: float = 1.0,
        **unused_kwargs: Any,
    ):
        super().__init__()

        if cat_cardinalities is None:
            raise ValueError("cat_cardinalities must be provided")
        if len(cat_cardinalities) != num_clin_cat:
            raise ValueError("Length of cat_cardinalities must equal num_clin_cat")
        if methy_transform not in ["none", "logit"]:
            raise ValueError("methy_transform must be 'none' or 'logit'")
        if probe_fusion_type not in ["add", "concat", "attention", "film"]:
            raise ValueError("probe_fusion_type must be 'add', 'concat', 'attention', or 'film'")
        if num_probes is None:
            raise ValueError("num_probes must be provided")
        if num_latent_tokens <= 0:
            raise ValueError("num_latent_tokens must be positive")
        if num_alignment_tokens <= 0:
            raise ValueError("num_alignment_tokens must be positive")
        if gate_tau <= 0:
            raise ValueError("gate_tau must be positive")
        if router_tau <= 0:
            raise ValueError("router_tau must be positive")

        self.embed_dim = embed_dim
        self.num_region_global = int(num_region_global)
        self.num_clin_cont = int(num_clin_cont)
        self.num_clin_cat = int(num_clin_cat)
        self.num_probes = int(num_probes)
        self.layers_per_block = int(layers_per_block)
        self.heads = int(heads)
        self.dropout_rate = float(dropout)
        self.probe_fusion_type = probe_fusion_type
        self.methy_transform = methy_transform
        self.use_value_mlp = bool(use_value_mlp)
        self.p_clin_mask = float(p_clin_mask)
        self.p_methyl_mask = float(p_methyl_mask)
        self.use_probe_identity_embedding = bool(use_probe_identity_embedding)
        self.clinical_feature_dropout = float(clinical_feature_dropout)
        self.use_clin_cont_feature_tokens = bool(use_clin_cont_feature_tokens)
        self.use_region_tokens = bool(use_region_tokens)
        self.disable_region_film = bool(disable_region_film)
        self.use_latent_reencoding = bool(use_latent_reencoding)
        self.num_latent_tokens = int(num_latent_tokens)
        self.use_cpg_gate = bool(use_cpg_gate)
        self.gate_tau = float(gate_tau)
        self.use_alignment_tokens = bool(use_alignment_tokens)
        self.num_alignment_tokens = int(num_alignment_tokens)

        self.use_routed_adapters = bool(use_routed_adapters)
        self.adapter_layers = parse_adapter_layers(adapter_layers, self.layers_per_block)
        if not self.use_routed_adapters:
            self.adapter_layers = []
        self.num_adapter_experts = int(num_adapter_experts)
        self.adapter_rank = int(adapter_rank)
        self.adapter_scale = float(adapter_scale)
        self.adapter_scale_learnable = bool(adapter_scale_learnable)
        self.router_tau = float(router_tau)

        # =====================================================
        # 1. CpG methylation value embedding
        # =====================================================
        if self.use_value_mlp:
            self.val_proj = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.val_proj = nn.Linear(1, embed_dim)

        self.probe_id_embedding = (
            nn.Embedding(self.num_probes, embed_dim)
            if self.use_probe_identity_embedding
            else None
        )

        # =====================================================
        # 2. Region-global Static FiLM / fusion
        # =====================================================
        self.use_region_global = self.num_region_global > 0
        if self.use_region_global:
            self.region_proj_global = nn.Sequential(
                nn.Linear(self.num_region_global, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
            )
            if probe_fusion_type == "concat":
                self.probe_concat_proj = nn.Linear(2 * embed_dim, embed_dim)
            elif probe_fusion_type == "attention":
                self.probe_attn = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.probe_attn_norm = nn.LayerNorm(embed_dim)
            elif probe_fusion_type == "film":
                self.static_film_gen = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, 2 * embed_dim),
                )
                self.static_film_norm = nn.LayerNorm(embed_dim)
        else:
            self.region_proj_global = None
        self.probe_dropout = nn.Dropout(dropout)

        # =====================================================
        # 3. Explicit region identity tokens
        # =====================================================
        if self.use_region_tokens and self.num_region_global > 0:
            self.region_value_proj = nn.Linear(1, embed_dim)
            self.region_token_feature_embed = nn.Embedding(self.num_region_global, embed_dim)
            self.region_token_norm = nn.LayerNorm(embed_dim)
            self.region_token_dropout = nn.Dropout(dropout)
        else:
            self.region_value_proj = None
            self.region_token_feature_embed = None
            self.region_token_norm = None
            self.region_token_dropout = None

        # =====================================================
        # 4. Latent query re-encoding
        # =====================================================
        if self.use_latent_reencoding:
            self.latent_queries = nn.Parameter(torch.randn(self.num_latent_tokens, embed_dim))
            self.compression_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.compression_norm1 = nn.LayerNorm(embed_dim)
            self.compression_ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.compression_norm2 = nn.LayerNorm(embed_dim)
            self.latent_pos_embedding = nn.Embedding(self.num_latent_tokens, embed_dim)
            self.latent_pos_dropout = nn.Dropout(dropout)
        else:
            self.latent_queries = None
            self.compression_attention = None
            self.compression_norm1 = None
            self.compression_ffn = None
            self.compression_norm2 = None
            self.latent_pos_embedding = None
            self.latent_pos_dropout = None

        # =====================================================
        # 5. Clinical preprocessing
        # =====================================================
        if self.use_clin_cont_feature_tokens:
            self.clin_cont_proj = None
            self.clin_cont_value_proj = nn.Linear(1, embed_dim)
            self.clin_cont_feature_embed = nn.Embedding(self.num_clin_cont, embed_dim)
            num_clin_tokens_for_identity = self.num_clin_cont + self.num_clin_cat
        else:
            self.clin_cont_proj = nn.Linear(self.num_clin_cont, embed_dim)
            self.clin_cont_value_proj = None
            self.clin_cont_feature_embed = None
            num_clin_tokens_for_identity = 1 + self.num_clin_cat

        self.cat_embeds = nn.ModuleList([nn.Embedding(int(card), embed_dim) for card in cat_cardinalities])
        self.clin_dropout = nn.Dropout(dropout)
        self.clin_feature_type_embed = nn.Embedding(num_clin_tokens_for_identity, embed_dim)
        self.clin_token_dropout = nn.Dropout(clinical_feature_dropout)

        # =====================================================
        # 6. Encoders
        # =====================================================
        self.methyl_encoder = ClinicalRoutedTransformerStack(
            embed_dim=embed_dim,
            heads=heads,
            num_layers=layers_per_block,
            dropout=dropout,
            use_routed_adapters=self.use_routed_adapters,
            adapter_layers=self.adapter_layers,
            num_adapter_experts=self.num_adapter_experts,
            adapter_rank=self.adapter_rank,
            adapter_scale=self.adapter_scale,
            adapter_scale_learnable=self.adapter_scale_learnable,
            router_tau=self.router_tau,
        )
        self.clin_encoder = TransformerStack(
            embed_dim=embed_dim,
            heads=heads,
            num_layers=layers_per_block,
            dropout=dropout,
        )

        # =====================================================
        # 7. Pooling and clinical-guided CpG gate
        # =====================================================
        self.clin_pool_for_gate = AttentionPool1D(embed_dim, dropout)
        self.methyl_pool = AttentionPool1D(embed_dim, dropout)
        self.clin_pool = AttentionPool1D(embed_dim, dropout)
        self.align_pool = AttentionPool1D(embed_dim, dropout)

        self.cpg_gate_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, self.num_probes),
        )

        # =====================================================
        # 8. Shared alignment tokens
        # =====================================================
        if self.use_alignment_tokens:
            self.alignment_tokens = nn.Parameter(torch.randn(self.num_alignment_tokens, embed_dim))
            self.alignment_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.alignment_norm1 = nn.LayerNorm(embed_dim)
            self.alignment_ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            self.alignment_norm2 = nn.LayerNorm(embed_dim)
        else:
            self.alignment_tokens = None
            self.alignment_attn = None
            self.alignment_norm1 = None
            self.alignment_ffn = None
            self.alignment_norm2 = None

        # Projection heads for optional alignment loss.
        self.methyl_align_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.clin_align_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # =====================================================
        # 9. Classification head
        # =====================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    @staticmethod
    def _mha_forward(
        mha: nn.MultiheadAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        try:
            return mha(
                query=query,
                key=key,
                value=value,
                need_weights=need_weights,
                average_attn_weights=False,
            )
        except TypeError:
            return mha(query=query, key=key, value=value, need_weights=need_weights)

    def _build_clinical_tokens(
        self,
        x_clin_cont: torch.Tensor,
        x_clin_cat: torch.Tensor,
    ) -> torch.Tensor:
        x_clin_cont = x_clin_cont.float()
        x_clin_cat = x_clin_cat.long()

        if self.training and self.p_clin_mask > 0:
            clin_mask = (torch.rand_like(x_clin_cont) > self.p_clin_mask).float()
            x_clin_cont = x_clin_cont * clin_mask

        if self.use_clin_cont_feature_tokens:
            cont_tokens = self.clin_cont_value_proj(x_clin_cont.unsqueeze(-1))
            cont_ids = torch.arange(self.num_clin_cont, device=x_clin_cont.device)
            cont_tokens = cont_tokens + self.clin_cont_feature_embed(cont_ids).unsqueeze(0)
        else:
            cont_tokens = self.clin_cont_proj(x_clin_cont).unsqueeze(1)

        cat_tokens: List[torch.Tensor] = []
        for i, emb in enumerate(self.cat_embeds):
            cat_tokens.append(emb(x_clin_cat[:, i]))

        if cat_tokens:
            cat_tokens_tensor = torch.stack(cat_tokens, dim=1)
            clin_tokens = torch.cat([cont_tokens, cat_tokens_tensor], dim=1)
        else:
            clin_tokens = cont_tokens

        token_ids = torch.arange(clin_tokens.size(1), device=clin_tokens.device)
        clin_tokens = clin_tokens + self.clin_feature_type_embed(token_ids).unsqueeze(0)
        clin_tokens = self.clin_dropout(clin_tokens)
        clin_tokens = self.clin_token_dropout(clin_tokens)
        return clin_tokens

    def _build_probe_tokens(
        self,
        x_methy: torch.Tensor,
        x_region_global: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        aux: Dict[str, torch.Tensor] = {}

        if self.methy_transform == "logit":
            eps = 1e-6
            x_val = torch.clamp(x_methy.float(), min=eps, max=1.0 - eps)
            x_val = torch.log(x_val / (1.0 - x_val))
        else:
            x_val = x_methy.float()

        if int(x_val.shape[1]) != int(self.num_probes):
            raise ValueError(f"Expected {self.num_probes} probes, got {x_val.shape[1]}")

        val_emb = self.val_proj(x_val.unsqueeze(-1))

        if self.use_probe_identity_embedding and self.probe_id_embedding is not None:
            probe_ids = torch.arange(self.num_probes, device=x_val.device)
            val_emb = val_emb + self.probe_id_embedding(probe_ids).unsqueeze(0)

        if (not self.use_region_global) or self.disable_region_film or x_region_global.shape[1] == 0:
            probe_tokens = self.probe_dropout(val_emb)
            if return_aux:
                return probe_tokens, aux
            return probe_tokens

        if x_region_global.shape[1] != self.num_region_global:
            raise ValueError(
                f"Expected {self.num_region_global} region-global features, got {x_region_global.shape[1]}"
            )

        region_repr = self.region_proj_global(x_region_global.float())
        region_b = region_repr.unsqueeze(1).expand(-1, self.num_probes, -1)

        if self.probe_fusion_type == "add":
            probe_tokens = val_emb + region_b
        elif self.probe_fusion_type == "concat":
            probe_tokens = self.probe_concat_proj(torch.cat([val_emb, region_b], dim=-1))
        elif self.probe_fusion_type == "attention":
            attn_out, probe_region_attn = self._mha_forward(
                self.probe_attn,
                query=val_emb,
                key=region_repr.unsqueeze(1),
                value=region_repr.unsqueeze(1),
                need_weights=return_aux,
            )
            if return_aux:
                aux["probe_to_region_attention"] = probe_region_attn
            probe_tokens = self.probe_attn_norm(val_emb + attn_out)
        else:  # film
            film_params = self.static_film_gen(region_repr)
            gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)
            gamma = torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            if return_aux:
                aux["static_film_gamma"] = gamma
                aux["static_film_beta"] = beta
            probe_tokens = self.static_film_norm((1.0 + gamma) * val_emb + beta)

        probe_tokens = self.probe_dropout(probe_tokens)
        if return_aux:
            return probe_tokens, aux
        return probe_tokens

    def _build_region_tokens(self, x_region_global: torch.Tensor) -> torch.Tensor:
        if self.region_value_proj is None:
            raise RuntimeError("Region tokens requested, but region token layers were not initialized.")
        if x_region_global.shape[1] != self.num_region_global:
            raise ValueError(
                f"Expected {self.num_region_global} region-global features, got {x_region_global.shape[1]}"
            )
        region_tokens = self.region_value_proj(x_region_global.float().unsqueeze(-1))
        region_ids = torch.arange(self.num_region_global, device=x_region_global.device)
        region_tokens = region_tokens + self.region_token_feature_embed(region_ids).unsqueeze(0)
        region_tokens = self.region_token_norm(region_tokens)
        return self.region_token_dropout(region_tokens)

    def _latent_reencode(
        self,
        methyl_input: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.use_latent_reencoding:
            if return_attention:
                return methyl_input, None
            return methyl_input

        batch_size = methyl_input.shape[0]
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, attn_weights = self._mha_forward(
            self.compression_attention,
            query=queries,
            key=methyl_input,
            value=methyl_input,
            need_weights=return_attention,
        )
        latent_tokens = self.compression_norm1(queries + attn_out)
        latent_tokens = self.compression_norm2(latent_tokens + self.compression_ffn(latent_tokens))

        pos_ids = torch.arange(self.num_latent_tokens, device=methyl_input.device)
        latent_tokens = self.latent_pos_dropout(
            latent_tokens + self.latent_pos_embedding(pos_ids).unsqueeze(0)
        )

        if return_attention:
            return latent_tokens, attn_weights
        return latent_tokens

    def _make_cpg_gate(self, clinical_pool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.cpg_gate_mlp(clinical_pool)
        if self.use_cpg_gate:
            gate = torch.softmax(gate_logits / self.gate_tau, dim=-1) * float(self.num_probes)
        else:
            gate = torch.ones_like(gate_logits)
        return gate, gate_logits

    def _alignment_fusion(
        self,
        methyl_tokens: torch.Tensor,
        clinical_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = methyl_tokens.size(0)
        if not self.use_alignment_tokens:
            zeros = torch.zeros(
                batch_size,
                self.num_alignment_tokens,
                self.embed_dim,
                device=methyl_tokens.device,
                dtype=methyl_tokens.dtype,
            )
            return zeros, None

        memory = torch.cat([methyl_tokens, clinical_tokens], dim=1)
        align_query = self.alignment_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        align_out, align_attn = self._mha_forward(
            self.alignment_attn,
            query=align_query,
            key=memory,
            value=memory,
            need_weights=return_attention,
        )
        align_tokens = self.alignment_norm1(align_query + align_out)
        align_tokens = self.alignment_norm2(align_tokens + self.alignment_ffn(align_tokens))
        return align_tokens, align_attn

    def forward(
        self,
        x_methy: torch.Tensor,
        x_region_global: torch.Tensor,
        x_clin_cont: torch.Tensor,
        x_clin_cat: torch.Tensor,
        return_attention: bool = False,
        return_intermediate: bool = False,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        want_aux = return_attention or return_intermediate or return_aux
        aux: Dict[str, Any] = {}

        x_methy = x_methy.float()
        x_region_global = x_region_global.float()
        x_clin_cont = x_clin_cont.float()
        x_clin_cat = x_clin_cat.long()

        # Clinical branch first because it generates CpG gate and adapter router context.
        clinical_raw_tokens = self._build_clinical_tokens(x_clin_cont, x_clin_cat)
        clinical_tokens = self.clin_encoder(clinical_raw_tokens)
        clinical_pool_for_gate = self.clin_pool_for_gate(clinical_tokens)
        cpg_gate, cpg_gate_logits = self._make_cpg_gate(clinical_pool_for_gate)

        # Methylation branch: CpG + Static FiLM from raw ISLAND values.
        if return_attention:
            cpg_tokens, probe_aux = self._build_probe_tokens(
                x_methy,
                x_region_global,
                return_aux=True,
            )
            aux.update(probe_aux)
        else:
            cpg_tokens = self._build_probe_tokens(x_methy, x_region_global, return_aux=False)

        # Clinical-guided CpG gate is applied only to the original CpG tokens.
        gated_cpg_tokens = cpg_tokens * cpg_gate.unsqueeze(-1)

        methyl_input = gated_cpg_tokens
        num_region_tokens = 0
        if self.use_region_tokens and x_region_global.shape[1] > 0:
            region_tokens = self._build_region_tokens(x_region_global)
            num_region_tokens = int(region_tokens.shape[1])
            methyl_input = torch.cat([methyl_input, region_tokens], dim=1)
        else:
            region_tokens = None

        if self.training and self.p_methyl_mask > 0:
            methyl_mask = (
                torch.rand(methyl_input.size(0), methyl_input.size(1), 1, device=methyl_input.device)
                > self.p_methyl_mask
            ).float()
            methyl_input = methyl_input * methyl_mask

        # Optional latent query re-encoding.
        if return_attention:
            methyl_latent_tokens, compression_attn = self._latent_reencode(
                methyl_input,
                return_attention=True,
            )
            aux["compression_latent_to_probe"] = compression_attn
        else:
            methyl_latent_tokens = self._latent_reencode(methyl_input, return_attention=False)
        
        # V2: methylation encoder gets clinical context for routed adapters.
        if want_aux:
            methyl_tokens, methyl_aux = self.methyl_encoder(
                methyl_latent_tokens,
                clinical_context=clinical_pool_for_gate,
                return_aux=True,
                return_attention=return_attention,
            )
            aux.update(methyl_aux)
        else:
            methyl_tokens = self.methyl_encoder(
                methyl_latent_tokens,
                clinical_context=clinical_pool_for_gate,
                return_aux=False,
                return_attention=False,
            )
        
        # Shared alignment tokens attend to methylation + clinical tokens.
        align_tokens, align_attn = self._alignment_fusion(
            methyl_tokens,
            clinical_tokens,
            return_attention=return_attention,
        )

        # ------------------------------
        # Pool and classify.
        # ------------------------------
        if return_attention:
            methyl_pool, methyl_pool_w = self.methyl_pool(methyl_tokens, return_weights=True)
            clinical_pool, clinical_pool_w = self.clin_pool(clinical_tokens, return_weights=True)
            align_pool, align_pool_w = self.align_pool(align_tokens, return_weights=True)
            aux["methyl_pool_weights"] = methyl_pool_w
            aux["clinical_pool_weights"] = clinical_pool_w
            aux["alignment_pool_weights"] = align_pool_w
        else:
            methyl_pool = self.methyl_pool(methyl_tokens)
            clinical_pool = self.clin_pool(clinical_tokens)
            align_pool = self.align_pool(align_tokens)

        final_repr = torch.cat([methyl_pool, clinical_pool, align_pool], dim=-1)
        logits = self.classifier(final_repr).squeeze(-1)

        if want_aux:
            methyl_align_h = self.methyl_align_proj(methyl_pool)
            clinical_align_h = self.clin_align_proj(clinical_pool)

            aux.update({
                "cpg_gate": cpg_gate,
                "cpg_gate_logits": cpg_gate_logits,
                "methyl_pool": methyl_pool,
                "clinical_pool": clinical_pool,
                "alignment_pool": align_pool,

                # New raw projected embeddings for Barlow / anti-collapse loss
                "methyl_align_h": methyl_align_h,
                "clinical_align_h": clinical_align_h,

                # Existing normalized embeddings for cosine analysis
                "methyl_align_z": F.normalize(methyl_align_h, dim=-1),
                "clinical_align_z": F.normalize(clinical_align_h, dim=-1),

                "final_repr": final_repr if return_intermediate else None,
                "methyl_tokens": methyl_tokens if return_intermediate else None,
                "clinical_tokens": clinical_tokens if return_intermediate else None,
                "alignment_tokens": align_tokens if return_intermediate else None,
                "alignment_attention": align_attn,
                "meta": {
                    "num_original_cpg_tokens": int(self.num_probes),
                    "num_region_tokens": int(num_region_tokens),
                    "num_methyl_input_tokens": int(methyl_input.shape[1]),
                    "num_methyl_encoded_tokens": int(methyl_tokens.shape[1]),
                    "num_clinical_tokens": int(clinical_tokens.shape[1]),
                    "num_alignment_tokens": int(align_tokens.shape[1]),
                    "use_latent_reencoding": bool(self.use_latent_reencoding),
                    "use_cpg_gate": bool(self.use_cpg_gate),
                    "use_alignment_tokens": bool(self.use_alignment_tokens),
                    "use_routed_adapters": bool(self.use_routed_adapters),
                    "adapter_layer_indices": list(getattr(self.methyl_encoder, "adapter_layer_indices", [])),
                    "attention_layout": {
                        "cpg_gate": "clinical pooled representation -> per-CpG scale-preserving gate",
                        "compression_latent_to_probe": "query=latent_tokens, key/value=gated CpG tokens plus optional region tokens",
                        "alignment_attention": "query=shared alignment tokens, key/value=methylation encoded tokens plus clinical encoded tokens",
                    },
                },
            })
            return logits, aux

        return logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=max(0.02, self.embed_dim ** -0.5))
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.use_latent_reencoding and self.latent_queries is not None:
            nn.init.normal_(self.latent_queries, mean=0.0, std=max(0.02, self.embed_dim ** -0.5))
        if self.use_alignment_tokens and self.alignment_tokens is not None:
            nn.init.normal_(self.alignment_tokens, mean=0.0, std=max(0.02, self.embed_dim ** -0.5))

"""
GPT model with Mixture of Experts (MoE) support and MLP parameter sharing (loop transformer).
Based on dev-hetero/gpt_moe.py with the following additions:
- Configurable MLP parameter sharing across layers (shared_mlp_groups)
- Each layer keeps its own attention, but layers in a sharing group reuse the same MLP
- Assert that shared layers are all dense or all MoE (no mixing)

Design:
- MoE: 1 shared expert (always active) + 7 routed experts (top-3 selected) = 4 active
- Each expert has intermediate_dim = n_embd (1/4 of dense MLP's 4*n_embd)
- Total MoE layer params = 2x dense, active FLOPs = 1x dense
- Parameter sharing: layers in a group share the same MLP module (attention stays independent)
"""

from functools import partial
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    # MoE config
    moe_layers: str = "2:"  # slice notation for which layers use MoE (e.g. "2:", ":4", "2:8")
    n_routed_experts: int = 7
    moe_top_k: int = 3
    balance_loss_alpha: float = 0.0001
    balance_bias_gamma: float = 0.001
    # Parameter sharing config
    shared_mlp_groups: str = ""  # slice notation for which layers share MLP (e.g. "3:6" = layers 3,4,5 share)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, intermediate_dim=None):
        super().__init__()
        intermediate_dim = intermediate_dim or 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, intermediate_dim, bias=False)
        self.c_proj = nn.Linear(intermediate_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class MoERouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.moe_top_k
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, config.n_embd))
        self.register_buffer("balance_bias", torch.zeros(self.n_routed_experts))

    def forward(self, x):
        # x: (B, T, n_embd) or (B*T, n_embd)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        # Compute router logits in float32 for stability
        logits = F.linear(x_flat.float(), self.weight.float())  # (N, n_routed_experts)
        scores = torch.sigmoid(logits)  # original scores (no bias)
        # Add bias only for selection (not for gating values)
        scores_for_selection = scores + self.balance_bias
        # Top-k selection
        topk_weights, topk_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1)
        # Gather original scores (without bias) for the selected experts
        topk_weights = scores.gather(1, topk_indices)
        # Normalize
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_indices, topk_weights, logits


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        expert_dim = config.n_embd  # 1/4 of default 4*n_embd
        self.shared_expert = MLP(config, intermediate_dim=expert_dim)
        self.experts = nn.ModuleList([
            MLP(config, intermediate_dim=expert_dim)
            for _ in range(config.n_routed_experts)
        ])
        self.router = MoERouter(config)
        # Accumulated during forward for balance loss computation (list for shared layers)
        self._router_logits_list = []
        self._topk_indices_list = []
        # Accumulated expert counts across micro-batches for bias updates
        self.register_buffer("_accumulated_expert_counts", torch.zeros(config.n_routed_experts))
        self._accumulated_tokens = 0

    def forward(self, x):
        # Shared expert (always active)
        shared_out = self.shared_expert(x)

        # Route tokens to experts
        B, T, C = x.shape
        topk_indices, topk_weights, router_logits = self.router(x)
        # Accumulate for balance loss (supports shared MoE layers called multiple times)
        self._router_logits_list.append(router_logits)
        self._topk_indices_list.append(topk_indices)
        # Accumulate expert counts for bias updates across gradient accumulation
        with torch.no_grad():
            N = topk_indices.shape[0]
            for k in range(self.config.moe_top_k):
                self._accumulated_expert_counts.scatter_add_(
                    0, topk_indices[:, k],
                    torch.ones(N, device=topk_indices.device))
            self._accumulated_tokens += N

        # Compute routed expert outputs
        x_flat = x.view(-1, C)  # (N, C)
        routed_out = torch.zeros_like(x_flat)

        # Loop over experts that have at least one assigned token
        with torch.no_grad():
            expert_mask = F.one_hot(topk_indices, num_classes=self.config.n_routed_experts)  # (N, top_k, n_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (n_experts, top_k, N)
            expert_hit = torch.greater(expert_mask.sum(dim=(1, 2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            # Find which tokens are assigned to this expert and which top-k slot
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])  # top_k_pos, token_idx
            current_state = x_flat[token_idx]
            current_out = self.experts[expert_idx](current_state)
            # Weight by gating value
            current_out = current_out * topk_weights[token_idx, top_k_pos, None]
            routed_out.index_add_(0, token_idx, current_out.to(routed_out.dtype))

        routed_out = routed_out.view(B, T, C)
        return shared_out + routed_out


def _parse_moe_layers(moe_layers_str, n_layer):
    """Parse slice notation string into a set of layer indices.
    Examples: "2:" -> {2,3,...,n_layer-1}, ":4" -> {0,1,2,3}, "2:8" -> {2,3,4,5,6,7}, "" -> {}
    """
    if not moe_layers_str or moe_layers_str.strip() == "":
        return set()
    s = moe_layers_str.strip()
    parts = s.split(":")
    if len(parts) == 1:
        # Single index
        return {int(parts[0])}
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else n_layer
        return set(range(start, end))
    else:
        raise ValueError(f"Invalid moe_layers format: {moe_layers_str}")


def _parse_shared_mlp_groups(shared_mlp_groups_str, n_layer):
    """Parse slice notation string into a sorted list of layer indices that share MLP params.
    Examples: "3:6" -> [3, 4, 5], "" -> []
    """
    if not shared_mlp_groups_str or shared_mlp_groups_str.strip() == "":
        return []
    s = shared_mlp_groups_str.strip()
    parts = s.split(":")
    if len(parts) == 1:
        return [int(parts[0])]
    elif len(parts) == 2:
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else n_layer
        return list(range(start, end))
    else:
        raise ValueError(f"Invalid shared_mlp_groups format: {shared_mlp_groups_str}")


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        if use_moe:
            self.mlp = MoELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.moe_layer_indices = _parse_moe_layers(config.moe_layers, config.n_layer)
        self.shared_mlp_layer_indices = _parse_shared_mlp_groups(config.shared_mlp_groups, config.n_layer)
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config, layer_idx, use_moe=(layer_idx in self.moe_layer_indices))
                for layer_idx in range(config.n_layer)
            ]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Apply MLP parameter sharing
        if self.shared_mlp_layer_indices:
            layers = self.shared_mlp_layer_indices
            # Assert: no mixing dense and MoE in a sharing group
            types_in_group = [layer_idx in self.moe_layer_indices for layer_idx in layers]
            assert all(t == types_in_group[0] for t in types_in_group), \
                f"Shared MLP group {layers}: cannot mix dense and MoE layers. " \
                f"Types: {dict(zip(layers, ['moe' if t else 'dense' for t in types_in_group]))}"
            # Replace later blocks' .mlp with the canonical (first) block's .mlp
            canonical = layers[0]
            for layer_idx in layers[1:]:
                self.transformer.h[layer_idx].mlp = self.transformer.h[canonical].mlp

        # Print MoE info
        n_moe = len(self.moe_layer_indices)
        n_dense = config.n_layer - n_moe
        if n_moe > 0:
            print0(f"MoE config: {n_dense} dense layers + {n_moe} MoE layers (1 shared + {config.n_routed_experts} routed, top-{config.moe_top_k})")
            print0(f"MoE layer indices: {sorted(self.moe_layer_indices)}")

        # Print sharing info
        if self.shared_mlp_layer_indices:
            print0(f"MLP sharing: layers {self.shared_mlp_layer_indices} share MLP params (canonical: layer {self.shared_mlp_layer_indices[0]})")

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model.
        Dense layers: same as original nanochat.
        MoE layers: expert c_fc -> uniform, expert c_proj -> zeros, router -> normal.
        Shared MLP layers: only initialize the canonical (first) block's MLP.
        """
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Determine which layers are non-canonical shared (skip MLP init for them)
        shared_non_canonical = set()
        if self.shared_mlp_layer_indices:
            shared_non_canonical = set(self.shared_mlp_layer_indices[1:])

        for i, block in enumerate(self.transformer.h):
            # Attention init (same for all layers, always independent)
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

            # Skip MLP init for non-canonical shared layers (same object, already initialized)
            if i in shared_non_canonical:
                continue

            if i in self.moe_layer_indices:
                # MoE layer init
                moe = block.mlp
                # Shared expert
                torch.nn.init.uniform_(moe.shared_expert.c_fc.weight, -s, s)
                torch.nn.init.zeros_(moe.shared_expert.c_proj.weight)
                # Routed experts
                for expert in moe.experts:
                    torch.nn.init.uniform_(expert.c_fc.weight, -s, s)
                    torch.nn.init.zeros_(expert.c_proj.weight)
                # Router
                torch.nn.init.normal_(moe.router.weight, mean=0.0, std=n_embd**-0.5)
                # balance_bias is already zeros from register_buffer
            else:
                # Dense MLP init (same as original)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        For MoE layers, only count activated params (shared + top_k routed experts).
        Shared MLP params are counted once per layer they appear in (FLOPs = actual compute).
        """
        # Start with non-MoE parameters
        total_active_matmul_params = 0
        # Count lm_head
        total_active_matmul_params += self.lm_head.weight.numel()

        for i, block in enumerate(self.transformer.h):
            # Attention params (always fully active)
            attn = block.attn
            total_active_matmul_params += attn.c_q.weight.numel()
            total_active_matmul_params += attn.c_k.weight.numel()
            total_active_matmul_params += attn.c_v.weight.numel()
            total_active_matmul_params += attn.c_proj.weight.numel()
            if attn.ve_gate is not None:
                total_active_matmul_params += attn.ve_gate.weight.numel()

            if i in self.moe_layer_indices:
                moe = block.mlp
                # Shared expert (always active)
                total_active_matmul_params += moe.shared_expert.c_fc.weight.numel()
                total_active_matmul_params += moe.shared_expert.c_proj.weight.numel()
                # Only top_k routed experts are active per token
                one_expert_params = (moe.experts[0].c_fc.weight.numel() +
                                     moe.experts[0].c_proj.weight.numel())
                total_active_matmul_params += self.config.moe_top_k * one_expert_params
                # Router weight is tiny, skip it (it's a classifier, not compute)
            else:
                # Dense MLP (fully active)
                total_active_matmul_params += block.mlp.c_fc.weight.numel()
                total_active_matmul_params += block.mlp.c_proj.weight.numel()

        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq

        num_flops_per_token = 6 * total_active_matmul_params + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        For shared MLP layers, count shared params once per layer they appear in
        (so that scaling params match the non-shared baseline for fair ablation).
        """
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        # Count per-block with fresh memo each time, so shared params are counted N times
        transformer_matrices = sum(
            sum(p.numel() for p in block.parameters())
            for block in self.transformer.h
        )
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars

        # Unique param count (deduped) for reporting
        transformer_matrices_unique = sum(p.numel() for p in self.transformer.h.parameters())
        total_unique = wte + value_embeds + lm_head + transformer_matrices_unique + scalars
        assert total_unique == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"

        # Count MoE-specific stats (per-layer, shared counted N times)
        moe_total = 0  # total params in MoE layers
        moe_inactive = 0  # params in inactive (non-selected) experts
        moe_router = 0  # router/gating params in MoE layers
        for i, block in enumerate(self.transformer.h):
            if i in self.moe_layer_indices:
                moe = block.mlp
                moe_total += sum(p.numel() for p in moe.parameters())
                moe_router += sum(p.numel() for p in moe.router.parameters())
                # Inactive = (n_routed - top_k) experts worth of params
                one_expert_params = sum(p.numel() for p in moe.experts[0].parameters())
                n_inactive = self.config.n_routed_experts - self.config.moe_top_k
                moe_inactive += n_inactive * one_expert_params

        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'transformer_matrices_unique': transformer_matrices_unique,
            'scalars': scalars,
            'total': total,
            'total_unique': total_unique,
            'moe_total': moe_total,
            'moe_inactive': moe_inactive,
            'moe_router': moe_router,
            'total_active': total - moe_inactive,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate router params from matrix params (named_parameters deduplicates shared params)
        router_params = []
        matrix_params = []
        for name, p in self.transformer.h.named_parameters():
            if 'router' in name:
                router_params.append(p)
            else:
                matrix_params.append(p)

        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        # Verify all params are accounted for (self.parameters() deduplicates)
        total_grouped = (len(matrix_params) + len(router_params) + len(embedding_params) +
                        len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        assert len(list(self.parameters())) == total_grouped, \
            f"Parameter count mismatch: {len(list(self.parameters()))} vs {total_grouped}"

        # Scale the LR for the AdamW parameters
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Router params -> AdamW (same LR as unembedding)
        if router_params:
            param_groups.append(dict(
                kind='adamw', params=router_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))

        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _compute_balance_loss(self, moe_layer):
        """
        Compute the auxiliary balance loss for a single MoE layer.
        Switch Transformer style: alpha * N * sum(f_i * p_i)
        where f_i = fraction of tokens routed to expert i,
              p_i = mean routing probability for expert i.
        Handles shared MoE layers by aggregating stats from all forward passes.
        """
        if not moe_layer._router_logits_list:
            return 0.0

        # Concatenate stats from all forward passes (supports shared layers)
        router_logits = torch.cat(moe_layer._router_logits_list, dim=0)  # (total_N, n_experts)
        topk_indices = torch.cat(moe_layer._topk_indices_list, dim=0)  # (total_N, top_k)

        n_experts = self.config.n_routed_experts
        N = router_logits.shape[0]

        # f_i: fraction of tokens routed to each expert
        expert_counts = torch.zeros(n_experts, device=router_logits.device)
        for k in range(self.config.moe_top_k):
            expert_counts.scatter_add_(0, topk_indices[:, k], torch.ones(N, device=router_logits.device))
        f = expert_counts / (N * self.config.moe_top_k)  # fraction assigned

        # p_i: mean routing probability for each expert
        p = torch.sigmoid(router_logits).mean(dim=0)

        balance_loss = self.config.balance_loss_alpha * n_experts * (f * p).sum()
        return balance_loss

    def update_balance_bias(self):
        """
        Update balance bias buffers for all MoE layers (DeepSeek-V3 style).
        Called after each optimizer step. Uses expert counts accumulated across
        all micro-batches in the gradient accumulation cycle.
        Handles shared MoE layers by deduplicating via object identity.
        """
        gamma = self.config.balance_bias_gamma
        # Track which MoE layers we've already processed (for shared layers)
        processed_moe_ids = set()
        for i, block in enumerate(self.transformer.h):
            if i not in self.moe_layer_indices:
                continue
            moe = block.mlp
            moe_id = id(moe)
            if moe_id in processed_moe_ids:
                continue
            processed_moe_ids.add(moe_id)

            if moe._accumulated_tokens == 0:
                continue

            n_experts = self.config.n_routed_experts
            expert_counts = moe._accumulated_expert_counts
            # Expected count per expert if balanced
            expected = moe._accumulated_tokens * self.config.moe_top_k / n_experts
            # Increase bias for underloaded experts, decrease for overloaded
            moe.router.balance_bias += gamma * (expected - expert_counts).sign()
            # Reset accumulators
            moe._accumulated_expert_counts.zero_()
            moe._accumulated_tokens = 0
            # Clear saved state
            moe._router_logits_list = []
            moe._topk_indices_list = []

    def _clear_moe_stats(self):
        """Clear accumulated routing stats before each forward pass."""
        processed_moe_ids = set()
        for i, block in enumerate(self.transformer.h):
            if i not in self.moe_layer_indices:
                continue
            moe = block.mlp
            moe_id = id(moe)
            if moe_id in processed_moe_ids:
                continue
            processed_moe_ids.add(moe_id)
            moe._router_logits_list = []
            moe._topk_indices_list = []

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Clear MoE routing stats before forward (important for shared MoE layers)
        if self.moe_layer_indices:
            self._clear_moe_stats()

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # Add balance loss from all MoE layers
            if self.moe_layer_indices:
                # Use processed_moe_ids to avoid double-counting shared MoE layers
                processed_moe_ids = set()
                balance_loss = 0.0
                for i, block in enumerate(self.transformer.h):
                    if i in self.moe_layer_indices:
                        moe_id = id(block.mlp)
                        if moe_id not in processed_moe_ids:
                            processed_moe_ids.add(moe_id)
                            balance_loss += self._compute_balance_loss(block.mlp)
                loss = loss + balance_loss
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

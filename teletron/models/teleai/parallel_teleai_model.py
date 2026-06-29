from typing import Tuple, Optional
import torch
import torch.nn as nn
from teletron.core.context_parallel import ContextParallelMixin
from teletron.core.tensor_parallel import TensorParallelMixin
from teletron.core.transformer import TransformerGeneralMixin
from .teleai_model import TeleaiModel, DiTBlock, sinusoidal_embedding_1d, AttentionModule, TeleaiLogitsModel
import logging
from teletron.utils import set_config
# from megatron.core import mpu
from torch.utils.checkpoint import checkpoint
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

class ParallelTeleaiDitBlock(TensorParallelMixin, ContextParallelMixin, DiTBlock):
    def __init__(
        self, 
        config,
        has_image_input: bool, 
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        **kwargs,
        ):
        DiTBlock.__init__(self, has_image_input, dim, num_heads, ffn_dim, eps, **kwargs)
        # from ContextParallelMixin
        self.enable_context_parallel(self.self_attn.attn)
        
        # from Tensor ParallelMixin
        self.enable_ffn_tensor_parallel(self.ffn, config)
        self.enable_self_attn_tensor_parallel(self.self_attn, config)
        self.enable_cross_attn_tensor_parallel(self.cross_attn, config)
        
    def forward(self, x, context, t_mod, freqs):
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        modulation = modulation + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

        normed_x1 = self.norm1(x)
        modulated_x1 = self.modulate_with_cp_grad_reduce(normed_x1, shift_msa, scale_msa)
        attn_output = self.self_attn(modulated_x1, freqs)
        gated_x1 = self.gate_with_cp_grad_reduce(x, gate_msa, attn_output)

        normed_x3 = self.norm3(gated_x1)
        cross_attn_output = self.cross_attn(normed_x3, context)
        x = gated_x1 + cross_attn_output

        normed_x2 = self.norm2(x)
        modulated_x2 = self.modulate_with_cp_grad_reduce(normed_x2, shift_mlp, scale_mlp)
        ffn_output = self.ffn(modulated_x2)
        x = self.gate_with_cp_grad_reduce(x, gate_mlp, ffn_output)

        return x

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0, device="cuda"):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class ParallelTeleaiModel(ContextParallelMixin, TensorParallelMixin, TransformerGeneralMixin, TeleaiModel):
    def __init__(self, config):
        dit_model_config = set_config().get('model_config', None).get('dit', None).get('config', None)
        TeleaiModel.__init__(self, **dit_model_config)
        self.config = config

        self.blocks = nn.ModuleList([
            ParallelTeleaiDitBlock(self.config, self.has_image_input, self.dim, self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])

        # from TransformerGeneralMixin
        from teletron.utils import get_args
        args = get_args()
        if args.activation_offload:
            self.enable_activation_offload(self.blocks)
        else:
            self.enable_activation_checkpointing(self.blocks)

        # from ContextParallelMixin
        self.register_cp_grad_reduce_hook()

    def register_cp_grad_reduce_hook(self):

        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_emb") or \
                name.startswith("time") or \
                    name.startswith("head") or \
                    "modulation" in name:
                continue

            param.register_hook(self.cp_grad_reduce)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        cn_images=None,
        **kwargs,
    ):
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
        t = self.time_emb(t_emb)
        t_mod = self.time_proj(t)
        t_mod = t_mod.unflatten(1, (6, self.dim))

        context_emb = self.text_emb(context)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        if self.has_image_input:
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_embedding, context_emb], dim=1)

        if cn_images is not None:
            x = torch.cat([x, cn_images], dim=1)

        x, (f, h, w) = self.patchify(x)

        head_dim = self.dim // self.num_heads
        freq_f = precompute_freqs_cis(head_dim - 2 * (head_dim // 3), f).view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h = precompute_freqs_cis(head_dim // 3, h).view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w = precompute_freqs_cis(head_dim // 3, w).view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([freq_f, freq_h, freq_w], dim=-1)
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        # freqs = self.split_input(freqs, dim=-1)
        x = self.blocks(x, context_emb, t_mod, freqs)
        x = self.gather_output(x, dim=1)

        # Now x is in full shape, just do regular forward
        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()




class ParallelTeleaiLogitsModel(ContextParallelMixin, TensorParallelMixin, TransformerGeneralMixin, TeleaiLogitsModel):
    def __init__(self, config):
        dit_model_config = set_config().get('model_config', None).get('ddit', None).get('config', None)
        TeleaiLogitsModel.__init__(self, **dit_model_config)
        self.config = config

        self.blocks = nn.ModuleList([
            ParallelTeleaiDitBlock(self.config, self.has_image_input, self.dim, self.num_heads, self.ffn_dim, self.eps)
            for _ in range(self.num_layers)
        ])

        # from TransformerGeneralMixin
        # from teletron.utils import get_args
        # args = get_args()
        # if args.activation_offload:
        #     self.enable_activation_offload(self.blocks)
        # else:
        #     self.enable_activation_checkpointing(self.blocks)

        # from ContextParallelMixin
        self.register_cp_grad_reduce_hook()

    def register_cp_grad_reduce_hook(self):

        # layers with parallel input sequence need to reduce its param gradient.
        # list the parameters that needs grad reduce and register tensor grad hook

        for name, param in self.named_parameters():
            if name.startswith("patch_emb") or \
                name.startswith("time") or \
                    name.startswith("head") or \
                    "modulation" in name:
                continue

            param.register_hook(self.cp_grad_reduce)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
        t = self.time_emb(t_emb)
        t_mod = self.time_proj(t)
        t_mod = t_mod.unflatten(1, (6, self.dim))

        context_emb = self.text_emb(context)

        if y is not None:
            x = torch.cat([x, y], dim=1)

        if self.has_image_input:
            if clip_feature is not None:
                clip_embedding = self.img_emb(clip_feature)
                context_emb = torch.cat([clip_embedding, context_emb], dim=1)

        x, (f, h, w) = self.patchify(x)

        head_dim = self.dim // self.num_heads
        freq_f = precompute_freqs_cis(head_dim - 2 * (head_dim // 3), f).view(f, 1, 1, -1).expand(f, h, w, -1)
        freq_h = precompute_freqs_cis(head_dim // 3, h).view(1, h, 1, -1).expand(f, h, w, -1)
        freq_w = precompute_freqs_cis(head_dim // 3, w).view(1, 1, w, -1).expand(f, h, w, -1)
        freqs = torch.cat([freq_f, freq_h, freq_w], dim=-1)
        freqs = freqs.reshape(f * h * w, 1, -1).to(x.device)

        x = self.split_input(x, dim=1)
        freqs = self.split_input(freqs, dim=0)
        #x = self.blocks(x, context_emb, t_mod, freqs)

        pooled_tokens = []
        features = []

        for layer_idx, block in enumerate(self.blocks):

            x = checkpoint(block, x, context_emb, t_mod, freqs, use_reentrant=False)
            #x = block(x, context_emb, t_mod, freqs)
                    # 命中插点：用 cross-attn 抽取单 token 表达
            if self.produce_logits and layer_idx in self.logit_layers:
                # 1) CP: gather 序列维（得到全量 tokens，自动去 pad）
                features.append(x)
                x_seq_full = self.gather_output(x, dim=1)           # (B, N_full, C_local)

                # 2) TP: gather 隐藏维（得到完整 hidden 维度）
                x_full = gather_from_tensor_model_parallel_region(x_seq_full)  # (B, N_full, C_full)

                # 3) cross-attn pooling（Q=1，KV=全量视觉 tokens）
                pool_idx = self.logit_layers.index(layer_idx)
                
                pooled = checkpoint(self.logit_pools[pool_idx], x_full, use_reentrant=False)          # (B, C_full)
                pooled_tokens.append(pooled)


        logits = None
        if self.produce_logits and len(pooled_tokens) > 0:
            pooled_cat = torch.cat(pooled_tokens, dim=-1)                  # (B, C * 3)
            logits = self.logit_proj(self.logit_norm(pooled_cat)).squeeze(-1)  # (B,)

        return logits, features

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        state_dict = self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    def sharded_state_dict(self):
        return self.state_dict()

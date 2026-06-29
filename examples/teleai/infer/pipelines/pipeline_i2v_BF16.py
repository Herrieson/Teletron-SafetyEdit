from __future__ import annotations

import gzip
from typing import Optional, Tuple
import struct
import numpy as np


import types
import math

from teletron.models.teleai.models.dit import TeleaiPrompter
from teletron.models.teleai.models.dit import TeleaiTextEncoder
from teletron.models.teleai.models.dit import TeleaiVideoVAE
from teletron.models.teleai.models.dit import TeleaiVideoVAE_2_2
from teletron.models.teleai.models.dit import TeleaiImageEncoder
from teletron.models.teleai.taehv import TAEW2_1DiffusersWrapper
from teletron.models.teleai import TeleaiModel
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop
# from utils.video_entropy_models import BitEstimator

from teletron.models.flow_match import FlowMatchScheduler
from .base import BasePipeline
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image

def plot_quantization_histogram(q, save_path="quantization_histogram.png", dpi=150):
    """
    Plot quantization value histogram and save as PNG
    Args:
        q: uint8 quantized tensor
        save_path: save path
        dpi: image resolution
    """
    
    # Calculate histogram
    q = q.cpu()
    hist = torch.bincount(q.flatten(), minlength=256)
    total = hist.sum().item()
    freq = hist.float() / total * 100  # Convert to percentage
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # ========== Main Histogram ==========
    x = np.arange(256)
    bars = ax1.bar(x, hist.cpu().numpy(), width=1.0, 
                   edgecolor='black', linewidth=0.5, 
                   color='steelblue', alpha=0.8)
    
    # Highlight specific regions (0 and 255)
    highlight_ranges = [(0, 0), (255, 255)]
    for start, end in highlight_ranges:
        mask = (x >= start) & (x <= end)
        ax1.bar(x[mask], hist.cpu().numpy()[mask], width=1.0,
                color='red', alpha=0.6, edgecolor='darkred', linewidth=0.5)
    
    # Configure main plot
    ax1.set_xlabel('Quantization Value (0-255)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency Count', fontsize=12, fontweight='bold')
    ax1.set_title(f'Quantization Value Distribution (Total: {total:,}, Entropy: {-(freq/100 * torch.log2(freq/100 + 1e-10)).sum().item():.3f} bits)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Set X-axis ticks
    ax1.set_xlim(-0.5, 255.5)
    ax1.set_xticks(np.arange(0, 256, 16))
    ax1.set_xticklabels([f'{i}' for i in range(0, 256, 16)], rotation=45)
    
    # Add secondary Y-axis for percentage
    ax1_right = ax1.twinx()
    ax1_right.set_ylim(0, hist.cpu().numpy().max() / total * 100)
    ax1_right.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold', color='darkgreen')
    ax1_right.tick_params(axis='y', labelcolor='darkgreen')
    
    # ========== Bottom CDF Plot ==========
    # Create cumulative distribution curve
    cumulative = torch.cumsum(hist.float(), dim=0) / total * 100
    ax2.plot(x, cumulative.cpu().numpy(), 
             color='crimson', linewidth=2.5, 
             label='Cumulative Distribution', marker='o', markersize=2, markevery=16)
    
    # Fill area under CDF
    ax2.fill_between(x, 0, cumulative.cpu().numpy(), 
                     color='crimson', alpha=0.2)
    
    # Mark important quantiles
    quantiles = [25, 50, 75, 90, 95, 99]
    for q_val in quantiles:
        # Find where cumulative distribution reaches q_val%
        idx = torch.searchsorted(cumulative, q_val).item()
        if idx < 256:
            ax2.plot(idx, q_val, 'bo', markersize=8)
            ax2.annotate(f'{q_val}%\n(idx:{idx})', 
                        xy=(idx, q_val), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor="yellow", 
                                 alpha=0.7))
    
    ax2.set_xlabel('Quantization Value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, 255.5)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    
    # ========== Add Statistics Text Box ==========
    stats_text = f"""
Statistics:
• Total elements: {total:,}
• Unique values: {torch.sum(hist > 0).item()}
• Mean: {torch.mean(q.float()).item():.2f}
• Std Dev: {torch.std(q.float()).item():.2f}
• Min: {q.min().item()}
• Max: {q.max().item()}
• Entropy: {-(freq/100 * torch.log2(freq/100 + 1e-10)).sum().item():.3f} bits
• Theoretical compression: {(1 - (-(freq/100 * torch.log2(freq/100 + 1e-10)).sum().item()/8))*100:.1f}%
    """
    
    # Place text box in upper right corner
    ax1.text(0.98, 0.98, stats_text,
             transform=ax1.transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', 
                      facecolor='wheat', 
                      alpha=0.9,
                      edgecolor='gray'))
    
    # ========== Highlight Top Frequency Values ==========
    top_k = 5
    sorted_indices = torch.argsort(hist, descending=True)[:top_k]
    
    for i, idx in enumerate(sorted_indices):
        count = hist[idx].item()
        percentage = freq[idx].item()
        ax1.text(idx, count * 1.05, f'#{i+1}\n{percentage:.1f}%', 
                fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor="gold", 
                         alpha=0.8))
    
    plt.tight_layout()
    
    # Save image
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Histogram saved to: {save_path}")
    
    # Show image
    plt.show()
    
    # Return statistics
    return {
        'histogram': hist,
        'total': total,
        'entropy': -(freq/100 * torch.log2(freq/100 + 1e-10)).sum().item(),
        'unique_values': torch.sum(hist > 0).item()
    }

def resize_and_crop(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale = max(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image

def inference_encoder(quantizer, prob_model, arithmetic_coder, latent, device='cuda'):
    """
    Encoder side inference: compress latent to bitstream
    Args:
        quantizer: trained quantizer
        prob_model: trained probability model
        arithmetic_coder: arithmetic coder instance
        latent: input tensor [B, C, T, H, W]
        device: device to run on
    Returns:
        compressed_data: bytes to transmit
        metadata: 3 bytes (quantization params, including scale and zero_point)
    """
    quantizer.eval()
    prob_model.eval()
    # bitEstimator = BitEstimator(16).to(device)
    # bitEstimator.eval()
    # bitEstimator.update(force=True)
    with torch.no_grad():
        # Encode
        latent = latent.to(device)
        # # Quantize
        latent_dequantized, latent_uint8 = quantizer(latent)
        # plot_quantization_histogram(latent_uint8, "/gemini/space/yifq/xjy/data/quantization_histogram.png")
        # import pdb; pdb.set_trace()
        # # Predict probabilities
        # logits = prob_model(latent_dequantized)
        # probs = F.softmax(logits, dim=-1)  # [1, 16, 10, 7, 13, 256]
        
        # # Convert to numpy
        latent_np = latent_uint8.cpu().numpy()
        # probs_np = probs.float().cpu().numpy()
        
        # # Arithmetic encode
        # # compressed = arithmetic_coder.encode(latent_np, probs_np)
        
        # Get quantization parameters (3 bytes)
        metadata = quantizer.get_quantization_params()

        # bitrate estimation
        # scale, zero_point = quantizer.set_quantization_params(metadata) # 3 sigma quantization 
        # q_centered = latent_uint8.to(torch.int8)
        # B, C, T, H, W = q_centered.shape
        # q_for_entropy = q_centered.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
        # q_for_entropy = q_for_entropy.reshape(B*T, C, H, W).contiguous()  # B*T,C,H,W

        # strings_all = []

        # for b in range(B*T):
            # symbols = q_for_entropy[b, :, :, :].unsqueeze(0)  # (1, C, H, W)
            # strings = bitEstimator.compress(symbols)
            # strings_all.append(strings)
        # import pdb; pdb.set_trace()
        
        return latent_np, metadata

def inference_decoder(quantizer, prob_model, arithmetic_coder, compressed_data, metadata, shape, device='cuda'):
    """
    Decoder side inference: decompress bitstream and decode
    Args:
        quantizer: trained quantizer
        prob_model: trained probability model
        arithmetic_coder: arithmetic coder instance
        compressed_data: compressed bytes
        metadata: 3 bytes quantization params
        shape: tuple (B, C, T, H, W) of the output latent
        device: device to run on
    Returns:
        latent_dequantized: reconstructed latent tensor [B, C, T, H, W]
    """
    prob_model.eval()
    
    with torch.no_grad():
        # Parse metadata
        scale, zero_point = quantizer.set_quantization_params(metadata)
        quantizer.scale = scale
        quantizer.zero_point = zero_point
        
        # We need to reconstruct probabilities for decoding
        # This requires autoregressive decoding (decode symbol by symbol)
        B, C, T, H, W = shape
        num_symbols = B * C * T * H * W
        
        # Create dummy probs for decoding (in practice, use autoregressive model)
        # For simplicity, use uniform distribution first pass
        # In production, decode autoregressively with prob_model
        probs_uniform = np.ones((num_symbols, 256)) / 256
        
        # Arithmetic decode
        # latent_np = arithmetic_coder.decode(compressed_data, num_symbols, probs_uniform)
        # latent_np = latent_np.reshape(shape)
        
        # Convert to tensor
        # latent_uint8 = torch.from_numpy(latent_np).to(device)
        # latent_uint8 = torch.from_numpy(latent_np).to(device)

        
        # Dequantize
        latent_dequantized = quantizer.dequantize(compressed_data).to(device)

        return latent_dequantized




import gzip
import numpy as np
from typing import Optional, Tuple


# 2-bit tags (packed 4 per byte):
# 00: prev same-position token (send nothing)      -> out = prev[i]
# 10: codebook reference (send uint16 index)       -> out = codebook[idx]
# 11: raw token (send C*2 bytes BF16)              -> out = raw
TAG00 = np.uint8(0)
TAG10 = np.uint8(2)
TAG11 = np.uint8(3)


def _try_get_bfloat16_dtype():
    """
    Optional support for ml_dtypes.bfloat16, if installed.
    """
    try:
        import ml_dtypes  # type: ignore
        return ml_dtypes.bfloat16
    except Exception:
        return None


_BFLOAT16_DTYPE = _try_get_bfloat16_dtype()


def f32_to_bf16_u16(x_f32: np.ndarray) -> np.ndarray:
    """
    float32 -> BF16 bitpattern (uint16), truncation (round-to-zero in mantissa bits).
    """
    x = np.asarray(x_f32, dtype=np.float32)
    u32 = x.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def bf16_u16_to_f32(bf16_u16: np.ndarray) -> np.ndarray:
    """
    BF16 bitpattern (uint16) -> float32 (exact expansion).
    """
    # print("here1018")
    b = np.asarray(bf16_u16, dtype=np.uint16)
    u32 = (b.astype(np.uint32) << 16)
    return u32.view(np.float32)


def latent_to_bf16_u16(latent_bf16: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - np.uint16 array interpreted as BF16 bitpattern
      - ml_dtypes.bfloat16 if available
      - float16/float32/float64 (will be quantized/truncated to BF16)

    Returns:
      - np.uint16 BF16 bitpattern
    """
    import pdb; pdb.set_trace()
    x = np.asarray(latent_bf16)  
    if x.dtype == np.uint16:
        return np.ascontiguousarray(x, dtype=np.uint16)

    # If ml_dtypes bfloat16 exists and input is bfloat16, convert through float32
    if _BFLOAT16_DTYPE is not None and x.dtype == _BFLOAT16_DTYPE:
        return f32_to_bf16_u16(x.astype(np.float32, copy=False))

    # Otherwise treat as float and truncate to BF16
    if np.issubdtype(x.dtype, np.floating):
        return f32_to_bf16_u16(x.astype(np.float32, copy=False))

    raise TypeError(f"latent dtype must be BF16 (uint16 bits or bfloat16) or float; got {x.dtype}")


def bf16_u16_to_latent_dtype(u16: np.ndarray, like: np.ndarray) -> np.ndarray:
    """
    Convert BF16 bits (uint16) back to the "bf16 type" matching `like`:
      - if like.dtype is np.uint16 => return uint16 bits
      - if like.dtype is ml_dtypes.bfloat16 => return bfloat16 (if available)
      - else => return uint16 bits (since you要求输入输出bf16；这里最稳定就是uint16 bits)
    """
    u16 = np.ascontiguousarray(u16, dtype=np.uint16)
    if like.dtype == np.uint16:
        return u16
    if _BFLOAT16_DTYPE is not None and like.dtype == _BFLOAT16_DTYPE:
        # Make float32 then cast to bfloat16 (should preserve bf16 bits)
        return bf16_u16_to_f32(u16).astype(_BFLOAT16_DTYPE)
    # Fallback: return bits
    return u16


class CodebookCodecNP_BF16_Gzip:
    """
    统一BF16 Codebook版本
    
    编码端和解码端都只维护BF16 codebook
    关键：编码端的BF16 token必须是"模拟反量化"后的，确保两端codebook一致
    """

    def __init__(
        self,
        metric: str = "cosine",
        threshold: float = 0.02,
        max_codebook_size: int = 65535,
        eps: float = 1e-12,
        compute_dtype=np.float32,
    ):
        self.metric = metric
        self.threshold = float(threshold)
        self.max_codebook_size = int(max_codebook_size)
        self.eps = float(eps)
        self.compute_dtype = compute_dtype

        # ========== 编码端 ==========
        self.prev_pos_bf16: Optional[np.ndarray] = None  # (m, C) uint16 BF16
        self.codebook_bf16: Optional[np.ndarray] = None  # (L, C) uint16 BF16
        self.use_count: Optional[np.ndarray] = None
        self.last_used: Optional[np.ndarray] = None
        
        # ========== 解码端 ==========
        self.codebook_bf16_decode: Optional[np.ndarray] = None  # (L, C) uint16 BF16
        self.prev_pos_bf16_decode: Optional[np.ndarray] = None
        self.use_count_decode: Optional[np.ndarray] = None
        self.last_used_decode: Optional[np.ndarray] = None
        
        self._step: np.uint32 = np.uint32(0)



    def encode(
        self,
        latent_bf16: np.ndarray,  # (B,C,T,H,W) 原始BF16，用于相似度计算
        latent_u8: np.ndarray,    # (B,C,T,H,W) 量化后uint8，用于传输
        metadata: bytes,          # 3字节量化参数
        gzip_level: int = 9,
    ) -> bytes:
        """
        编码流程：
        1. 用原始latent_bf16计算相似度
        2. TAG11时传输uint8，但codebook里存的是"模拟反量化"后的BF16
        """
        tokens_bf16_original = self._latent_to_tokens(latent_bf16)  # 原始BF16
        tokens_u8 = self._latent_to_tokens(latent_u8)
        m, C = tokens_bf16_original.shape
        
        # 解析当前metadata（用于模拟反量化）
        scale, zero_point = self._parse_metadata(metadata)
        
        has_prev = (self.prev_pos_bf16 is not None and self.prev_pos_bf16.shape == (m, C))
        has_codebook = (self.codebook_bf16 is not None and self.codebook_bf16.shape[0] > 0)
        
        tags = np.full((m,), TAG11, dtype=np.uint8)
        payload = bytearray()
        out_bf16 = np.empty((m, C), dtype=np.uint16)
        
        # === 1. 用原始BF16计算相似度 ===
        if has_prev:
            prev_val = self._pair_metric(tokens_bf16_original, self.prev_pos_bf16, self.metric)
            prev_ok = self._within_threshold(prev_val, self.metric, self.threshold)
        else:
            prev_ok = np.zeros((m,), dtype=bool)
        
        if has_codebook:
            nn_idx, cb_val = self._nearest(tokens_bf16_original, self.codebook_bf16, self.metric)
            cb_ok = self._within_threshold(cb_val, self.metric, self.threshold)
        else:
            nn_idx = None
            cb_ok = np.zeros((m,), dtype=bool)
        
        # === 2. 决定tag ===
        if has_prev and has_codebook:
            prev_better = self._better_or_equal(prev_val, cb_val, self.metric)
            use00 = prev_ok & (~cb_ok | prev_better)
            use10 = cb_ok & (~prev_ok | ~prev_better)
        elif has_prev:
            use00 = prev_ok
            use10 = np.zeros((m,), dtype=bool)
        else:
            use00 = np.zeros((m,), dtype=bool)
            use10 = cb_ok if has_codebook else np.zeros((m,), dtype=bool)
        
        tags[use00] = TAG00
        tags[use10] = TAG10
        
        # === 3. 生成payload + 重建BF16（关键：TAG11模拟反量化）===
        tag10_indices = []
        tag11_bf16_list = []
        
        for i in range(m):
            t = int(tags[i])
            
            if t == TAG00:
                out_bf16[i] = self.prev_pos_bf16[i]
                
            elif t == TAG10:
                idx = int(nn_idx[i])
                payload += idx.to_bytes(2, "little")
                out_bf16[i] = self.codebook_bf16[idx]
                tag10_indices.append(idx)
                
            else:  # TAG11 - 关键修改
                # 传输uint8
                payload += tokens_u8[i].tobytes()
                # 但codebook存的是"模拟反量化"后的BF16
                out_bf16[i] = self._dequantize_single_token(tokens_u8[i], scale, zero_point)
                tag11_bf16_list.append(out_bf16[i])
        
        # === 4. 更新状态 ===
        self.prev_pos_bf16 = np.ascontiguousarray(out_bf16, dtype=np.uint16)
        self._update_codebook_encode(tag10_indices, tag11_bf16_list)
        
        # === 5. 打包 ===
        tags_packed = self._pack_tags_2bit(tags)
        raw_stream = metadata + tags_packed.tobytes() + bytes(payload)
        bitstream = gzip.compress(raw_stream, compresslevel=gzip_level)
        
        c00, c10, c11 = int(np.sum(tags == TAG00)), int(np.sum(tags == TAG10)), int(np.sum(tags == TAG11))
        print(f"[encode clip={self._step}] 00={c00}({c00/m:.3f}) 10={c10}({c10/m:.3f}) 11={c11}({c11/m:.3f}) | gzip={len(bitstream)}B")
        
        self._step = np.uint32(int(self._step) + 1)
        return bitstream

    



    def decode(
        self,
        bitstream: bytes,
        shape: Tuple[int, int, int, int, int],
    ) -> np.ndarray:
        """
        解码流程：
        1. 解析metadata
        2. TAG11时反量化uint8 → BF16
        3. 更新codebook（与编码端一致）
        """
        B, C, T, H, W = shape
        assert B == 1
        m = T * H * W
        
        # === 1. 解压缩 ===
        raw = gzip.decompress(bitstream)
        
        # === 2. 解析metadata ===
        metadata = raw[:3]
        scale, zero_point = self._parse_metadata(metadata)
        
        # === 3. 解析tags ===
        offset = 3
        tag_bytes = (m + 3) // 4
        tags_packed = np.frombuffer(raw, dtype=np.uint8, count=tag_bytes, offset=offset)
        tags = self._unpack_tags_2bit(tags_packed, m)
        offset += tag_bytes
        
        # === 4. 重建BF16 tokens ===
        buf = memoryview(raw)
        out_bf16 = np.empty((m, C), dtype=np.uint16)
        
        tag10_indices = []
        tag11_bf16_list = []
        
        has_prev = (self.prev_pos_bf16_decode is not None and 
                    self.prev_pos_bf16_decode.shape == (m, C))
        has_codebook = (self.codebook_bf16_decode is not None and 
                       self.codebook_bf16_decode.shape[0] > 0)
        
        for i in range(m):
            t = int(tags[i])
            
            if t == TAG00:
                if not has_prev:
                    raise ValueError("TAG00 but no prev")
                out_bf16[i] = self.prev_pos_bf16_decode[i]
                
            elif t == TAG10:
                if not has_codebook:
                    raise ValueError("TAG10 but no codebook")
                if offset + 2 > len(buf):
                    raise ValueError("Missing index")
                idx = int.from_bytes(buf[offset:offset+2], "little")
                offset += 2
                if idx >= self.codebook_bf16_decode.shape[0]:
                    raise ValueError(f"Index out of range: {idx}")
                out_bf16[i] = self.codebook_bf16_decode[idx]
                tag10_indices.append(idx)
                
            else:  # TAG11
                if offset + C > len(buf):
                    raise ValueError("Missing token")
                token_u8 = np.frombuffer(buf[offset:offset+C], dtype=np.uint8)
                offset += C
                # 反量化uint8 → BF16
                out_bf16[i] = self._dequantize_single_token(token_u8, scale, zero_point)
                tag11_bf16_list.append(out_bf16[i])
        
        if offset != len(buf):
            raise ValueError(f"Trailing bytes: {offset} != {len(buf)}")
        
        # === 5. 更新状态 ===
        self.prev_pos_bf16_decode = np.ascontiguousarray(out_bf16, dtype=np.uint16)
        self._update_codebook_decode(tag10_indices, tag11_bf16_list)
        
        # === 6. 返回BF16 latent ===
        latent_bf16 = self._tokens_to_latent(out_bf16, B, C, T, H, W)
        
        print(f"[decode clip={self._step}] codebook_size={len(self.codebook_bf16_decode) if self.codebook_bf16_decode is not None else 0}")
        
        self._step = np.uint32(int(self._step) + 1)
        return latent_bf16
    

    # ========== 反量化单个token ========== 
    def _dequantize_single_token(
        self, 
        token_u8: np.ndarray,  # (C,) uint8
        scale: float, 
        zero_point: float
    ) -> np.ndarray:
        """uint8 token → BF16 token (uint16 bits)"""
        token_f32 = (token_u8.astype(np.float32) - zero_point) * scale
        return f32_to_bf16_u16(token_f32)  # (C,) uint16 BF16 bits



    # ========== Codebook更新 ==========
    def _update_codebook_encode(self, tag10_indices: list, tag11_bf16: list):
        """编码端更新：TAG11追加（已经是反量化后的BF16）"""
        if self.codebook_bf16 is None:
            if len(tag11_bf16) > 0:
                C = tag11_bf16[0].shape[0]
                self.codebook_bf16 = np.empty((0, C), dtype=np.uint16)
                self.use_count = np.empty((0,), dtype=np.uint32)
                self.last_used = np.empty((0,), dtype=np.uint32)
            else:
                return
        
        # TAG10统计
        if len(tag10_indices) > 0 and len(self.use_count) > 0:
            idxs = np.asarray(tag10_indices, dtype=np.int64)
            np.add.at(self.use_count, idxs, 1)
            self.last_used[idxs] = self._step
        
        # TAG11追加
        if len(tag11_bf16) > 0:
            new_bf16 = np.stack(tag11_bf16, axis=0).astype(np.uint16)
            n_new = new_bf16.shape[0]
            
            self.codebook_bf16 = np.concatenate([self.codebook_bf16, new_bf16])
            self.use_count = np.concatenate([self.use_count, np.zeros((n_new,), dtype=np.uint32)])
            self.last_used = np.concatenate([self.last_used, np.full((n_new,), self._step, dtype=np.uint32)])
        
        self._evict_if_needed_encode()




    def _update_codebook_decode(self, tag10_indices: list, tag11_bf16: list):
        """解码端更新：TAG11追加（反量化后的BF16）"""
        if self.codebook_bf16_decode is None:
            if len(tag11_bf16) > 0:
                C = tag11_bf16[0].shape[0]
                self.codebook_bf16_decode = np.empty((0, C), dtype=np.uint16)
                self.use_count_decode = np.empty((0,), dtype=np.uint32)
                self.last_used_decode = np.empty((0,), dtype=np.uint32)
            else:
                return
        
        # TAG10统计
        if len(tag10_indices) > 0 and len(self.use_count_decode) > 0:
            idxs = np.asarray(tag10_indices, dtype=np.int64)
            np.add.at(self.use_count_decode, idxs, 1)
            self.last_used_decode[idxs] = self._step
        
        # TAG11追加
        if len(tag11_bf16) > 0:
            new_bf16 = np.stack(tag11_bf16, axis=0).astype(np.uint16)
            n_new = new_bf16.shape[0]
            
            self.codebook_bf16_decode = np.concatenate([self.codebook_bf16_decode, new_bf16])
            self.use_count_decode = np.concatenate([self.use_count_decode, np.zeros((n_new,), dtype=np.uint32)])
            self.last_used_decode = np.concatenate([self.last_used_decode, np.full((n_new,), self._step, dtype=np.uint32)])
        
        self._evict_if_needed_decode()





    def _evict_if_needed_encode(self):
        """编码端LFU淘汰"""
        if self.codebook_bf16 is None:
            return
        
        L = self.codebook_bf16.shape[0]
        if L <= self.max_codebook_size:
            return
        
        overflow = L - self.max_codebook_size
        
        # LFU策略：use_count升序，tie-break用last_used升序
        order = np.lexsort((self.last_used, self.use_count))
        drop = order[:overflow]
        
        keep = np.ones((L,), dtype=bool)
        keep[drop] = False
        
        self.codebook_bf16 = self.codebook_bf16[keep]
        self.use_count = self.use_count[keep]
        self.last_used = self.last_used[keep]


    def _evict_if_needed_decode(self):
        """解码端LFU淘汰"""
        if self.codebook_bf16_decode is None:  # ← 修正：用正确的变量名
            return
        
        L = self.codebook_bf16_decode.shape[0]  # ← 修正
        if L <= self.max_codebook_size:
            return
        
        overflow = L - self.max_codebook_size
        order = np.lexsort((self.last_used_decode, self.use_count_decode))
        drop = order[:overflow]
        
        keep = np.ones((L,), dtype=bool)
        keep[drop] = False
        
        self.codebook_bf16_decode = self.codebook_bf16_decode[keep]  # ← 修正
        self.use_count_decode = self.use_count_decode[keep]
        self.last_used_decode = self.last_used_decode[keep]





    # ========================================================================
    # 量化/反量化
    # ========================================================================
    def _parse_metadata(self, metadata: bytes) -> Tuple[float, float]:
        """
        解析3字节metadata -> (scale, zero_point)
        格式: scale(2字节 float16) + zero_point(1字节 uint8)
        """
        if len(metadata) != 3:
            raise ValueError(f"Metadata must be 3 bytes, got {len(metadata)}")
        
        # 使用struct解析 (与quantizer.py中的格式一致)
        scale = struct.unpack('e', metadata[:2])[0]  # 'e' = float16
        zero_point = struct.unpack('B', metadata[2:3])[0]  # 'B' = uint8
        
        return float(scale), float(zero_point)


    # ========================================================================
    # 相似度计算（BF16）
    # ========================================================================
    def _pair_metric(
        self,
        A_u16: np.ndarray,  # (m, C) uint16 BF16
        B_u16: np.ndarray,  # (m, C) uint16 BF16
        metric: str,
    ) -> np.ndarray:
        """计算成对距离/相似度"""
        A = bf16_u16_to_f32(A_u16).astype(self.compute_dtype)
        B = bf16_u16_to_f32(B_u16).astype(self.compute_dtype)
        
        if metric == "l2":
            d2 = np.sum((A - B) ** 2, axis=1)
            return np.sqrt(np.maximum(d2, 0.0))
        elif metric == "l2_squared":
            return np.sum((A - B) ** 2, axis=1)
        elif metric == "l1":
            return np.sum(np.abs(A - B), axis=1)
        elif metric == "cosine":
            An = self._row_normalize(A)
            Bn = self._row_normalize(B)
            cos = np.sum(An * Bn, axis=1)
            return 1.0 - cos
        elif metric == "dot":
            return np.sum(A * B, axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _nearest(
        self,
        tokens_u16: np.ndarray,     # (m, C) uint16 BF16
        codebook_u16: np.ndarray,   # (L, C) uint16 BF16
        metric: str,
    ):
        """最近邻搜索"""
        X = bf16_u16_to_f32(tokens_u16).astype(self.compute_dtype)
        Y = bf16_u16_to_f32(codebook_u16).astype(self.compute_dtype)
        
        if metric == "l2" or metric == "l2_squared":
            d2 = self._pairwise_l2_squared(X, Y)
            nn_idx = np.argmin(d2, axis=1)
            if metric == "l2":
                best_val = np.sqrt(d2[np.arange(len(nn_idx)), nn_idx])
            else:
                best_val = d2[np.arange(len(nn_idx)), nn_idx]
        elif metric == "l1":
            d1 = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
            nn_idx = np.argmin(d1, axis=1)
            best_val = d1[np.arange(len(nn_idx)), nn_idx]
        elif metric == "cosine":
            Xn = self._row_normalize(X)
            Yn = self._row_normalize(Y)
            cos = Xn @ Yn.T
            dist = 1.0 - cos
            nn_idx = np.argmin(dist, axis=1)
            best_val = dist[np.arange(len(nn_idx)), nn_idx]
        elif metric == "dot":
            score = X @ Y.T
            nn_idx = np.argmax(score, axis=1)
            best_val = score[np.arange(len(nn_idx)), nn_idx]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return nn_idx.astype(np.int64), best_val

    @staticmethod
    def _pairwise_l2_squared(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x2 = np.sum(X * X, axis=1, keepdims=True)
        y2 = np.sum(Y * Y, axis=1, keepdims=True).T
        xy = X @ Y.T
        d2 = x2 + y2 - 2.0 * xy
        return np.maximum(d2, 0.0)

    def _row_normalize(self, X: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.maximum(denom, self.eps)
        return X / denom

    @staticmethod
    def _within_threshold(val: np.ndarray, metric: str, threshold: float) -> np.ndarray:
        if metric == "dot":
            return val >= threshold
        return val <= threshold

    @staticmethod
    def _better_or_equal(val_a: np.ndarray, val_b: np.ndarray, metric: str) -> np.ndarray:
        if metric == "dot":
            return val_a >= val_b
        return val_a <= val_b

    # ========================================================================
    # Tag打包/解包
    # ========================================================================
    @staticmethod
    def _pack_tags_2bit(tags: np.ndarray) -> np.ndarray:
        """2-bit tags打包到uint8"""
        tags = np.asarray(tags, dtype=np.uint8)
        m = len(tags)
        nbytes = (m + 3) // 4
        pad = nbytes * 4 - m
        if pad:
            tags = np.concatenate([tags, np.zeros(pad, dtype=np.uint8)])
        x = tags.reshape(nbytes, 4) & 0x3
        packed = (x[:, 0] | (x[:, 1] << 2) | (x[:, 2] << 4) | (x[:, 3] << 6)).astype(np.uint8)
        return packed

    @staticmethod
    def _unpack_tags_2bit(packed: np.ndarray, m: int) -> np.ndarray:
        """从uint8解包2-bit tags"""
        packed = np.asarray(packed, dtype=np.uint8)
        tags = np.stack([
            (packed >> 0) & 0x3,
            (packed >> 2) & 0x3,
            (packed >> 4) & 0x3,
            (packed >> 6) & 0x3
        ], axis=1).reshape(-1)
        return tags[:m]

    # ========================================================================
    # Reshape helpers
    # ========================================================================
    @staticmethod
    def _latent_to_tokens(latent: np.ndarray) -> np.ndarray:
        """latent -> tokens: (B,C,T,H,W) -> (m,C) where m=T*H*W"""
        B, C, T, H, W = latent.shape
        x = np.transpose(latent, (0, 2, 3, 4, 1))  # (B,T,H,W,C)
        return x.reshape(B * T * H * W, C)

    @staticmethod
    def _tokens_to_latent(tokens: np.ndarray, B: int, C: int, T: int, H: int, W: int) -> np.ndarray:
        """tokens -> latent: (m,C) -> (B,C,T,H,W)"""
        x = tokens.reshape(B, T, H, W, C)
        return np.transpose(x, (0, 4, 1, 2, 3))
    


    




class WanVideoI2VNewCodebookPipeline_BF16(BasePipeline):

    def __init__(self, pipeline_config):
        torch_dtype = pipeline_config.get("torch_dtype", torch.bfloat16)
        device = pipeline_config.get("device", "cuda")
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        
        self.codebook1 = CodebookCodecNP_BF16_Gzip(metric="l2", threshold=10.0)  # encode端
        self.codebook2 = CodebookCodecNP_BF16_Gzip(metric="l2", threshold=10.0)   # decode端

        self.token_ratio = []
        # model config
        self.model_config = pipeline_config.get("model_config", None)

        # dit config
        dit_config = pipeline_config.get("model_config", None).get("dit", None)
        print(f"加载 DiT 模型... {dit_config.get('path')}")
        # with torch.device('meta'):
        self.dit = TeleaiModel(**dit_config.get("config"))
        if "ema" in dit_config.get("path"):
            state_static = torch.load(dit_config.get("path"), map_location='cpu', weights_only=False)[0]
        else:
            state_static = torch.load(dit_config.get("path"), map_location='cpu', weights_only=False)["model"]

        self.dit.load_state_dict(state_static, strict=True, assign=True)
        self.dit.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)

        # assert 0
        # encoder config
        self.encoder_model_config = pipeline_config.get("model_config", None).get("encoder", None)
        
        # vae config
        self.vae_path = self.encoder_model_config.get("vae", None).get("path", None)
        self.vae_type = self.encoder_model_config.get("vae", None).get("type", "TeleaiVideoVAE_2_1")
        self.tiler_kwargs = self.encoder_model_config.get("vae", None).get("tiler_kwargs", {})
        if self.tiler_kwargs is None:
            self.tiler_kwargs = dict(
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
        print(f"加载 VAE 模型... {self.vae_path}")
        if self.vae_type == "TeleaiVideoVAE_2_1":
            self.vae = TeleaiVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.compression = (4,8,8)
        elif self.vae_type == "TeleaiVideoTAE_2_1":
            self.compression = (4,8,8)
            self.vae = TAEW2_1DiffusersWrapper(self.vae_path, device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        else:
            self.vae = TeleaiVideoVAE_2_2().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.compression = (4,16,16)
        if self.vae_type != "TeleaiVideoTAE_2_1":
            self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu', weights_only=False), strict=True)
        
        # text encoder config
        text_encoder_path = self.encoder_model_config.get("text_encoder", None).get("path", None)
        tokenizer_path = self.encoder_model_config.get("text_encoder", None).get("tokenizer_path", None)
        print(f"加载 Text Encoder 模型... {text_encoder_path}")
        with torch.device('meta'):
            self.text_encoder = TeleaiTextEncoder()
        self.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location='cpu', weights_only=False), strict=True, assign=True)
        self.text_encoder.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.prompter = TeleaiPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(tokenizer_path)

        if self.encoder_model_config.get("image_encoder", None) is not None and self.dit.has_image_input:
            image_encoder_path = self.encoder_model_config.get("image_encoder", None).get("path", None)
            print(f"加载 Image Encoder 模型... {image_encoder_path}")
            self.image_encoder = TeleaiImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(image_encoder_path, map_location='cpu', weights_only=False), strict=False)
        else:
            self.image_encoder = None
        
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder'] if self.image_encoder is not None else ['text_encoder', 'dit', 'vae']

        self.height_division_factor = 16
        self.width_division_factor = 16

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}

    def encode_ref_images(self, ref_images, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        ref_images = [(int(frame_id), self.preprocess_image(resize_and_crop(image, (width, height))).to(self.device)) for frame_id, image in ref_images]
        ref_video = torch.zeros(1, num_frames, 3, height, width)
        for frame_id, ref_image in ref_images:
            ref_video[:, frame_id] = ref_image.unsqueeze(0)
        ref_video = ref_video.to(dtype=self.torch_dtype, device=self.device).permute(0, 2, 1, 3, 4)
        ref_latents = self.vae.encode(
            ref_video, device=self.device, 
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        ).to(dtype=self.torch_dtype, device=self.device)

        msk = torch.zeros(1, num_frames, height//8, width//8, device=self.device)
        for frame_id, _ in ref_images:
            msk[:, frame_id] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, ref_latents], dim=1)

        if self.dit.has_image_input:
            assert ref_images[0][0] == 0 # first frame
            clip_context = self.image_encoder.encode_image(
                [ref_images[0][1].to(dtype=self.torch_dtype, device=self.device)]
            ).to(dtype=self.torch_dtype, device=self.device)

        if self.dit.has_image_input:
            return {"clip_feature": clip_context, "y": y}
        else:
            return {"y": y}


    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # import ipdb;ipdb.set_trace()
        image = self.preprocess_image(image.resize((width, height))).to(device="cuda",dtype=torch.bfloat16) # 训练
        # print(image.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(device="cuda",dtype=torch.bfloat16)# 训练
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        if self.vae_type == "TeleaiVideoVAE_2_1":
            y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        elif self.vae_type == "TeleaiVideoTAE_2_1":
            vae_input = vae_input.unsqueeze(0)
            y = self.vae.encode(vae_input.to(dtype=self.torch_dtype, device=self.device), device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            y = y.squeeze(0)
        else:
            raise NotImplementedError("Only TeleaiVideoVAE_2_1 and TeleaiVideoTAE_2_1 are supported now.")
        
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        if frames.max().item() > 10: # TAE
            frames = frames.cpu().numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
        else:
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), **kwargs):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    def encode_condition(self, condition, num_frames, height, width, has_mask=False, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        import time
        t1 = time.time()
        latents = self.vae.encode(condition, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) # [1, 16, T, H, W]
        t2 = time.time()
        print(f"VAE encode time: {t2 - t1} seconds.")
        msk = torch.zeros(1, num_frames, height//8, width//8, device=self.device)
        index = (condition > -0.999).any(dim=(1, 3, 4))[0]
        msk[:, index] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
        if has_mask:
            y = torch.concat([msk, latents], dim=1) # [1, 16+4, T, H, W]
        else:
            y = latents
        return y

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        ref_images=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        progress_bar_cmd=tqdm,
        **kwargs
    ):
        # Parameter check
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        target_area = width * height
        if ref_images is not None:
            original_width, original_height = ref_images[0][1].size
            ratio = original_height / original_width
            new_width, new_height = math.sqrt(target_area / ratio), math.sqrt(target_area * ratio)
            width = int((new_width // 16) * 16)
            height = int((new_height // 16) * 16)
        else:
            width = (width // 16) * 16
            height = (height // 16) * 16

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if ref_images is not None:
            self.load_models_to_device(["vae"]) if self.image_encoder is None else self.load_models_to_device(["vae", "image_encoder"])
            image_emb = self.encode_ref_images(ref_images, num_frames, height, width) # without tilling

        # Denoise
        self.load_models_to_device(self.dit)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc='Denoising ...')):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(
                x=latents, timestep=timestep, **prompt_emb_posi, **image_emb,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    x=latents, timestep=timestep, **prompt_emb_nega, **image_emb,
                    )
                noise_pred = cfg_scale * noise_pred_posi + (1 - cfg_scale) * noise_pred_nega
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, denoising_strength=1.)

        # Decode
        self.load_models_to_device(['vae'])
        
        frames = self.decode_video(latents, **self.tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
    
    @torch.no_grad
    def recon(
        self,
        input_video, 
        prompt,
        negative_prompt="",
        input_image=None,
        last_image=None, 
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=1.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        num_inference_steps=50,
        sigma_shift=5.0,
        progress_bar_cmd=tqdm,
        cn_images=None,
        add_cn_noise=False,
        return_compressed=False,
        has_mask=False,
        **kwargs
    ):
        time_record_dict = {}

        # Parameter check
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])

            import time
            encode_start = time.time()
            latents = self.encode_video(input_video.to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            encode_time = time.time()-encode_start
            print(f"VAE encode time: {encode_time} seconds.")
            time_record_dict["vae_encode"] = encode_time

            downsample_latents1, downsample_latents2, downsample_latents3 = self.dit.compressor_down(latents) # 16x64x64,  16x32x32, 8x16x16

            # quantization
            if not add_cn_noise: # 1.3B
                # compression
                compressed1, metadata1 = inference_encoder(
                    self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                    self.dit.quantizer.arithmetic_coder, downsample_latents1, device=self.device
                )
                compressed2, metadata2 = inference_encoder(
                    self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                    self.dit.quantizer.arithmetic_coder, downsample_latents2, device=self.device
                )
                compressed3, metadata3 = inference_encoder(
                    self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                    self.dit.quantizer.arithmetic_coder, downsample_latents3, device=self.device
                )


                # 新代码
                latents3_bf16 = f32_to_bf16_u16(downsample_latents3.cpu().float().numpy())
                bitstream = self.codebook1.encode(
                    latent_bf16=latents3_bf16,
                    latent_u8=compressed3,
                    metadata=metadata3,
                    gzip_level=9
                )

                byte_num = len(bitstream)
                print("byte_num =", byte_num)

                decoded_np = self.codebook2.decode(bitstream, shape=downsample_latents3.shape)
                downsample_latents3 = torch.from_numpy(bf16_u16_to_f32(decoded_np)).to(
                    device=self.device, 
                    dtype=self.torch_dtype
                )

                # 验证codebook一致性
                if self.codebook1.codebook_bf16 is not None and self.codebook2.codebook_bf16_decode is not None:
                    if not np.array_equal(self.codebook1.codebook_bf16, self.codebook2.codebook_bf16_decode):
                        print("[WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!] Codebook mismatch!")
                        print(f"Encode codebook: {self.codebook1.codebook_bf16.shape}")
                        print(f"Decode codebook: {self.codebook2.codebook_bf16_decode.shape}")
                        assert 0
                    else:
                        print(f"[OK] Codebooks match! Size: {self.codebook1.codebook_bf16.shape[0]}")



                # # decompression
                # downsample_latents1 = inference_decoder(
                #     self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                #     self.dit.quantizer.arithmetic_coder, compressed1, metadata1, 
                #     downsample_latents1.shape, device=self.device
                # )
                # downsample_latents2 = inference_decoder(
                #     self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                #     self.dit.quantizer.arithmetic_coder, compressed2, metadata2, 
                #     downsample_latents2.shape, device=self.device
                # )
                # downsample_latents3 = inference_decoder(
                #     self.dit.quantizer.quantizer, self.dit.quantizer.prob_model, 
                #     self.dit.quantizer.arithmetic_coder, compressed3, metadata3, 
                #     downsample_latents3.shape, device=self.device
                # )   

            upsample_latents1 = self.dit.compressor_up(downsample_latents3) # v3

            #compress_time = time.time()-compress_down_start
            #print(f"compression time: {compress_time} seconds.")
            #time_record_dict["compression_down_up_v3"] = compress_time-codebook_enc_time-codebook_dec_time

            mu, logvar = upsample_latents1.chunk(2, dim=1)
            noise = mu * 0.5 + noise * 0.5
            # noise = mu
        latents = noise
        # torch.save(latents.cpu(), "/gemini/space/yifq/xjy/data/debug_latents.pt")
        if cn_images is not None:
            bs, c, num_frames, h, w = cn_images.shape
            if add_cn_noise: # 14B
                cn_images = self.encode_condition(cn_images.to(dtype=self.torch_dtype, device=self.device), num_frames=num_frames, height=h, width=w, has_mask=False,**tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
                cn_noise = torch.rand_like(cn_images)
                cn_images = 0.5 * cn_images + 0.5 * cn_noise
                cn_images = cn_images.to(dtype=self.torch_dtype, device=self.device)
            else: # 1.3B
                cn_images = self.encode_condition(cn_images.to(dtype=self.torch_dtype, device=self.device), num_frames=num_frames, height=h, width=w, has_mask=True,**tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
                cn_images = cn_images.to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, last_image, num_frames, height, width)
        else:
            image_emb = {}

        # Denoise
        self.load_models_to_device(self.dit)
        # import pdb; pdb.set_trace()
        t1 = time.time()
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc='Denoising ...')):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(
                x=latents, timestep=timestep, **prompt_emb_posi, **image_emb, cn_images=cn_images,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    x=latents, timestep=timestep, **prompt_emb_nega, **image_emb, cn_images=cn_images,
                    )
                noise_pred = cfg_scale * noise_pred_posi + (1 - cfg_scale) * noise_pred_nega
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, denoising_strength=1.)
        t2 = time.time()
        print(f"Denoising time: {t2 - t1} seconds.")
        time_record_dict["denoising"] = t2 - t1

        # Decode
        self.load_models_to_device(['vae'])
        

        t1 = time.time()
        frames = self.decode_video(latents, **self.tiler_kwargs)
        t2 = time.time()
        print(f"VAE decode time: {t2 - t1} seconds.")
        time_record_dict["vae_decode"] = t2 - t1
        
        self.load_models_to_device([])
        
        frames = self.tensor2video(frames[0])


        if not return_compressed:
            return frames
        else:
            # return frames, packet_bytes_dict
            return frames, byte_num
            # return frames, num_tokens, time_record_dict




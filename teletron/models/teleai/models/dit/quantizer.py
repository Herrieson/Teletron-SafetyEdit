import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import struct


class ArithmeticCoder:
    """Context-based Adaptive Binary Arithmetic Coding"""
    
    def __init__(self):
        self.precision = 32
        self.one = 1 << self.precision
        self.half = self.one >> 1
        self.quarter = self.half >> 1
        
    def encode(self, symbols: np.ndarray, probs: np.ndarray) -> bytes:
        """
        Encode symbols using arithmetic coding
        Args:
            symbols: uint8 array of shape [N]
            probs: probability distribution [N, 256]
        Returns:
            compressed bytes
        """
        low = 0
        high = self.one
        pending_bits = 0
        output_bits = []
        
        symbols_flat = symbols.flatten()
        probs_flat = probs.reshape(-1, 256)
        
        for i, symbol in enumerate(symbols_flat):
            symbol = int(symbol)
            prob = probs_flat[i]
            
            # Create cumulative distribution
            cumulative = np.cumsum(prob)
            cumulative = np.concatenate([[0], cumulative])
            cumulative = (cumulative * (high - low)).astype(np.uint64) + low
            
            low = cumulative[symbol]
            high = cumulative[symbol + 1]
            
            # Renormalization
            while True:
                if high <= self.half:
                    output_bits.append(0)
                    for _ in range(pending_bits):
                        output_bits.append(1)
                    pending_bits = 0
                elif low >= self.half:
                    output_bits.append(1)
                    for _ in range(pending_bits):
                        output_bits.append(0)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high <= 3 * self.quarter:
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                    
                low = low << 1
                high = (high << 1) | 1
                low = low & (self.one - 1)
                high = high & (self.one - 1)
        
        # Flush remaining bits
        pending_bits += 1
        if low < self.quarter:
            output_bits.append(0)
            for _ in range(pending_bits):
                output_bits.append(1)
        else:
            output_bits.append(1)
            for _ in range(pending_bits):
                output_bits.append(0)
        
        # Convert bits to bytes
        return self._bits_to_bytes(output_bits)
    
    def decode(self, compressed: bytes, num_symbols: int, probs: np.ndarray) -> np.ndarray:
        """
        Decode compressed bytes back to symbols
        Args:
            compressed: compressed bytes
            num_symbols: number of symbols to decode
            probs: probability distribution [num_symbols, 256]
        Returns:
            decoded uint8 array
        """
        bits = self._bytes_to_bits(compressed)
        
        low = 0
        high = self.one
        value = 0
        
        # Initialize value with first 32 bits
        for i in range(min(self.precision, len(bits))):
            value = (value << 1) | bits[i]
        
        bit_index = self.precision
        symbols = []
        
        for i in range(num_symbols):
            prob = probs[i]
            
            # Create cumulative distribution
            cumulative = np.cumsum(prob)
            cumulative = np.concatenate([[0], cumulative])
            cumulative = (cumulative * (high - low)).astype(np.uint64) + low
            
            # Find symbol
            symbol = np.searchsorted(cumulative[1:], value, side='right')
            symbols.append(symbol)
            
            low = cumulative[symbol]
            high = cumulative[symbol + 1]
            
            # Renormalization
            while True:
                if high <= self.half:
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    value -= self.half
                elif low >= self.quarter and high <= 3 * self.quarter:
                    low -= self.quarter
                    high -= self.quarter
                    value -= self.quarter
                else:
                    break
                    
                low = low << 1
                high = (high << 1) | 1
                low = low & (self.one - 1)
                high = high & (self.one - 1)
                
                if bit_index < len(bits):
                    value = ((value << 1) | bits[bit_index]) & (self.one - 1)
                    bit_index += 1
                else:
                    value = (value << 1) & (self.one - 1)
        
        return np.array(symbols, dtype=np.uint8)
    
    def _bits_to_bytes(self, bits):
        """Convert bit list to bytes"""
        # Pad to multiple of 8
        while len(bits) % 8 != 0:
            bits.append(0)
        
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            byte_array.append(byte)
        
        return bytes(byte_array)
    
    def _bytes_to_bits(self, byte_data):
        """Convert bytes to bit list"""
        bits = []
        for byte in byte_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits


class ProbabilityModel(nn.Module):
    """Neural network to predict probability distribution for entropy coding"""
    
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: quantized latent [B, C, T, H, W] in float32
        Returns:
            logits: [B, C, T, H, W, 256] probability distribution
        """
        # Use causal masking for autoregressive modeling
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)  # [B, 256, T, H, W]
        
        # Rearrange to [B, C, T, H, W, 256]
        h = h.unsqueeze(1).expand(-1, x.shape[1], -1, x.shape[2], x.shape[3], x.shape[4])
        
        return h


class LatentQuantizer(nn.Module):
    """Learnable quantization module"""
    
    def __init__(self, num_levels=256):
        super().__init__()
        self.num_levels = num_levels
        
        # Learnable scale and zero point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(127.0))
        
    def forward(self, x):
        """
        Quantize input to uint8 range with STE
        Args:
            x: [B, C, T, H, W] float tensor
        Returns:
            quantized: [B, C, T, H, W] float tensor (quantized values)
            quantized_int: [B, C, T, H, W] uint8 tensor
        """
        # 之前的min-max量化方法
        # Update scale based on input range
        with torch.no_grad():
            x_min = x.min()
            x_max = x.max()

            # x_min = x.float().quantile(0.01).to(x.dtype)   # 忽略最极端的 0.1%
            # x_max = x.float().quantile(0.99).to(x.dtype)
        
            self.scale = (x_max - x_min) / (self.num_levels - 1)
            self.zero_point = -x_min / self.scale
        
        # Quantize
        x_scaled = x / self.scale + self.zero_point
        x_clipped = torch.clamp(x_scaled, 0, self.num_levels - 1)
        quantized_int = torch.round(x_clipped)
        
        # Straight-through estimator for gradients
        quantized = quantized_int + (x_clipped - x_clipped.detach())
        
        # Dequantize
        dequantized = (quantized - self.zero_point) * self.scale
        
        return dequantized, quantized_int.to(torch.uint8)

        # 新的3σ量化方法
        # 计算scale
        # self.Qmax=50
        # with torch.no_grad():
        #     sigma = x.std()
        #     scale = 3.0 * sigma / self.Qmax   # 3σ → Qmax

        # # 量化（对称）
        # x_scaled = x / scale
        # q = torch.round(x_scaled)
        # q = torch.clamp(q, -self.Qmax, self.Qmax)

        # # STE
        # q_ste = q + (x_scaled - x_scaled.detach())

        # # 反量化
        # x_hat = q_ste * scale
        # self.scale = scale
        # self.zero_point = 0.0

        # return x_hat, q.to(torch.int8)

    def dequantize(self, quantized_int):
        """
        Dequantize uint8 tensor back to float
        Args:
            quantized_int: [B, C, T, H, W] uint8 tensor
        Returns:
            dequantized: [B, C, T, H, W] float tensor
        """
        quantized_float = torch.tensor(quantized_int).to(dtype=torch.float32)
        output = (quantized_float - self.zero_point) * self.scale
        output = output.to(torch.bfloat16)
        return output
    
    def get_quantization_params(self):
        """Get scale and zero_point for decoder"""
        # Pack into 3 bytes: scale (2 bytes float16) + zero_point (1 byte uint8)
        scale_half = self.scale.half()
        zero_point_uint8 = torch.clamp(self.zero_point, 0, 255).to(torch.uint8)
        
        scale_bytes = struct.pack('e', scale_half.item())  # 2 bytes
        zp_bytes = struct.pack('B', zero_point_uint8.item())  # 1 byte
        
        return scale_bytes + zp_bytes
    
    def set_quantization_params(self, param_bytes):
        """Set scale and zero_point from bytes"""
        scale = struct.unpack('e', param_bytes[:2])[0]
        zero_point = struct.unpack('B', param_bytes[2:3])[0]
        return torch.tensor(scale), torch.tensor(float(zero_point))


class E2ECompressionSystem(nn.Module):
    """End-to-end compression system"""
    
    def __init__(self):
        super().__init__()
        self.quantizer = LatentQuantizer()
        self.prob_model = ProbabilityModel(channels=16)
        self.arithmetic_coder = ArithmeticCoder()
        
    def forward(self, x, training=True):
        """
        Forward pass for training
        Args:
            x: input tensor
            training: if True, use differentiable path
        Returns:
            latent_dequantized: dequantized latent
            latent_quantized: quantized latent (for rate loss)
            probs: predicted probability distribution
        """
        latent = x  # [B, 16, T, H, W]
        
        # Quantize with STE
        latent_dequantized, latent_uint8 = self.quantizer(latent)
        
        # Predict probability distribution
        logits = self.prob_model(latent_dequantized)
        probs = F.softmax(logits, dim=-1)  # [B, C, T, H, W, 256]
               
        return latent_dequantized, latent_uint8, probs
    
    # "decoded" must be provided from outside decoder
    def compute_loss(self, x, decoded, latent_uint8, probs, lambda_rate=0.01):
        """
        Compute end-to-end loss
        Args:
            x: original input
            decoded: reconstructed output
            latent_uint8: quantized latent
            probs: predicted probability distribution
            lambda_rate: rate-distortion tradeoff
        Returns:
            total_loss, distortion_loss, rate_loss
        """
        # Distortion loss (MSE or perceptual loss)
        distortion = F.mse_loss(decoded, x)
        
        # Rate loss (cross-entropy with predicted distribution)
        # Compute negative log likelihood
        latent_flat = latent_uint8.flatten()
        probs_flat = probs.reshape(-1, 256)
        
        # Gather probabilities of actual symbols
        indices = latent_flat.long().unsqueeze(1)
        selected_probs = torch.gather(probs_flat, 1, indices).squeeze(1)
        
        # Rate in bits (negative log likelihood)
        rate = -torch.log2(selected_probs + 1e-10).mean()
        
        # Total loss
        total_loss = distortion + lambda_rate * rate
        
        return total_loss, distortion, rate
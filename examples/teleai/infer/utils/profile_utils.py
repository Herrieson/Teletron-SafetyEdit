"""
Utilities for profiling model MACs and execution time using thop
"""
import time
import torch
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import json


class ProfileStats:
    """Class to accumulate profiling statistics"""
    def __init__(self):
        self.macs = defaultdict(float)
        self.times = defaultdict(float)
        self.call_counts = defaultdict(int)
        self.macs_cache = {}  # Cache for calculated MACs

    def add_mac(self, name: str, mac: float):
        """Add MAC count for a module"""
        self.macs[name] += mac
        self.call_counts[name] += 1

    def add_time(self, name: str, elapsed: float):
        """Add execution time for a module"""
        self.times[name] += elapsed

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all statistics"""
        summary = {}
        all_names = set(self.macs.keys()) | set(self.times.keys())

        for name in all_names:
            summary[name] = {
                'macs (GMACs)': self.macs.get(name, 0) / 1e9,
                'time (ms)': self.times.get(name, 0) * 1000,
                'calls': self.call_counts.get(name, 0),
                'avg_time_per_call (ms)': self.times.get(name, 0) * 1000 / max(self.call_counts.get(name, 1), 1),
            }
        return summary

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("PROFILING SUMMARY")
        print("=" * 80)

        # Group by encoder/decoder
        encoder_stats = {}
        decoder_stats = {}

        for name, stats in summary.items():
            if name.startswith("encoder"):
                encoder_stats[name] = stats
            elif name.startswith("decoder"):
                decoder_stats[name] = stats

        if encoder_stats:
            print("\n[ENCODER]")
            print("-" * 80)
            total_enc_mac = 0
            total_enc_time = 0
            for name, stats in sorted(encoder_stats.items()):
                print(f"{name}:")
                print(f"  MACs: {stats['macs (GMACs)']:.4f} GMACs")
                print(f"  Time: {stats['time (ms)']:.2f} ms")
                print(f"  Calls: {stats['calls']}")
                total_enc_mac += stats['macs (GMACs)']
                total_enc_time += stats['time (ms)']
            print("-" * 80)
            print(f"TOTAL ENCODER - MACs: {total_enc_mac:.4f} GMACs, Time: {total_enc_time:.2f} ms")

        if decoder_stats:
            print("\n[DECODER]")
            print("-" * 80)
            total_dec_mac = 0
            total_dec_time = 0
            for name, stats in sorted(decoder_stats.items()):
                print(f"{name}:")
                print(f"  MACs: {stats['macs (GMACs)']:.4f} GMACs")
                print(f"  Time: {stats['time (ms)']:.2f} ms")
                print(f"  Calls: {stats['calls']}")
                total_dec_mac += stats['macs (GMACs)']
                total_dec_time += stats['time (ms)']
            print("-" * 80)
            print(f"TOTAL DECODER - MACs: {total_dec_mac:.4f} GMACs, Time: {total_dec_time:.2f} ms")

        print("\n" + "=" * 80)
        if encoder_stats and decoder_stats:
            print(f"OVERALL - MACs: {total_enc_mac + total_dec_mac:.4f} GMACs, Time: {total_enc_time + total_dec_time:.2f} ms")
        elif encoder_stats:
            print(f"OVERALL - MACs: {total_enc_mac:.4f} GMACs, Time: {total_enc_time:.2f} ms")
        elif decoder_stats:
            print(f"OVERALL - MACs: {total_dec_mac:.4f} GMACs, Time: {total_dec_time:.2f} ms")
        print("=" * 80 + "\n")

    def save_to_file(self, filepath: str):
        """Save summary to JSON file with encoder/decoder totals"""
        summary = self.get_summary()

        # Calculate encoder and decoder totals
        encoder_stats = {}
        decoder_stats = {}

        for name, stats in summary.items():
            if name.startswith("encoder"):
                encoder_stats[name] = stats
            elif name.startswith("decoder"):
                decoder_stats[name] = stats

        # Calculate totals
        total_enc_mac = sum(stats['macs (GMACs)'] for stats in encoder_stats.values())
        total_enc_time = sum(stats['time (ms)'] for stats in encoder_stats.values())
        total_dec_mac = sum(stats['macs (GMACs)'] for stats in decoder_stats.values())
        total_dec_time = sum(stats['time (ms)'] for stats in decoder_stats.values())

        # Add summary section
        output = {
            'modules': summary,
            'summary': {
                'encoder': {
                    'total_macs_gmacs': total_enc_mac,
                    'total_time_ms': total_enc_time,
                    'module_count': len(encoder_stats)
                },
                'decoder': {
                    'total_macs_gmacs': total_dec_mac,
                    'total_time_ms': total_dec_time,
                    'module_count': len(decoder_stats)
                },
                'overall': {
                    'total_macs_gmacs': total_enc_mac + total_dec_mac,
                    'total_time_ms': total_enc_time + total_dec_time
                }
            }
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)


class ProfilerContext:
    """Context manager for profiling"""
    def __init__(self, stats: ProfileStats, name: str, enable: bool = True, macs: float = 0.0):
        self.stats = stats
        self.name = name
        self.enable = enable
        self.macs = macs
        self.start_time = None
        self.start_cuda_time = None

    def __enter__(self):
        if not self.enable:
            return self
        self.start_time = time.time()
        if torch.cuda.is_available():
            self.start_cuda_time = torch.cuda.Event(enable_timing=True)
            self.start_cuda_time.record()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        elapsed = time.time() - self.start_time
        if torch.cuda.is_available():
            end_cuda_time = torch.cuda.Event(enable_timing=True)
            end_cuda_time.record()
            torch.cuda.synchronize()
            cuda_elapsed = self.start_cuda_time.elapsed_time(end_cuda_time) / 1000.0
            self.stats.add_time(self.name, cuda_elapsed)
        else:
            self.stats.add_time(self.name, elapsed)

        # Add MACs if provided
        if self.macs > 0:
            self.stats.add_mac(self.name, self.macs)


class MACsProfiler:
    """Profiler for calculating MACs using thop library"""

    @staticmethod
    def calculate_macs_thop(module: torch.nn.Module,
                             inputs: Tuple[Any, ...],
                             name: str = "module") -> float:
        """
        Calculate MACs using thop library (actual measurement)

        Args:
            module: PyTorch module to profile
            inputs: Input arguments for the module
            name: Name for logging

        Returns:
            MACs count (total number of multiply-accumulate operations)
        """
        try:
            from thop import profile
            
            # Ensure inputs are tuples
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)
            
            # Calculate MACs using thop
            macs, params = profile(module, inputs=inputs, verbose=False)
            return macs
            
        except ImportError:
            print("Warning: thop not installed. Install with: pip install thop")
            return 0.0
        except Exception as e:
            print(f"Warning: Failed to calculate MACs for {name}: {e}")
            return 0.0

    @staticmethod
    def estimate_vae_macs(batch_size: int, num_frames: int, height: int, width: int) -> float:
        """
        Estimate MACs for VAE encoder/decoder (fallback if thop not available)
        
        Args:
            batch_size: Batch size
            num_frames: Number of frames
            height: Frame height
            width: Frame width

        Returns:
            Estimated MACs
        """
        # VAE typically has a compression factor of 8x8 spatially
        # and 4x temporally for video
        latent_h, latent_w = height // 8, width // 8
        latent_t = (num_frames + 3) // 4  # Temporal compression

        # Encoder MACs (rough estimate based on 3D convolutions)
        # Input: [B, 3, T, H, W] -> [B, 16, T/4, H/8, W/8]

        # Stage 1: [B, 3, T, H, W] -> [B, 64, T, H/2, W/2]
        mac1 = batch_size * 64 * 3 * 3 * 3 * 3 * latent_t * 2 * (height // 2) * (width // 2)

        # Stage 2: [B, 64, T, H/2, W/2] -> [B, 128, T, H/4, W/4]
        mac2 = batch_size * 128 * 64 * 3 * 3 * 3 * latent_t * (height // 4) * (width // 4)

        # Stage 3: [B, 128, T, H/4, W/4] -> [B, 256, T, H/8, W/8]
        mac3 = batch_size * 256 * 128 * 3 * 3 * 3 * latent_t * (height // 8) * (width // 8)

        # Stage 4: [B, 256, T, H/8, W/8] -> [B, 16, T/4, H/8, W/8]
        mac4 = batch_size * 16 * 256 * 4 * 3 * 3 * (latent_t) * (height // 8) * (width // 8)

        total_macs = mac1 + mac2 + mac3 + mac4
        return total_macs


@contextmanager
def profile_time(stats: ProfileStats, name: str, enable: bool = True, macs: float = 0.0):
    """Context manager for profiling execution time and MACs"""
    ctx = ProfilerContext(stats, name, enable, macs)
    yield ctx.__enter__()
    ctx.__exit__(None, None, None)


def profile_module_with_thop(stats: ProfileStats, 
                               module: torch.nn.Module,
                               name: str,
                               inputs: Tuple[Any, ...],
                               enable: bool = True):
    """
    Profile a module using thop for accurate MACs calculation
    
    Args:
        stats: ProfileStats object to store results
        module: PyTorch module to profile
        name: Name for the module
        inputs: Input arguments for the module
        enable: Whether profiling is enabled
    """
    if not enable:
        return module(*inputs) if isinstance(inputs, (tuple, list)) else module(inputs)
    
    # Calculate MACs only on first call (cached)
    cache_key = f"{name}_macs"
    if cache_key not in stats.macs_cache:
        # Clone inputs to avoid modifying the originals
        cloned_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                cloned_inputs.append(inp.detach().clone())
            else:
                cloned_inputs.append(inp)
        
        macs = MACsProfiler.calculate_macs_thop(module, tuple(cloned_inputs), name)
        stats.macs_cache[cache_key] = macs
    
    # Profile time and use cached MACs
    macs = stats.macs_cache.get(cache_key, 0.0)
    
    with profile_time(stats, name, enable=enable, macs=macs):
        output = module(*inputs)
    
    return output

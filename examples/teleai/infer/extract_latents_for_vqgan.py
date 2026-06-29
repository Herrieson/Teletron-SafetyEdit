"""
潜在表示提取脚本 - 用于 VQGAN 自回归预测实验

用途：从大量视频中提取 Quantizer 后的潜在表示 (latent_uint8_3)
数据形状：(8, T/4, H/32, W/32) - uint8 格式

运行方式：
    python extract_latents_for_vqgan.py --video_dir /path/to/videos --output_dir /path/to/latents --num_videos 1000
"""

import torch
import os
import sys
import decord
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from pipelines import WanVideoI2VPipeline
from torchvision.io import read_video
import torch.nn.functional as F


# ==================== 配置 ====================

PIPELINE_CONFIG = dict(
    model_config=dict(
        dit=dict(
            path="/gemini/platform/shared/yifq1/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_binarycanny_dynamicstride_tae/distilled_f29_bfcanny.pt",
            config=dict(
                has_image_input=True,
                patch_size=[1, 2, 2],
                in_dim=36+16+4,
                dim=1536,
                ffn_dim=8960,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12,
                num_layers=30,
                eps=1e-6,
                has_image_pos_emb=True,
                has_compressor={"use":True, "up_T":True, "pixel_shuffle": False, "enable_attn": True},
                has_quantizer=True,
            ),
        ),
        encoder=dict(
            vae=dict(
                path="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/taew2_1.pth",
                type="TeleaiVideoTAE_2_1",
                tiler_kwargs=dict(
                    tiled=False,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                    has_mask=True,
                ),
            ),
            text_encoder=dict(
                path="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        ),
        keyframe=dict(
            model_path_i="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/keyframe_i.pth.tar",
            model_path_p="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/keyframe_p.pth.tar",
        ),
    ),
    torch_dtype=torch.bfloat16,
    device="cuda",
)

EXTRACT_CONFIG = {
    "num_frames": 29,           # 每个clip的帧数
    "target_height": 720,        # 目标高度
    "target_width": 1280,        # 目标宽度
}


# ==================== 工具函数 ====================

def make_json_serializable(obj):
    """将对象转换为 JSON 可序列化的格式"""
    if isinstance(obj, torch.dtype):
        return str(obj)
    elif hasattr(obj, 'dtype'):  # numpy dtype
        return str(obj.dtype)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    else:
        return obj


def pad_tensor(tensor):
    """Pad tensor to make dimensions divisible by 64"""
    C, T, H, W = tensor.shape
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    padding = (0, 0, pad_w, pad_h)
    padded = F.pad(tensor, padding, padding_mode="constant")
    return padded, padding


def resize_with_pyav(frames, target_size=(720, 1280)):
    """使用 PyAV 进行高质量缩放"""
    import av
    T, orig_h, orig_w, _ = frames.shape
    resized_frames = []
    for i in range(T):
        frame = av.VideoFrame.from_ndarray(frames[i], format='rgb24')
        resized_frame = frame.reformat(target_size[1], target_size[0],
                                       interpolation=av.video.reformatter.Interpolation.LANCZOS)
        resized_frame_array = resized_frame.to_ndarray(format='rgb24')
        resized_frames.append(resized_frame_array)
    return np.stack(resized_frames, axis=0)


def get_video_list(video_dir, num_videos=None, extension=".mp4"):
    """获取视频列表"""
    video_dir = Path(video_dir)
    videos = sorted(video_dir.glob(f"**/*{extension}"))
    if num_videos is not None:
        videos = videos[:num_videos]
    return [str(v) for v in videos]


# ==================== 核心提取类 ====================

class LatentExtractor:
    """潜在表示提取器"""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """加载 Pipeline"""
        print("加载 Pipeline...")
        self.pipeline = WanVideoI2VPipeline(PIPELINE_CONFIG)
        self.pipeline.dit.eval()

    @torch.no_grad()
    def extract_single_video(self, video_path, save_dir, video_id=None):
        """
        提取单个视频的潜在表示

        Args:
            video_path: 视频路径
            save_dir: 保存目录
            video_id: 视频ID（用于命名）

        Returns:
            info_dict: 包含元信息的字典
        """
        try:
            # 读取视频
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            num_clips = total_frames // EXTRACT_CONFIG["num_frames"]

            if num_clips == 0:
                print(f"跳过 {video_path}: 帧数不足")
                return None

            # 创建保存目录
            save_dir = Path(save_dir)
            video_name = Path(video_path).stem
            video_save_dir = save_dir / video_name
            video_save_dir.mkdir(parents=True, exist_ok=True)

            # 元信息
            meta_info = {
                "video_path": video_path,
                "video_id": video_id or video_name,
                "total_frames": total_frames,
                "num_clips": num_clips,
                "clips": [],
                "extract_time": datetime.now().isoformat(),
            }

            # 提取每个 clip 的潜在表示
            for clip_idx in range(num_clips):
                # 读取视频帧
                start_frame = clip_idx * EXTRACT_CONFIG["num_frames"]
                end_frame = start_frame + EXTRACT_CONFIG["num_frames"]
                chunk_frames = vr[start_frame:end_frame].asnumpy()  # [T, H, W, C] uint8

                # Resize 到目标尺寸（直接使用 uint8 数据）
                chunk_resized = resize_with_pyav(
                    chunk_frames,
                    target_size=(EXTRACT_CONFIG["target_height"], EXTRACT_CONFIG["target_width"])
                )
                chunk_tensor = torch.from_numpy(chunk_resized).permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]

                # 保存原始尺寸信息
                original_h, original_w = chunk_frames.shape[1], chunk_frames.shape[2]

                # Pad 到 64 的倍数
                chunk_tensor, padding = pad_tensor(chunk_tensor)
                chunk_tensor = chunk_tensor / 255.0

                # ========== 关键提取步骤 ==========
                # 1. VAE Encode
                self.pipeline.load_models_to_device(['vae'])
                latents = self.pipeline.vae.encode(
                    chunk_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device=self.device),
                    device=self.device,
                    tiled=False
                )

                # 2. Compressor Down (获取 3 个尺度)
                self.pipeline.load_models_to_device(self.pipeline.dit)
                down_latents1, down_latents2, down_latents3 = self.pipeline.dit.compressor_down(latents)
                # down_latents1: [1, 16, T, H/8, W/8]
                # down_latents2: [1, 16, T, H/16, W/16]
                # down_latents3: [1, 8, T, H/32, W/32]  <- 这是我们要的

                # 3. Quantize (获取 uint8 潜在表示)
                dequantized, latent_uint8, _ = self.pipeline.dit.quantizer(down_latents3)
                # latent_uint8: [1, 8, T/4, H/32, W/32] uint8

                # 转换为 numpy 并保存
                latent_np = latent_uint8[0].cpu().numpy()  # [8, T/4, H/32, W/32]

                # 保存为 npz 格式
                clip_save_path = video_save_dir / f"clip_{clip_idx:04d}.npz"
                np.savez_compressed(
                    clip_save_path,
                    latent=latent_np,
                    shape=latent_np.shape,
                    dtype=str(latent_np.dtype)
                )

                # 记录元信息
                clip_info = {
                    "clip_idx": clip_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "latent_shape": list(latent_np.shape),
                    "latent_path": str(clip_save_path),
                    "original_size": [original_w, original_h],
                    "padded_size": [chunk_tensor.shape[3], chunk_tensor.shape[2]],
                    "padding": list(padding),
                }
                meta_info["clips"].append(clip_info)

            # 保存元信息
            meta_path = video_save_dir / "meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_info, f, indent=2, ensure_ascii=False)

            return meta_info

        except Exception as e:
            print(f"处理 {video_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def extract_batch(self, video_list, save_dir, max_videos=None):
        """批量提取"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 创建可序列化的配置副本
        serializable_pipeline_config = make_json_serializable(PIPELINE_CONFIG)

        global_meta = {
            "extract_config": EXTRACT_CONFIG,
            "videos": [],
            "extract_time": datetime.now().isoformat(),
        }

        if max_videos:
            video_list = video_list[:max_videos]

        for idx, video_path in enumerate(tqdm(video_list, desc="提取潜在表示")):
            video_id = f"vid_{idx:06d}"
            meta = self.extract_single_video(video_path, save_dir, video_id)
            if meta:
                global_meta["videos"].append(meta)

        # 保存全局元信息
        global_meta_path = save_dir / "global_meta.json"
        with open(global_meta_path, "w", encoding="utf-8") as f:
            json.dump(global_meta, f, indent=2, ensure_ascii=False)

        print(f"\n提取完成！")
        print(f"处理视频数: {len(global_meta['videos'])}")
        print(f"保存目录: {save_dir}")
        return global_meta


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="提取视频潜在表示用于 VQGAN 实验")
    parser.add_argument("--video_dir", type=str, required=True, help="视频目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_videos", type=int, default=None, help="处理视频数量")
    parser.add_argument("--video_list", type=str, default=None, help="视频列表文件（每行一个路径）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    args = parser.parse_args()

    # 获取视频列表
    if args.video_list:
        with open(args.video_list, "r") as f:
            video_list = [line.strip() for line in f if line.strip()]
    else:
        video_list = get_video_list(args.video_dir, args.num_videos)

    print(f"找到 {len(video_list)} 个视频")

    # 创建提取器
    extractor = LatentExtractor(device=args.device)

    # 批量提取
    extractor.extract_batch(video_list, args.output_dir)


if __name__ == "__main__":
    '''
        python extract_latents_for_vqgan.py \
            --video_dir /path/to/videos \
            --output_dir /path/to/latents \
            --num_videos 1000
    '''
    main()

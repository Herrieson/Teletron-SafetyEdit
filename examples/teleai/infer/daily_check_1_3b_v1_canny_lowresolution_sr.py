"""
Daily Check Script for WanVideoI2V Model with Canny Edge Processing

This script processes video chunks using WanVideoI2V model with Canny edge detection
for video reconstruction and visualization.
"""

import torch
import os
import sys
import decord
import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import copy
import numpy as np
import cv2
import imageio
import av
import math
import struct
import gzip
from scipy.special import comb
from scipy.sparse import csr_matrix
from io import BytesIO


import torch.multiprocessing as mp
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_video
import torchvision.transforms.functional as F

# Import model pipeline
from pipelines import WanVideoI2VPipeline
from diffusers.utils import export_to_video
# cd examples/teleai/infer/third_party/keyframe_src/cpp/
# python3 setup.py install
from keyframes.keyframe_encoder import encoder_keyframe_fun, init_encoder
from keyframes.keyframe_decoder import decoder_keyframe_fun, init_decoder

# Configuration Section
# ============================================================================

# Model checkpoint configuration
folder = "/gemini/user/shared/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_randomcanny_tae_lowresolution_encaug"
with open(os.path.join(folder, "latest_checkpointed_iteration.txt"), "r") as f:
    iter_num = f.read().strip()

CKPT_PATH = f"{folder}/iter_{int(iter_num):07d}/mp_rank_00/model_optim_rng.pt"
# CKPT_PATH = '/gemini/space/zxl/train_gvc_14b_0129/workdirs_1_3b/train_randomcanny_lr_pd4-2_1e-7/iter_0004250/mp_rank_00/model_optim_rng.pt'
SAVEDIR = f"/gemini/user/shared/vis_result/20260316/jk_360sr_distilled"

# Default inference parameters
DEFAULT_CONFIG = {
    "num_frames": 29,
    "cfg_scale": 1,
    "num_inference_step": 10,
    "save_fps": 25,
    "tiled": False,
    "qp_i": 42,
    "qp_p": 42,
}

# Global settings
NUM_OF_CLIPS = 50
GPU_IDS = [0]

# Dataset Configuration
# ============================================================================

# def get_video_list():
#     """
#     Get the list of videos to process.

#     Returns:
#         List[List[str]]: List of video file paths and corresponding Canny files
#     """
#     video_list = []
#     video_list.append([
#         "/gemini/space/zxl/dataset/shop.mp4",
#         "/gemini/space/data/batch2/hksb9d_1806279446274838528.mp4"
#     ])
#     return video_list

# def get_video_list():
#     root_dir = "/gemini/space/code/data/yifq1/yifq/public_benchmarks/MCL_JCV/720P/YUV_source"
#     video_list = [[os.path.join(root_dir, file), os.path.join(root_dir, file)] for file in os.listdir(root_dir) if ".mp4" in file]
#     print(video_list)
#     return video_list[0:1]

def get_video_list():
    # root_dir = "/gemini/space/zxl/dataset/sr_videos"
    # video_list = [[os.path.join(root_dir, file), os.path.join(root_dir, file)] for file in os.listdir(root_dir) if ".mp4" in file]
    # print(video_list)
    # return video_list[0:1]

    return [["/gemini/user/shared/scripts/30zhanting.mp4", "/gemini/space/zxl/dataset/sr_videos/day_mixedtraffic_360p_384k_fps30_gop60.mp4"]]



# Model Pipeline Configuration
# ============================================================================

PIPELINE_CONFIG = dict(
    model_config=dict(
        dit=dict(
            path=CKPT_PATH,  # ema_model.pt
            config=dict(
                has_image_input=True,  # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36+16+4,  # t2v:16 i2v:36, s2v: 20 * numof_s
                dim=1536,  # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=8960,  # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12,  # 1.3B:12 10B:40 14B:40
                num_layers=30,  # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=True,  # be true
                has_compressor={"use":True, "up_T":True, "pixel_shuffle": True, "enable_attn": True},
                has_quantizer=True,
            ),
        ),
        encoder=dict(
            vae=dict(
                path="/gemini/platform/shared/yifq1/Wan2.1-Fun-V1.1-1.3B-InP/taew2_1.pth",
                type="TeleaiVideoTAE_2_1",
                tiler_kwargs=dict(
                    tiled=False,  #
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                    has_mask=True,  # True 会带mask的4维
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



# Utility Functions
# ============================================================================

def recover(tensor):
    """Normalize tensor from [-1, 1] to [0, 255]"""
    return (tensor * 0.5 + 0.5) * 255

def load_pipeline(device: torch.device) -> WanVideoI2VPipeline:
    """Load the WanVideoI2V pipeline"""
    pipe = WanVideoI2VPipeline(PIPELINE_CONFIG)
    return pipe

def load_video_cthw_norm(video):
    frames = video  # T,H,W,C
    frames, padding = pad(frames)
    new_H, new_W = frames.shape[2], frames.shape[3]
    print(f"frames.shape: {frames.shape}")
    return frames, new_W, new_H, padding  # T,H,W,C → C,T,H,W

def random_dropout_canny(canny_array, dropout_rate):
    """
    Randomly set some 1s to 0s in canny array

    Args:
        canny_array: Array of shape (T, height, width) containing only 0s and 1s
        dropout_rate: Proportion to drop (0 to 1)

    Returns:
        Processed array
    """
    result = canny_array.copy()
    ones_mask = (result == 1)
    random_values = np.random.random(result.shape)
    to_drop = (ones_mask & (random_values < dropout_rate))
    result[to_drop] = 0
    return result

def resize_canny_direct(canny_array, target_size=(448, 832)):
    """
    Resize canny array directly

    Args:
        canny_array: (t, h, w) numpy array
        target_size: Target size (height, width)

    Returns:
        resized_canny: (t, target_h, target_w) numpy array
    """
    canny_tensor = torch.from_numpy(canny_array).float()
    canny_tensor = canny_tensor.unsqueeze(1)
    # Use nearest neighbor interpolation for binary data
    resized = torch.nn.functional.interpolate(canny_tensor, size=target_size, mode='nearest')
    resized_canny = resized.squeeze(1).numpy()
    # Re-binarize
    resized_canny = (resized_canny > 0.4).astype(np.uint8)
    return resized_canny

def pad(tensor, divisor=128):
    """
    Pad tensor to make height and width divisible by divisor (default: 128)

    Args:
        tensor: Input tensor with shape (C, T, H, W)
        divisor: The number to make H and W divisible by (default: 128)

    Returns:
        padded_tensor: Padded tensor with shape (C, T, H', W')
        padding: Tuple containing the padding amounts for (left, right, top, bottom)
    """
    C, T, H, W = tensor.shape

    # Calculate padding needed
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor

    # Padding applied as (left, right, top, bottom)
    padding = (0, 0, pad_w, pad_h)

    padded_tensor = F.pad(tensor, padding, padding_mode='constant')
    return padded_tensor, padding



def resize_canny_with_pyav(frames_array, target_size=(448, 832)):
    """Resize video frames using PyAV with Lanczos interpolation"""
    T, orig_height, orig_width, _ = frames_array.shape
    resized_frames = []

    for i in range(T):
        frame = av.VideoFrame.from_ndarray(frames_array[i], format='rgb24')
        resized_frame = frame.reformat(target_size[1], target_size[0],
                                     interpolation=av.video.reformatter.Interpolation.LANCZOS)
        resized_frame_array = resized_frame.to_ndarray(format='rgb24')
        resized_frames.append(resized_frame_array)

    return np.stack(resized_frames, axis=0)


def binary_array_from_rgb(canny_images):
    """Convert RGB canny images to binary array"""
    gray_images = np.mean(canny_images, axis=3)
    gray_uint8 = gray_images.astype(np.uint8)
    binary = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
              for gray in gray_uint8]
    return np.stack(binary).astype(bool)


def _encode_keyframe(keyframe_encoder, first_frame_bytes, last_frame_bytes, qp_i, qp_p):
    # 对关键帧进行压缩
    first_frame_bytes_io = BytesIO(first_frame_bytes)
    last_frame_bytes_io = BytesIO(last_frame_bytes)
    encoder_data = encoder_keyframe_fun(keyframe_encoder, first_frame_bytes_io, last_frame_bytes_io, qp_i=qp_i, qp_p=qp_p)
    return encoder_data


def f8tl_process_clip(first_frame, last_frame, quality=95):
    """提取首尾帧并返回 JPEG 二进制数据

    参数:
        first_frame, last_frame
        quality: JPEG 质量，默认 95

    返回:
        first_frame_bytes: 第一帧 JPEG 二进制内容
        last_frame_bytes: 最后一帧 JPEG 二进制内容
    """
    # 写入内存缓冲区而不是落盘
    first_buffer = BytesIO()
    first_frame.save(first_buffer, format='JPEG', quality=quality)
    first_frame_bytes = first_buffer.getvalue()
    last_buffer = BytesIO()
    last_frame.save(last_buffer, format='JPEG', quality=quality)
    last_frame_bytes = last_buffer.getvalue()

    return first_frame_bytes, last_frame_bytes

def frame_handle(data):
    """处理帧数据（JPEG格式）"""
    try:
        frame = Image.open(BytesIO(data))
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')
        return frame
    except Exception as e:
        print("Error decoding frame data: %s", e)
        raise


def flf_process(keyframe_encoder, keyframe_decoder, first_frame, last_frame, qp_i, qp_p):
    first_frame_bytes, last_frame_bytes = f8tl_process_clip(first_frame, last_frame)
    frame_data = _encode_keyframe(keyframe_encoder, first_frame_bytes, last_frame_bytes, qp_i, qp_p)
    frame_data = BytesIO(frame_data)
    frame_bytes = len(frame_data.getvalue())
    first_frame, last_frame = decoder_keyframe_fun(keyframe_decoder, frame_data)
    last_frame = frame_handle(last_frame)
    if first_frame is not None:
        first_frame = frame_handle(first_frame)
    return first_frame, last_frame, frame_bytes

# Varint Encoding/Decoding Utilities
# ============================================================================

def encode_varint(value: int) -> bytes:
    """Encode integer as varint"""
    if value < 0:
        raise ValueError("varint only supports non-negative integers")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)

def decode_varint(buffer: bytes, start: int = 0):
    """Decode varint from buffer"""
    shift = 0
    result = 0
    pos = start
    while True:
        if pos >= len(buffer):
            raise EOFError("Unexpected end while decoding varint")
        b = buffer[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
        if shift > 64:
            raise ValueError("Varint too large")
    return result, pos

# Sparse Matrix Utilities
# ============================================================================

def private_save_npz(path: str, matrix: csr_matrix, gzip_level: int = 9) -> bytes:
    """Compress a scipy.sparse.csr_matrix into bytes"""
    if not isinstance(matrix, csr_matrix):
        raise TypeError("matrix must be scipy.sparse.csr_matrix")

    rows, cols = matrix.shape
    nnz = matrix.nnz
    indices = matrix.indices
    indptr = matrix.indptr

    parts = []
    # metadata
    parts.append(struct.pack('<III', rows, cols, nnz))

    # per-row encoding
    for r in range(rows):
        start = indptr[r]
        end = indptr[r+1]
        count = end - start
        parts.append(struct.pack('<I', count))  # u32 count

        if count == 0:
            continue

        row_idxs = indices[start:end]
        # Ensure row internal sort
        if not np.all(row_idxs[:-1] <= row_idxs[1:]):
            row_idxs = np.sort(row_idxs)

        # Write first index as u32
        first = int(row_idxs[0])
        parts.append(struct.pack('<I', first))

        # Write diffs for the rest using varint
        for val in row_idxs[1:]:
            diff = int(val) - first
            parts.append(encode_varint(diff))
            first = int(val)

    raw = b"".join(parts)
    compressed = gzip.compress(raw, compresslevel=gzip_level)
    with open(path, "wb") as f:
        f.write(compressed)
    return compressed

def decompress_csr(compressed_path: str) -> csr_matrix:
    """Decompress compressed CSR matrix back to csr_matrix"""
    with open(compressed_path, "rb") as f:
        compressed = f.read()
    raw = gzip.decompress(compressed)
    pos = 0

    if pos + 12 > len(raw):
        raise ValueError("Data too short for metadata")
    rows, cols, nnz = struct.unpack_from('<III', raw, pos)
    pos += 12

    indices_list = []
    indptr = [0]
    total = 0

    for r in range(rows):
        if pos + 4 > len(raw):
            raise EOFError("Unexpected EOF while reading row count")
        (count,) = struct.unpack_from('<I', raw, pos)
        pos += 4

        if count == 0:
            indptr.append(total)
            continue

        # Read first index (u32)
        if pos + 4 > len(raw):
            raise EOFError("Unexpected EOF while reading first index")
        (first_idx,) = struct.unpack_from('<I', raw, pos)
        pos += 4

        row_indices = [first_idx]
        prev = first_idx

        # Read remaining count-1 diffs (varint)
        for _ in range(count - 1):
            val, pos = decode_varint(raw, pos)
            curr = prev + val
            row_indices.append(curr)
            prev = curr

        indices_list.extend(row_indices)
        total += count
        indptr.append(total)

    if total != nnz:
        raise ValueError(f"Decoded nnz mismatch: header {nnz} vs decoded {total}")

    indices_arr = np.array(indices_list, dtype=np.uint32)
    data = np.ones(indices_arr.shape[0], dtype=np.uint8)
    indptr_arr = np.array(indptr, dtype=np.int32)

    return csr_matrix((data, indices_arr, indptr_arr), shape=(rows, cols))


# Canny Edge Detection Utilities
# ============================================================================

def get_percentile_thresholds(image, low_percentile=95, high_percentile=98):
    """Calculate adaptive Canny thresholds based on gradient percentiles"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    low_threshold = np.percentile(gradient_magnitude, low_percentile)
    high_threshold = np.percentile(gradient_magnitude, high_percentile)
    low_threshold = int(low_threshold)
    high_threshold = int(high_threshold)

    # Ensure minimum thresholds
    if low_threshold < 50:
        low_threshold = 50
        high_threshold = 150

    return low_threshold, high_threshold

def canny_process_one_frame(frame, low_threshold=50, high_threshold=150):
    """Process single frame using Canny edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    data = cv2.Canny(gray, low_threshold, high_threshold) / 255
    return data



# Video Segmentation Utilities
# ============================================================================

def motion_abs_simple(images, num_segments):
    """
    Segment video based on accumulated motion

    Args:
        images: torch.Tensor, shape (T, C, H, W) or (T, H, W, C)
        num_segments: Number of intermediate segment points to choose

    Returns:
        List[int]: Segment indices including first and last frames [0, ..., T-1]
    """
    # Convert to standard (T, C, H, W) format
    if images.dim() == 4:
        if images.shape[-1] == 3:  # THWC format
            images = images.permute(0, 3, 1, 2)  # Convert to TCHW

    T, C, H, W = images.shape

    # Handle edge cases
    if T <= 2 or num_segments <= 0:
        return [0, T-1]

    if num_segments >= T - 1:
        return list(range(T))

    # Convert to grayscale
    if C == 3:
        weights = torch.tensor([0.299, 0.587, 0.114],
                              device=images.device).view(1, 3, 1, 1)
        gray_images = (images.float() * weights).sum(dim=1, keepdim=True)
    else:
        gray_images = images.float().mean(dim=1, keepdim=True)

    # Calculate frame differences
    motion_diffs = []
    for t in range(T - 1):
        diff = torch.abs(gray_images[t] - gray_images[t + 1]).sum()
        motion_diffs.append(diff.item())

    # Calculate cumulative motion
    cumulative_motion = np.cumsum([0] + motion_diffs)
    total_motion = cumulative_motion[-1]

    # Use uniform segmentation if motion is minimal
    if total_motion < 1e-6:
        step = (T - 1) / (num_segments + 1)
        segment_indices = [0] + [int(i * step) for i in range(1, num_segments + 1)] + [T-1]
        return [int(idx) for idx in segment_indices]

    # Select segment points based on accumulated motion
    segment_indices = [0]  # Always include first frame

    # Calculate target cumulative motion for each segment
    segment_targets = [(i + 1) * total_motion / (num_segments + 1)
                      for i in range(num_segments)]

    # Find closest frame indices for each target
    current_idx = 0
    for target in segment_targets:
        while current_idx < T and cumulative_motion[current_idx] < target:
            current_idx += 1

        current_idx = min(current_idx, T - 1)
        if current_idx <= segment_indices[-1]:
            current_idx = segment_indices[-1] + 1

        if current_idx < T and current_idx not in segment_indices:
            segment_indices.append(current_idx)

    # Ensure last frame is included
    if segment_indices[-1] != T - 1:
        segment_indices.append(T - 1)

    # Remove duplicates and sort
    segment_indices = sorted(set(segment_indices))

    return segment_indices



def load_canny_cthw_norm(frames, W, H, canny_cnt):
    """Load and process Canny edge maps with normalization and padding"""
    # Select frames for Canny processing based on motion
    gaussian_indexes = motion_abs_simple(frames.permute(1, 0, 2, 3), canny_cnt)
    new_frames = frames.permute(1, 2, 3, 0).numpy()
    canny_images = []

    for i, idx in enumerate(range(len(new_frames))):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        if i % len(new_frames) in gaussian_indexes:
            # data = canny_process_one_frame(new_frames[idx], *get_percentile_thresholds(new_frames[idx]))
            data = canny_process_one_frame(new_frames[idx])
            canvas[data == 1] = 255
        canny_images.append(canvas)

    canny_images = np.stack(canny_images, axis=0)

    # Downsample for processing
    downsample_stride = round(H / 480 * 4)
    print(downsample_stride)
    canny_images = resize_canny_with_pyav(
        frames_array=canny_images,
        target_size=(H // downsample_stride, W // downsample_stride)
    )

    # Convert to binary array and back
    binary_array = binary_array_from_rgb(canny_images)
    canny_images = np.stack([np.stack([binary.astype(np.uint8) * 255] * 3, axis=-1)
                             for binary in binary_array], axis=0)

    # Upsample back to original size
    canny_images = resize_canny_with_pyav(
        frames_array=canny_images,
        target_size=(H*2, W*2)
    )

    canny_images = torch.from_numpy(canny_images).permute(3,0,1,2).contiguous()  # T,H,W,C
    canny_images = (canny_images / 255.0 - 0.5) / 0.5
    print(f"loaded canny shape: {canny_images.shape}")
    return canny_images


# Image Utilities
# ============================================================================

def concat_images(images, direction="horizontal", pad=0, pad_value=0):
    """Concatenate multiple images horizontally or vertically"""
    if len(images) == 1:
        return images[0]

    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]

    if direction == "horizontal":
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full(
            (height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype
        )
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == "vertical":
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full(
            (height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype
        )
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image

def concat_images_grid(images, cols, pad=0, pad_value=0):
    """Arrange images in a grid with specified number of columns"""
    new_images = []
    while len(images) > 0:
        new_image = concat_images(images[:cols], pad=pad, pad_value=pad_value)
        new_images.append(new_image)
        images = images[cols:]
    new_image = concat_images(
        new_images, direction="vertical", pad=pad, pad_value=pad_value
    )
    return new_image


# Inference Functions
# ============================================================================

def inference_worker(
    rank: int,
    world_size: int,
    inference_configs,
    gpu_ids: List[int],
    canny_cnt: int,
    save_dir: str,
    enable_key_frame,
):
    """Worker function for distributed inference"""
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    # Assign tasks to this worker
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return

    pipe = load_pipeline(device)

    for test_name, config in infer_task:
        # Load video
        vr = decord.VideoReader(test_name[0], ctx=decord.cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        NUM_OF_CLIPS = total_frames // DEFAULT_CONFIG["num_frames"]
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        if enable_key_frame:
            keyframe_encoder = init_encoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)
            keyframe_decoder = init_decoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)

        # Process video chunks
        for t in range(NUM_OF_CLIPS):
            try:
                save_name = os.path.splitext(os.path.basename(test_name[0]))[0]
                save_path = os.path.join(save_dir, f"{save_name}_p{t}.mp4")

                # if os.path.exists(save_path):
                #     print(f"[Rank {rank}] File exists, skipping: {save_path}")
                #     continue

                # Extract chunk
                chunk_frames = vr[t*DEFAULT_CONFIG["num_frames"]:(t+1)*DEFAULT_CONFIG["num_frames"]].asnumpy()
                chunk_tensor = torch.from_numpy(chunk_frames)
                chunk_tensor = chunk_tensor.permute(3, 0, 1, 2).contiguous()  # C,T,H,W

                chunk_frames_np = chunk_tensor.permute(1, 2, 3, 0).numpy()  # C,T,H,W → T,H,W,C
                chunk_resized = resize_canny_with_pyav(chunk_frames_np, target_size=(360, 640))
                chunk_tensor = torch.from_numpy(chunk_resized).permute(3, 0, 1, 2).contiguous()  # T,H,W,C → C,T,H,W
                # C, T, H, W = chunk_tensor.shape
                # chunk_tensor = chunk_tensor.permute(1, 0, 2, 3).reshape(C * T, H, W)  # C*T, H, W
                # chunk_tensor = F.resize(chunk_tensor, size=[360, 640])  # resize to (H=360, W=640)
                # chunk_tensor = chunk_tensor.reshape(T, C, 360, 640).permute(1, 0, 2, 3).contiguous()  # C,T,H,W

                ori_W, ori_H = chunk_tensor.shape[3], chunk_tensor.shape[2]

                # Load and normalize video
                video, new_W, new_H, padding = load_video_cthw_norm(chunk_tensor)
                # resize
                frames_np = video.permute(1, 2, 3, 0).numpy()  # C,T,H,W → T,H,W,C
                frames_upsampled = resize_canny_with_pyav(frames_np, target_size=(new_H*2, new_W*2))
                upsampled_video = torch.from_numpy(frames_upsampled).permute(3, 0, 1, 2).contiguous()  # T,H,W,C → C,T,H,W

                # Load and process canny
                canny = load_canny_cthw_norm(video, new_W, new_H, canny_cnt)
                video = (video / 255.0 - 0.5) / 0.5
                upsampled_video = (upsampled_video / 255.0 - 0.5) / 0.5

                # Prepare images
                first_img = to_pil_image(recover(video[:, 0].permute(1, 2, 0)).numpy().astype(np.uint8))
                last_img = to_pil_image(recover(video[:, -1].permute(1, 2, 0)).numpy().astype(np.uint8))

                # first_img = to_pil_image(recover(upsampled_video[:, 0].permute(1, 2, 0)).numpy().astype(np.uint8))
                # last_img = to_pil_image(recover(upsampled_video[:, -1].permute(1, 2, 0)).numpy().astype(np.uint8))


                if enable_key_frame:
                    first_img, last_img, frame_bytes = flf_process(keyframe_encoder, keyframe_decoder, first_img, last_img, qp_i=DEFAULT_CONFIG["qp_i"], qp_p=DEFAULT_CONFIG["qp_p"]) 
                    if first_img is None:
                        first_img = prev_img
                
                # Run inference
                result = pipe.recon(
                    input_video=video.unsqueeze(0),
                    cn_images=canny.unsqueeze(0),
                    prompt="",
                    negative_prompt="",
                    input_image=first_img,
                    last_image=last_img,
                    height=new_H*2,
                    width=new_W*2,
                    num_frames=DEFAULT_CONFIG['num_frames'],
                    cfg_scale=DEFAULT_CONFIG['cfg_scale'],
                    num_inference_steps=DEFAULT_CONFIG['num_inference_step'],
                    seed=42,
                    has_mask=True,
                )

                # Create visualization
                vis_images = []
                for k in range(len(result)):
                    vis_image = [
                        to_pil_image(recover(upsampled_video[:, k].permute(1, 2, 0)).numpy().astype(np.uint8)).crop((0, 0, 2*ori_W, 2*ori_H)),
                        result[k].crop((0, 0, 2*ori_W, 2*ori_H))
                    ]
                    vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
                    vis_images.append(vis_image)
                    os.makedirs( os.path.join(save_dir, save_name), exist_ok=True)
                    png_path = os.path.join(save_dir, save_name, f"clip{t:02d}_frame{k:02d}.png")
                    result[k].crop((0, 0, 2*ori_W, 2*ori_H)).save(png_path, compress_level=0)

                # Save result
                print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.mimsave(save_path, vis_images, fps=15)
                print(f"[Rank {rank}] Saved successfully: {save_path}")
                if enable_key_frame:
                    prev_img = last_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Rank {rank}] Error processing {test_name}: {str(e)}")


def run_inference_pipeline(
    video_list: List[str],
    gpu_ids: List[int] = None,
    canny_cnt=0,
    save_dir="",
    enable_key_frame=False,
):
    """Run complete inference pipeline with optimized processing"""
    # GPU configuration
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)

    # Prepare inference configurations
    inference_config_list = []
    for test_name in video_list:
        inference_config_list.append((test_name, DEFAULT_CONFIG))

    # Start batch processing (currently using single process for simplicity)
    print(f"\n{'='*40}\nStarting batch processing for all test cases\n{'='*40}")
    import torch.multiprocessing as mp
    from functools import partial


    # Run with single process
    inference_worker(rank=0, world_size=1, inference_configs=inference_config_list,
                   gpu_ids=gpu_ids, canny_cnt=canny_cnt, save_dir=save_dir, enable_key_frame=enable_key_frame)


# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """Main execution entry point for the video processing script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WanVideoI2V Video Processing with Canny Edges')
    parser.add_argument('--canny_cnt', type=int, default=100,
                        help='Number of frames for Canny edge processing threshold.')
    parser.add_argument('--enable_key_frame', action="store_true")

    args = parser.parse_args()

    # Run the inference pipeline
    run_inference_pipeline(
        video_list=get_video_list(),
        gpu_ids=GPU_IDS,
        canny_cnt=args.canny_cnt,
        save_dir=SAVEDIR,
        enable_key_frame=args.enable_key_frame,
    )
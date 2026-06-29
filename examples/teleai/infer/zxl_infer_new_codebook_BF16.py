import torch
import os, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
from PIL import Image, ImageDraw, ImageFont
from typing import List

USE_CODEBOOK = True
if USE_CODEBOOK:
    from pipelines import WanVideoI2VNewCodebookPipeline_BF16 as WanVideoI2VPipeline
    # from pipelines import WanVideoI2VNewCodebookPipeline_0128 as WanVideoI2VPipeline
    print("=====pipeline====: WanVideoI2VNewCodebookPipeline_BF16")
else:
    from pipelines import WanVideoI2VPipeline_V3 as WanVideoI2VPipeline
    print("=====pipeline====: WanVideoI2VPipeline_V3")

from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import numpy as np
import json
import imageio
import torchvision.transforms.functional as F
import cv2, re
from scipy.sparse import csr_matrix
import av
from moviepy import VideoFileClip, concatenate_videoclips
import struct
import gzip
import decord
from io import BytesIO
from keyframes.keyframe_encoder import encoder_keyframe_fun, init_encoder
from keyframes.keyframe_decoder import decoder_keyframe_fun, init_decoder

# 导入DCVC的自定义算术编码器模块
# import sys
# sys.path.insert(0, '/gemini/platform/shared/xujingyu/xujy/code/DCVC-main/DCVC-family/DCVC/src/cpp/build/rans')
# sys.path.insert(0, '/gemini/platform/shared/xujingyu/xujy/code/DCVC-main/DCVC-family/DCVC/src/cpp/build/ops')

# python daily_check_1_3b_v1.py
# os.environ["USE_HALF_TIME"] = "1"
# os.environ["USE_HALF_CHANNEL"] = "1"

kf_threshold = 0.001
qp = 42
# src = "06"
# src_fps = 25

# src = "19"
# src_fps = 30

# prefix = "/gemini/space/code/data/yifq1/yifq1/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_binarycanny_dynamicstride_tae"
folder = "/gemini/user/shared/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_binarycanny_dynamicstride_tae"
with open(os.path.join(folder, "latest_checkpointed_iteration.txt"), "r") as f:
    iter_num = f.read().strip()

CKPT_PATH = f"{folder}/iter_{int(iter_num):07d}/mp_rank_00/model_optim_rng.pt"
BASEDIR =  f"/gemini/user/shared/vis_result/20260317/jk_720p_oldmodel"


VIDEODIR = BASEDIR + "/video"
PICDIR = BASEDIR + "/pic"

# if USE_CODEBOOK:
#     SAVEDIR = f"/gemini/space/zxl/generated/1_3B/monitor_videos_0224/all_dyanmic_3steps"
# else:
#     SAVEDIR = f"/gemini/space/yifq/yanjq/train_gvc/new_monitor_videos/29_frames/MCL_JCV_720P/no_codebook/"



# 管理参数, 
# low: [26, 26, 6],  border: [32, 32, 6],  mid : [32, 32, 5],  high: [42, 42, 4]
DEFAULT_CONFIG = {
    "down_stride_ratio": 4,
    "qp_i":qp,
    "qp_p":qp,
    "num_frames": 29,
    "cfg_scale": 1,
    "num_inference_step": 10,
    "save_fps": 15,
    "tiled":False,
    "init_interval": None,
}

# NUM_OF_CLIPS = 1000
GPU_IDS = [0]
ENABLE_KEYFRAME_COMPRESSION = True


def get_video_list():
    root_dir = "/gemini/space/code/data/yifq1/yifq/public_benchmarks/MCL_JCV/720P/YUV_source"
    video_list = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if ".mp4" in file]
    return video_list



PIPELINE_CONFIG = dict(
    model_config=dict(
        dit=dict(
            path=CKPT_PATH, # ema_model.pt
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36+16+4, # t2v:16 i2v:36, s2v: 20 * numof_s
                dim=1536, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=8960, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12, # 1.3B:12 10B:40 14B:40
                num_layers=30, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=True,  # be true
                has_compressor={"use":True, "up_T":True, "pixel_shuffle": False, "enable_attn": True},
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

def recover(tensor):
    return (tensor * 0.5 + 0.5) * 255

def load_pipeline(
    device: torch.device,
) -> WanVideoI2VPipeline:
    pipe = WanVideoI2VPipeline(PIPELINE_CONFIG)
    return pipe


def add_text_and_return(image, text, position=(10, 10), text_size=10):
    """在图片上添加文字并返回 PIL Image"""
    # 注意：text() 操作是直接在原图上绘制，但我们会返回同一个对象
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=text_size)
    draw.text(position, text, fill="red", font=font)
    return image  # 返回修改后的原对象

def load_video_cthw_norm(video, info):
    frames = video  # T,H,W,C  # C,T,H,W
    fps = info['video_fps']
    frames, padding = pad(frames)
    new_H, new_W = frames.shape[2], frames.shape[3]
    print(f"frames.shape: {frames.shape}")
    return frames, new_W, new_H, fps, padding  # T,H,W,C → C,T,H,W


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


def f8tl_process_clip_one_frame(frame, quality=95):
    # 写入内存缓冲区而不是落盘
    first_buffer = BytesIO()
    frame.save(first_buffer, format='JPEG', quality=quality)
    frame_bytes = first_buffer.getvalue()

    return frame_bytes

def _encode_one_keyframe(keyframe_encoder, frame_bytes, qp_i, qp_p):
    # 对关键帧进行压缩
    frame_bytes_io = BytesIO(frame_bytes)
    encoder_data = encoder_keyframe_fun(keyframe_encoder, frame_bytes_io, None, qp_i=qp_i, qp_p=qp_p)
    return encoder_data

def flf_process_one_frame(keyframe_encoder, keyframe_decoder, frame, qp_i, qp_p):
    frame_bytes = f8tl_process_clip_one_frame(frame)
    frame_data = _encode_one_keyframe(keyframe_encoder, frame_bytes, qp_i, qp_p)
    frame_data = BytesIO(frame_data)
    frame_bytes = len(frame_data.getvalue())
    _, frame = decoder_keyframe_fun(keyframe_decoder, frame_data)
    frame = frame_handle(frame)
    return frame, frame_bytes

def pad(tensor):
    """
    Pad the tensor to make its height and width divisible by 64.
    Arguments:
        tensor (Tensor): Input tensor with shape (C, T, H, W).
    Returns:
        padded_tensor (Tensor): Padded tensor with shape (C, T, H', W').
        padding (tuple): A tuple containing the padding amounts for (top, bottom, left, right).
    """
    C, T, H, W = tensor.shape
    
    # Calculate padding needed for height and width
    pad_h = (64 - H % 64) % 64  # Padding for height
    pad_w = (64 - W % 64) % 64  # Padding for width
    
    # Padding will be applied as (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    
    # Apply padding
    padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
    
    return padded_tensor, padding


def unpad(padded_tensor, padding):
    """
    Unpad the tensor based on the padding parameters to restore original size.
    Arguments:
        padded_tensor (Tensor): Padded tensor with shape (C, T, H', W').
        padding (tuple): A tuple containing the padding amounts for (top, bottom, left, right).
    Returns:
        unpadded_tensor (Tensor): Unpadded tensor with shape (C, T, H, W).
    """
    _, _, H, W = padded_tensor.shape
    top, bottom, left, right = padding
    
    # Slice to remove padding
    unpadded_tensor = padded_tensor[:, :, top:H-bottom, left:W-right]
    
    return unpadded_tensor


def resize_canny_with_pyav(frames_array, target_size=(448, 832)):
    """
    对视频帧进行缩放操作
    
    Args:
        frames_array: numpy数组，形状为 (T, H, W, 3)
        new_height: 目标高度
        new_width: 目标宽度
    
    Returns:
        resized_frames: 缩放后的视频帧，形状为 (T, new_height, new_width, 3)
    """
    T, orig_height, orig_width, _ = frames_array.shape
    resized_frames = []
    for i in range(T):
        frame = av.VideoFrame.from_ndarray(frames_array[i], format='rgb24')
        resized_frame = frame.reformat(target_size[1], target_size[0], interpolation=av.video.reformatter.Interpolation.LANCZOS)
        resized_frame_array = resized_frame.to_ndarray(format='rgb24')
        resized_frames.append(resized_frame_array) 
    resized_frames = np.stack(resized_frames, axis=0)
    return resized_frames

def binary_array_from_rgb(canny_images):
    """
    对缩放后的视频帧进行重新二值化
    
    Args:
        canny_images: numpy数组，形状为 (T, new_height, new_width, 3)
    
    Returns:
        binary: 二值化后的canny数组，形状为 (T, new_height, new_width)
    """
    gray_images = np.mean(canny_images, axis=3)
    gray_uint8 = gray_images.astype(np.uint8)
    binary = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for gray in gray_uint8]
    binary = np.stack(binary).astype(bool)
    return binary

def encode_varint(value: int) -> bytes:
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

def customized_save_npz(path: str, matrix: csr_matrix, gzip_level: int = 9) -> bytes:
    """
    Compress a scipy.sparse.csr_matrix into bytes.
    Format (before gzip):
    [meta: rows:u32 cols:u32 nnz:u32]
    then for each row:
      [count: u32]
      if count>0:
        [first_index: u32]
        [varint(diff1), varint(diff2), ...]  # count-1 diffs
    """
    if not isinstance(matrix, csr_matrix):
        raise TypeError("matrix must be scipy.sparse.csr_matrix")

    rows, cols = matrix.shape
    nnz = matrix.nnz

    indices = matrix.indices
    indptr = matrix.indptr

    parts = []
    # meta
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
        # Guarantee row internal sort
        if not np.all(row_idxs[:-1] <= row_idxs[1:]):
            row_idxs = np.sort(row_idxs)

        # write first index as u32
        first = int(row_idxs[0])
        parts.append(struct.pack('<I', first))

        # write diffs for the rest using varint
        for val in row_idxs[1:]:
            diff = int(val) - first
            # IMPORTANT: diffs must be non-negative and increasing per-row
            # we encode successive diffs relative to previous value
            parts.append(encode_varint(diff))
            first = int(val)

    raw = b"".join(parts)
    compressed = gzip.compress(raw, compresslevel=gzip_level)
    if path is not None:
        with open(path, "wb") as f:
            f.write(compressed)
    return compressed

def decompress_csr(compressed_path: str) -> csr_matrix:
    """
    Decompress bytes produced by compress_csr back to csr_matrix.
    """
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

        # read first index (u32)
        if pos + 4 > len(raw):
            raise EOFError("Unexpected EOF while reading first index")
        (first_idx,) = struct.unpack_from('<I', raw, pos)
        pos += 4

        row_indices = [first_idx]
        prev = first_idx

        # remaining count-1 diffs (varint)
        for _ in range(count - 1):
            val, pos = decode_varint(raw, pos)
            # val is diff from prev, but encode_varint stored (actual_diff)
            # In our encoder diff was (curr - prev), so:
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

def get_percentile_thresholds(image, low_percentile=95, high_percentile=98):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    low_threshold = np.percentile(gradient_magnitude, low_percentile)
    high_threshold = np.percentile(gradient_magnitude, high_percentile)
    low_threshold = int(low_threshold)
    high_threshold = int(high_threshold)
    if low_threshold < 50:
        low_threshold = 50
        high_threshold = 150
    return low_threshold, high_threshold

def canny_process_one_frame(frame, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    data = cv2.Canny(gray, low_threshold, high_threshold) / 255  # 获取边缘图
    return data



import io
from dataclasses import dataclass
from typing import Generator, Iterable, Optional, List, Tuple
from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz


@dataclass
class CannyPacket:
    #frame_idx: int
    use_prev_ref: bool          # False: full frame, True: diff to prev
    #shape_hw: tuple             # (H', W')
    payload_npz: bytes          # save_npz 输出的 bytes（CSR）


def _csr_to_npz_bytes(mat: csr_matrix) -> bytes:
    """把 CSR 稀疏矩阵编码成 npz bytes（用于网络传输/写文件/队列）"""
    buf = io.BytesIO()
    save_npz(buf, mat, compressed=True)
    return buf.getvalue()


def _npz_bytes_to_csr(b: bytes) -> csr_matrix:
    """把 npz bytes 解回 CSR（接收端会用到）"""
    buf = io.BytesIO(b)
    return load_npz(buf)


def _bool_frame_to_ids(frame_bool_2d: np.ndarray) -> np.ndarray:
    """
    把一帧 bool 边缘图转为“扁平索引id”（升序且唯一）：
    id = y * W + x
    """
    # np.flatnonzero 返回升序的一维索引（对 setxor 很友好）
    return np.flatnonzero(frame_bool_2d)


def _ids_to_csr(ids: np.ndarray, H: int, W: int) -> csr_matrix:
    """把扁平 id 列表转回 CSR (H,W)"""
    if ids.size == 0:
        return csr_matrix((H, W), dtype=bool)
    rows = ids // W
    cols = ids % W
    data = np.ones_like(ids, dtype=bool)
    return coo_matrix((data, (rows, cols)), shape=(H, W), dtype=bool).tocsr()


def _npz_bytes_to_csr(b: bytes):
    buf = io.BytesIO(b)
    return load_npz(buf)

def _csr_to_sorted_ids(csr_mat, W: int) -> np.ndarray:
    rr, cc = csr_mat.nonzero()
    ids = (rr * W + cc).astype(np.int64)
    if ids.size:
        ids.sort()
    return ids

def _ids_to_rgb_frame(ids: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    ids: 扁平索引 (y*W + x)，升序
    return: np.uint8 RGB frame [H, W, 3]，边缘=255，背景=0
    """
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    if ids.size == 0:
        return frame
    ys = ids // W
    xs = ids % W
    frame[ys, xs, :] = 255
    return frame

lclf_prev_ids = None

def in_save_npz(path: str, matrix: csr_matrix, gzip_level: int = 9) -> bytes:
    """
    Compress a scipy.sparse.csr_matrix into bytes.
    Format (before gzip):
    [meta: rows:u32 cols:u32 nnz:u32]
    then for each row:
      [count: u32]
      if count>0:
        [first_index: u32]
        [varint(diff1), varint(diff2), ...]  # count-1 diffs
    """
    if not isinstance(matrix, csr_matrix):
        raise TypeError("matrix must be scipy.sparse.csr_matrix")

    rows, cols = matrix.shape
    nnz = matrix.nnz

    indices = matrix.indices
    indptr = matrix.indptr

    parts = []
    # meta
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
        # Guarantee row internal sort
        if not np.all(row_idxs[:-1] <= row_idxs[1:]):
            row_idxs = np.sort(row_idxs)

        # write first index as u32
        first = int(row_idxs[0])
        parts.append(struct.pack('<I', first))

        # write diffs for the rest using varint
        for val in row_idxs[1:]:
            diff = int(val) - first
            # IMPORTANT: diffs must be non-negative and increasing per-row
            # we encode successive diffs relative to previous value
            parts.append(encode_varint(diff))
            first = int(val)

    raw = b"".join(parts)
    compressed = gzip.compress(raw, compresslevel=gzip_level)
    if path is not None:
        with open(path, "wb") as f:
            f.write(compressed)
    return compressed

def load_canny_cthw_norm(frames, W, H, stride):

    global lclf_prev_ids  # C,T,H,W

    new_frames = frames.permute(1, 2, 3, 0).numpy()
    canny_bytes_num = 0
    canny_images = []
    gaussian_indexes = list(range(0, DEFAULT_CONFIG["num_frames"]))

    Hs, Ws = H // stride, W // stride
    prev_ids= lclf_prev_ids
    canny_frames = []
    out_csrs = []
    for frame_idx, frame in enumerate(new_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        data = canny_process_one_frame(frame, *get_percentile_thresholds(frame))
        canvas[data==1] = 255

        canny_image = resize_canny_with_pyav(frames_array=canvas[None, ...], target_size=(Hs, Ws))
        frame_bool = binary_array_from_rgb(canny_image)[0]  # (Hs, Ws)

        # ---- 4) 转成 ids（升序扁平索引）----
        curr_ids = _bool_frame_to_ids(frame_bool)

        # ---- 5) 决定本帧用 full 还是 diff ----
        force_keyframe = (prev_ids is None)
        use_prev_ref = False
        out_ids = curr_ids

        if (not force_keyframe) and (prev_ids is not None):
            # XOR diff：diff_ids = symmetric difference(prev_ids, curr_ids)
            diff_ids = np.setxor1d(prev_ids, curr_ids, assume_unique=True)

            # 用“更小的那个”发（直观省流量）
            # full 需要发 curr_ids；diff 需要发 diff_ids
            if diff_ids.size < curr_ids.size:
                use_prev_ref = True
                out_ids = diff_ids

        # ---- 6) ids -> CSR -> npz bytes ----
        out_csr = _ids_to_csr(out_ids, Hs, Ws)
        out_csrs.append(out_csr)
        payload = _csr_to_npz_bytes(out_csr)

        canny_frame = CannyPacket(
            use_prev_ref=use_prev_ref,
            payload_npz=payload)
        canny_frames.append(canny_frame)
        prev_ids = curr_ids

        if frame_idx == len(new_frames) - 1:
            new_lclf_prev_ids = curr_ids

    out_csrs = np.stack([m.toarray() for m in out_csrs], axis=0)
    new_csr = csr_matrix(out_csrs.reshape(out_csrs.shape[2], -1), dtype=bool)
    canny_bytes = in_save_npz(None, new_csr)
    group_frames = []
    #prev_ids = None
    prev_ids= lclf_prev_ids
    for pkt in canny_frames:
        csr_mat = _npz_bytes_to_csr(pkt.payload_npz).tocsr()
        ids = _csr_to_sorted_ids(csr_mat, Ws)

        # 2) 还原当前帧 ids
        if (not pkt.use_prev_ref) or (prev_ids is None):
            curr_ids = ids
        else:
            # XOR 还原：curr = prev XOR diff
            curr_ids = np.setxor1d(prev_ids, ids, assume_unique=True)

        # h w 3
        frame_rgb = _ids_to_rgb_frame(curr_ids, Hs, Ws)
        group_frames.append(frame_rgb)

        prev_ids = curr_ids

    canny_images = np.stack(group_frames, axis=0) #T h w 3
    canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(H, W))
    # torch: [T,H,W,3] -> [3,T,H,W]
    canny_images = torch.from_numpy(canny_images).permute(3, 0, 1, 2).contiguous()
    canny_images = (canny_images / 255.0 - 0.5) / 0.5
    lclf_prev_ids = new_lclf_prev_ids

    return canny_images, len(canny_bytes)


def concat_images(images, direction="horizontal", pad=0, pad_value=0):
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
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image

def concat_images_grid(images, cols, pad=0, pad_value=0):
    new_images = []
    while len(images) > 0:
        new_image = concat_images(images[:cols], pad=pad, pad_value=pad_value)
        new_images.append(new_image)
        images = images[cols:]
    new_image = concat_images(
        new_images, direction="vertical", pad=pad, pad_value=pad_value
    )
    return new_image


def concatenate_mp4_files(save_paths, output_path="output.mp4"):
    """
    拼接多个MP4视频文件
    
    参数：
        save_paths: MP4文件路径列表
        output_path: 输出文件路径
    """
    if not save_paths:
        print("错误：没有提供视频文件路径")
        return False
    
    try:
        print(f"开始拼接 {len(save_paths)} 个视频文件...")
        
        # 1. 加载所有视频片段
        clips = []
        for i, path in enumerate(save_paths, 1):
            try:
                clip = VideoFileClip(path)
                clips.append(clip)
            except Exception as e:
                print(f"  错误：无法加载文件 {path}: {e}")
                return False
        
        # 2. 拼接视频
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # 3. 输出视频
        final_clip.write_videofile(
            output_path,
            codec="libx264",  # H.264编码
            audio_codec="aac",  # AAC音频
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            fps=clip.fps  # 帧率，可以调整
        )
        
        # 4. 关闭所有剪辑以释放内存
        for clip in clips:
            clip.close()
        final_clip.close()
        

        return True
        
    except Exception as e:
        print(f"❌ 拼接失败: {e}")
        return False




def scan_poc_fix_ranges(video_path, fps=25, cover_sec=1.0):
    """
    扫描视频日志，找 'Could not find ref with POC'，并返回需要修复的连续帧区间列表：
      ranges = [(s,e), ...]  (闭区间)

    逻辑：
      1) ffmpeg+showinfo 扫一遍，遇到 POC error 记录当时最近的 showinfo 帧号 n
      2) 每个 n 扩成 [n, n+cover] 的修复窗
      3) 把所有窗合并成若干个连续区间
    """
    p = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", video_path, "-vf", "showinfo", "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )

    n_cur = None
    windows = []
    cover = int(round(cover_sec * fps))

    for line in p.stderr.splitlines():
        m = re.search(r"\bn:\s*(\d+)\b", line)
        if m:
            n_cur = int(m.group(1))
        if "Could not find ref with POC" in line and n_cur is not None:
            windows.append((n_cur, n_cur + cover))

    if not windows:
        return []

    # 合并重叠/相邻窗口 -> 连续区间
    windows.sort()
    ranges = []
    s, e = windows[0]
    for a, b in windows[1:]:
        if a <= e + 1:          # overlap 或者紧挨着
            e = max(e, b)
        else:
            ranges.append((s, e))
            s, e = a, b
    ranges.append((s, e))

    print("===fix_ranges===", ranges)
    return ranges


def interp_fix_chunk(chunk_frames, vr, start_idx, end_idx, fix_ranges, total_frames):
    # chunk_frames: [T,H,W,C] uint8, 覆盖全局帧 [start_idx, end_idx)
    for s, e in fix_ranges:
        L0 = max(s, start_idx)
        R0 = min(e, end_idx - 1)
        if L0 > R0:
            continue

        L = s - 1
        if L < 0:
            continue  # 开头就坏，没左端，先跳过（你需要的话也可改成用右端 freeze）
        # 左端参考帧（优先 chunk 内取）
        left = chunk_frames[L - start_idx] if start_idx <= L < end_idx else vr[L].asnumpy()

        R = e + 1
        if R >= total_frames:
            # 没右端：整段 freeze
            chunk_frames[L0 - start_idx:R0 - start_idx + 1] = left
            continue
        # 右端参考帧
        right = chunk_frames[R - start_idx] if start_idx <= R < end_idx else vr[R].asnumpy()

        denom = (R - L)  # >0
        for fi in range(L0, R0 + 1):
            w = fi - L
            out = (left.astype(np.uint16) * (denom - w) + right.astype(np.uint16) * w) // denom
            chunk_frames[fi - start_idx] = out.astype(np.uint8)


def inference_worker(
    rank: int,
    world_size: int,
    inference_configs,
    gpu_ids: List[int],
):
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    # 分配模型处理任务
    infer_task = inference_configs[rank::world_size]
    if not infer_task:
        print(f"[Rank {rank}] No task")
        return
    
    pipe = load_pipeline(device)

    for test_name, config in infer_task:

        # if need_fix(test_name):
        #     root, ext = os.path.splitext(test_name)
        #     fixed_path = f"{root}_fixed{ext}"
        #     if not os.path.exists(fixed_path):
        #         bps = get_avg_bitrate_bps(test_name)
        #         print(f"[fix] POC missing detected. Re-encode at ~{bps} bps -> {fixed_path}")
        #         re_encode_video(test_name, fixed_path, bps)
        #     test_name = fixed_path  # 后面 decord 继续用 video_path，其他逻辑不变

        

        # 使用decord（更高效的视频读取库）
        vr = decord.VideoReader(test_name, ctx=decord.cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        # fps = 25
        NUM_OF_CLIPS = total_frames // config["num_frames"]
        # NUM_OF_CLIPS = 1034

        fix_ranges = scan_poc_fix_ranges(test_name, fps=fps, cover_sec=1.0)  # 1秒=25帧

        recon_video_save_dir = os.path.join(VIDEODIR, os.path.splitext(os.path.basename(test_name))[0])
        recon_record_savepath = os.path.join(VIDEODIR, os.path.splitext(os.path.basename(test_name))[0] + ".json")
        recon_save_paths, vis_save_paths = [], []
        os.makedirs(recon_video_save_dir, exist_ok=True)

        PICDIR = os.path.join(BASEDIR, "pic", os.path.splitext(os.path.basename(test_name))[0])
        os.makedirs(PICDIR, exist_ok=True)
        
        record = {
            "i_num": 0,
            "i_frame": [],
            "mse": [],
            "canny_bytes":[],
            "latent_bytes":[],
            "key_frame_bytes":[],
            "total_bytes":0,
            "kbps":0,
            "canny_bpp": 0.0,
            "latent_bpp": 0.0,
            "keyframe_bpp": 0.0,
            "fps": fps
        }

        keyframe_encoder = init_encoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)
        keyframe_decoder = init_decoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)

        prev_img = None
        for t in range(NUM_OF_CLIPS):
            try:
                # if t % DEFAULT_CONFIG["init_interval"] == 0 and ENABLE_KEYFRAME_COMPRESSION:
                #     keyframe_encoder = init_encoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"])
                #     keyframe_decoder = init_decoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"])
                with torch.no_grad():
                    start_idx = t * config["num_frames"]
                    end_idx = min((t + 1) * config["num_frames"], total_frames)
                    
                    # 读取chunk
                    chunk_frames = vr[start_idx:end_idx].asnumpy()  # [T, H, W, C]
                    interp_fix_chunk(chunk_frames, vr, start_idx, end_idx, fix_ranges, total_frames=len(vr))
                    

                    chunk_tensor = torch.from_numpy(chunk_frames) 
                    chunk_tensor = chunk_tensor.permute(3, 0, 1, 2).contiguous()  # C,T,H,W
                    ori_W, ori_H = chunk_tensor.shape[3], chunk_tensor.shape[2]
                    print(chunk_tensor.shape)
                    info = {
                        'video_fps': fps,
                    }
                    # save_dir = Path(SAVEDIR)
                    # save_dir.mkdir(parents=True, exist_ok=True)

                    save_name = os.path.splitext(os.path.basename(test_name[0]))[0]
                    save_path = os.path.join(recon_video_save_dir, f"{save_name}_p{t}.mp4")

                    # if os.path.exists(save_path):
                    #     print(f"[Rank {rank}] 文件已存在，跳过: {save_path}")
                    #     continue
                    video, new_W, new_H, _, padding = load_video_cthw_norm(chunk_tensor, info)
                    canny, canny_bytes = load_canny_cthw_norm(video, new_W, new_H, int((new_H/480) * config["down_stride_ratio"]))
                    # canny_bytes = 0
                    video = (video / 255.0 - 0.5) / 0.5
                    first_img = to_pil_image(recover(video[:, 0].permute(1,2,0)).numpy().astype(np.uint8))
                    last_img_ori = to_pil_image(recover(video[:, -1].permute(1,2,0)).numpy().astype(np.uint8))

                    
                    if ENABLE_KEYFRAME_COMPRESSION:
                        first_img, last_img, frame_bytes = flf_process(keyframe_encoder, keyframe_decoder, first_img, last_img_ori, qp_i=config["qp_i"], qp_p=config["qp_p"]) 
                        if first_img is None:
                            first_img = prev_img
                        x = np.asarray(last_img_ori).astype(np.float32) / 255.0
                        y = np.asarray(last_img).astype(np.float32) / 255.0
                        mse = np.mean((x - y) ** 2)
                        print("mse =", mse)
                        record["mse"].append(float(mse))
                        if mse > kf_threshold:
                            del keyframe_encoder
                            del keyframe_decoder
                            keyframe_encoder = init_encoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"])
                            keyframe_decoder = init_decoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"])
                            last_img, frame_bytes = flf_process_one_frame(keyframe_encoder, keyframe_decoder, last_img_ori, qp_i=config["qp_i"], qp_p=config["qp_p"])
                            record["i_num"] += 1
                            record["i_frame"].append(t)
                    else:
                        last_img = last_img_ori
                    
                    # 执行推理
                    result, latent_byte_num = pipe.recon(
                        input_video=video.unsqueeze(0),
                        cn_images=canny.unsqueeze(0), 
                        prompt="",
                        negative_prompt="",
                        input_image=first_img,
                        last_image=last_img,
                        height=new_H,
                        width=new_W,
                        num_frames=config['num_frames'],
                        cfg_scale=config['cfg_scale'],
                        num_inference_steps=config['num_inference_step'],
                        seed=42,
                        has_mask=True,
                        add_cn_noise=False,
                        return_compressed=True,
                    )

                    record["canny_bytes"].append(canny_bytes)
                    record["latent_bytes"].append(latent_byte_num) # 
                    record["key_frame_bytes"].append(frame_bytes)
                    gop_bytes= record["canny_bytes"][-1] + record["latent_bytes"][-1] + record["key_frame_bytes"][-1]
                    gop_kbps = (gop_bytes * 8) / (config["num_frames"] / fps) / 1024
                    gop_bpp = (gop_bytes * 8) / config["num_frames"] / ori_H / ori_W


                    # ========== 实时保存JSON（每个GOP处理完就刷新） ==========
                    # 更新统计信息
                    current_total_frames = (t + 1) * config["num_frames"]  # 已处理的总帧数
                    record["total_bytes"] = (sum(record["canny_bytes"]) + 
                                            sum(record["latent_bytes"]) + 
                                            sum(record["key_frame_bytes"]))
                    record["kbps"] = record["total_bytes"] * 8 / (current_total_frames / fps) / 1024
                    record["bpp"] = (record["total_bytes"] * 8) / current_total_frames / ori_H / ori_W  # 新增
                    record["keyframe_bpp"] = (sum(record["key_frame_bytes"]) * 8) / current_total_frames / ori_H / ori_W  # 新增
                    record["latent_bpp"] = (sum(record["latent_bytes"]) * 8) / current_total_frames / ori_H / ori_W  # 新增
                    record["canny_bpp"] = (sum(record["canny_bytes"]) * 8) / current_total_frames / ori_H / ori_W  # 新增
                    
                    # 实时保存，格式化输出
                    # 保存JSON - 数组在一行
                    json_str = json.dumps(record, indent=2, ensure_ascii=False)
                    json_str = re.sub(r'\[\s+(\d)', r'[\1', json_str)  # [ 数字
                    json_str = re.sub(r'(\d)\s+\]', r'\1]', json_str)  # 数字 ]
                    json_str = re.sub(r'(\d),\s+(\d)', r'\1, \2', json_str)  # 数字, 数字
                    
                    with open(recon_record_savepath, 'w') as f:
                        f.write(json_str)



                    vis_images = []
                    recon_images = []
                    for k in range(len(result)):
                        origin_image = to_pil_image(recover(video[:, k].permute(1,2,0)).numpy().astype(np.uint8)).crop((0, 0, ori_W, ori_H))
                        vis_image = [origin_image, \
                                    result[k].crop((0, 0, ori_W, ori_H))]
                        recon_image = result[k].crop((0, 0, ori_W, ori_H))
                        # recon_image = add_text_and_return(recon_image, 
                        #                             "Bitrate: {} kbps".format(int(gop_kbps)))
                        # recon_image = add_text_and_return(recon_image, 
                        #                                 "Bitrate: {} kbps".format(int(gop_kbps)),
                        #                                 (ori_W // 2 - ori_W // 8, ori_H//20), text_size=ori_H//15)
                        
                        # 无损保存PNG
                        png_path = os.path.join(PICDIR, f"clip{t:02d}_frame{k:02d}.png")
                        recon_image.save(png_path, compress_level=0)

                        # png_path = os.path.join(PICDIR, f"clip{t:02d}_frame{k:02d}.png")
                        # origin_image.save(png_path, compress_level=0)

                        recon_images.append(recon_image)

                        vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
                        # # draw record
                        # vis_image = add_text_and_return(vis_image, 
                        #                                 "kbps:{:.2f}, bpp:{:.5f}".format(gop_kbps, gop_bpp),
                        #                                 (ori_W // 2 + ori_W, ori_H//10), text_size=ori_H//40)
                        
                        # draw record, 保留整数
                        # vis_image = add_text_and_return(vis_image, 
                        #                                 "Bitrate: {} kbps".format(int(gop_kbps)),
                        #                                 (ori_W // 2 + ori_W - ori_W // 8, ori_H//20), text_size=ori_H//15)

                        vis_images.append(vis_image)
                        

                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # imageio.mimsave(save_path, vis_images, fps=fps)  
                    # imageio.mimsave(save_path.replace(".mp4", "_recon.mp4"), recon_images, fps=fps) 

                    writer = imageio.get_writer(
                        save_path,
                        fps=fps,
                        macro_block_size=1,   # 关键：禁止自动 pad 到 16 倍数
                    )
                    for im in vis_images:
                        writer.append_data(np.asarray(im))  # PIL -> numpy (H,W,3)
                    writer.close()

                    writer = imageio.get_writer(
                        save_path.replace(".mp4", "_recon.mp4"),
                        fps=fps,
                        macro_block_size=1,   # 关键：禁止自动 pad 到 16 倍数
                    )
                    for im in recon_images:
                        writer.append_data(np.asarray(im))  # PIL -> numpy (H,W,3)
                    writer.close()
    
                    print(f"[Rank {rank}] 保存成功: {save_path}")
                    vis_save_paths.append(save_path)
                    recon_save_paths.append(save_path.replace(".mp4", "_recon.mp4"))
                    prev_img = last_img
                

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {str(e)}")

        # record["total_bytes"] = sum(record["canny_bytes"]) +  sum(record["latent_bytes"]) + sum(record["key_frame_bytes"])
        # record["kbps"] = record["total_bytes"] * 8 / (NUM_OF_CLIPS * config["num_frames"] / fps) / 1024
        # with open(recon_record_savepath, "w") as f:
        #     json.dump(record, f)
                
        total_frames = NUM_OF_CLIPS * config["num_frames"]
        record["total_bytes"] = sum(record["canny_bytes"]) +  sum(record["latent_bytes"]) + sum(record["key_frame_bytes"])
        record["kbps"] = record["total_bytes"] * 8 / (total_frames / fps) / 1024
        record["bpp"] = (record["total_bytes"] * 8) / total_frames / ori_H / ori_W  # 新增
        record["keyframe_bpp"] = (sum(record["key_frame_bytes"]) * 8) / total_frames / ori_H / ori_W  # 新增
        record["latent_bpp"] = (sum(record["latent_bytes"]) * 8) / total_frames / ori_H / ori_W  # 新增
        record["canny_bpp"] = (sum(record["canny_bytes"]) * 8) / total_frames / ori_H / ori_W  # 新增
                
        # 实时保存，格式化输出
        # 保存JSON - 数组在一行
        json_str = json.dumps(record, indent=2, ensure_ascii=False)
        json_str = re.sub(r'\[\s+(\d)', r'[\1', json_str)  # [ 数字
        json_str = re.sub(r'(\d)\s+\]', r'\1]', json_str)  # 数字 ]
        json_str = re.sub(r'(\d),\s+(\d)', r'\1, \2', json_str)  # 数字, 数字
        
        with open(recon_record_savepath, 'w') as f:
            f.write(json_str)

        # concatenate_mp4_files(vis_save_paths, os.path.join(VIDEODIR, os.path.splitext(os.path.basename(test_name))[0] + "_vis.mp4"))
        # concatenate_mp4_files(recon_save_paths, os.path.join(VIDEODIR, os.path.splitext(os.path.basename(test_name))[0] + "_recon.mp4"))

    

def run_inference_pipeline(
    video_list: List[str],
    gpu_ids: List[int] = None,
):
    """运行完整推理流程（优化后版本）"""
    # GPU配置
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)

    # 准备所有推理配置
    inference_config_list = []
    for test_name in video_list:
        inference_config_list.append((test_name, DEFAULT_CONFIG))

    # 启动单次多进程推理
    print(f"\n{'='*40}\n开始批量处理所有测试用例\n{'='*40}")
    import torch.multiprocessing as mp
    from functools import partial

    inference_worker(rank=0, world_size=1, inference_configs=inference_config_list, gpu_ids=gpu_ids)

    # mp.spawn(
    #     partial(
    #         inference_worker,
    #         world_size=world_size,
    #         inference_configs=inference_config_list,
    #         gpu_ids=gpu_ids,
    #     ),
    #     nprocs=world_size,
    #     join=True,
    # )


if __name__ == "__main__":
    run_inference_pipeline(
            video_list=get_video_list(),
            gpu_ids=GPU_IDS,
        )


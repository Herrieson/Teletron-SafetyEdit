import torch
import os
import sys
import decord
from PIL import Image
from typing import Dict, List, Tuple
from pipelines import WanVideoI2VPipeline
from diffusers.utils import export_to_video
from keyframes.keyframe_encoder import encoder_keyframe_fun, init_encoder
from keyframes.keyframe_decoder import decoder_keyframe_fun, init_decoder
from io import BytesIO
import copy
from pathlib import Path
# import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np
import json
import imageio
from scipy.sparse import load_npz
from torchvision.io import read_video
import torchvision.transforms.functional as F
import cv2
from scipy.special import comb
from scipy.sparse import csr_matrix
import av
import math
import struct
import gzip

# python daily_check_1_3b_v1.py


folder = "/gemini/user/shared/workdirs/fl2v_1.3B_recon_multi_resolution_f29_sft_binarycanny_dynamicstride_tae"
with open(os.path.join(folder, "latest_checkpointed_iteration.txt"), "r") as f:
    iter_num = f.read().strip()

CKPT_PATH = f"{folder}/iter_{int(iter_num):07d}/mp_rank_00/model_optim_rng.pt"
# CKPT_PATH = "/gemini/space/yifq/yanjq/train_gvc/workdirs/train_29frames_pd4-2_1e-7/iter_0001250/mp_rank_00/model_optim_rng.pt"
SAVEDIR =  f"/gemini/user/shared/vis_result/20260317/jk_720p_oldmodel"


# 管理参数
DEFAULT_CONFIG = {
    "num_frames": 29,
    "cfg_scale": 1,
    "num_inference_step": 10,
    "save_fps": 25,
    "tiled":False,
    "qp_i": 42,
    "qp_p": 42,
}
NUM_OF_CLIPS = 50
GPU_IDS = [0]

# def get_video_list():
#     video_list = []
#     video_list.append(["/gemini/user/shared/yifq/test_data/348153587-1-208.mp4", "/gemini/user/shared/yifq/test_data/348153587-1-208.npz"])
#     return video_list

# def get_video_list():
#     video_list = []
#     video_list.append(['/gemini/user/shared/yifq/scripts/jiankong_demo.mp4', '/gemini/platform/shared/xujingyu/xujy/data/test/BV1PK411N75n_p63_canny_higher_threshold.npz'])
#     return video_list

def get_video_list():
    return [["/gemini/user/shared/20260311153740276.mp4", "/gemini/space/zxl/dataset/sr_videos/day_mixedtraffic_360p_384k_fps30_gop60.mp4"]]

# def get_video_list():
#     root_dir = "/gemini/platform/shared/yifq1/yifq/public_benchmarks/MCL_JCV/1080P/YUV_source"
#     video_list = [[os.path.join(root_dir, file), os.path.join(root_dir, file)] for file in os.listdir(root_dir) if ".mp4" in file]
#     print(video_list)
#     return video_list


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

def load_pipeline(
    device: torch.device,
) -> WanVideoI2VPipeline:
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
    在canny数组中对值为1的元素随机置零
    
    参数:
        canny_array: 形状为(T, height, width)的数组, 只包含0和1
        dropout_rate: 要置零的比例(0到1之间)
        
    返回:
        处理后的数组
    """
    result = canny_array.copy()
    ones_mask = (result == 1)
    random_values = np.random.random(result.shape)
    to_drop = (ones_mask & (random_values < dropout_rate))
    result[to_drop] = 0
    return result

def resize_canny_direct(canny_array, target_size=(448, 832)):
    """
    直接调整canny数组大小
    Args:
        canny_array: (t, h, w) 的numpy数组
        target_size: 目标大小 (height, width)
    Returns:
        resized_canny: (t, target_h, target_w) 的numpy数组
    """
    canny_tensor = torch.from_numpy(canny_array).float()
    canny_tensor = canny_tensor.unsqueeze(1)
    # 使用双线性插值调整大小
    resized = torch.nn.functional.interpolate(canny_tensor, size=target_size, mode='nearest')
    resized_canny = resized.squeeze(1).numpy()
    # 重新二值化
    resized_canny = (resized_canny > 0.4).astype(np.uint8)
    
    return resized_canny

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
    padding = (0, 0, pad_w, pad_h)  # (left, right, top, bottom)
    
    # Apply padding
    padded_tensor = F.pad(tensor, padding, padding_mode='constant')
    
    return padded_tensor, padding



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

def private_save_npz(path: str, matrix: csr_matrix, gzip_level: int = 9) -> bytes:
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
    data = cv2.Canny(gray, low_threshold, high_threshold) / 255  # 获取边缘图
    return data


def load_canny_cthw_norm(frames, W, H):
    new_frames = frames.permute(1, 2, 3, 0).numpy()
    canny_images = []
    gaussian_indexes = list(range(0, len(new_frames)))
    for i, idx in enumerate(range(len(new_frames))):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        if i % len(new_frames) in gaussian_indexes:
            data = canny_process_one_frame(new_frames[idx], *get_percentile_thresholds(new_frames[idx]))
            canvas[data==1] = 255
        canny_images.append(canvas)
    canny_images = np.stack(canny_images, axis=0)

    downsample_stride = round(H / 480 * 4)
    print(downsample_stride)
    canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(H//downsample_stride, W//downsample_stride))

    binary_array = binary_array_from_rgb(canny_images)
    canny_images = np.stack([np.stack([binary.astype(np.uint8)*255]*3, axis=-1) for binary in binary_array], axis=0)

    # 再上采样回去
    canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(H, W))
    canny_images = torch.from_numpy(canny_images).permute(3,0,1,2).contiguous()  # T,H,W,C
    canny_images = (canny_images / 255.0 - 0.5) / 0.5
    print(f"loaded canny shape: {canny_images.shape}")
    return canny_images


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

def inference_worker(
    rank: int,
    world_size: int,
    inference_configs,
    gpu_ids: List[int],
    enable_key_frame=False,
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

    if enable_key_frame:
        keyframe_encoder = init_encoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)
        keyframe_decoder = init_decoder(model_path_i=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_i"], model_path_p=PIPELINE_CONFIG["model_config"]["keyframe"]["model_path_p"], device=device)

    for test_name, config in infer_task:
        vr = decord.VideoReader(test_name[0], ctx=decord.cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        NUM_OF_CLIPS = total_frames // DEFAULT_CONFIG["num_frames"]
        save_dir = Path(SAVEDIR)
        save_dir.mkdir(parents=True, exist_ok=True)


        for t in range(0, NUM_OF_CLIPS):
            try:
                save_name = os.path.splitext(os.path.basename(test_name[0]))[0]
                save_path = os.path.join(save_dir, f"{save_name}_p{t}.mp4")

                if os.path.exists(save_path):
                    print(f"[Rank {rank}] 文件已存在，跳过: {save_path}")
                    continue
                
                chunk_frames = vr[t*DEFAULT_CONFIG["num_frames"]:(t+1)*DEFAULT_CONFIG["num_frames"]].asnumpy()  # [T, H, W, C]
                chunk_tensor = torch.from_numpy(chunk_frames) 
                chunk_tensor = chunk_tensor.permute(3, 0, 1, 2).contiguous()  # C,T,H,W

                # Resize to 720x1280 (height=720, width=1280)
                chunk_frames_np = chunk_tensor.permute(1, 2, 3, 0).numpy()  # C,T,H,W → T,H,W,C
                chunk_resized = resize_canny_with_pyav(chunk_frames_np, target_size=(720, 1280))
                chunk_tensor = torch.from_numpy(chunk_resized).permute(3, 0, 1, 2).contiguous()  # T,H,W,C → C,T,H,W

                ori_W, ori_H = chunk_tensor.shape[3], chunk_tensor.shape[2]
                print("ori_W, ori_H: ", ori_W, ori_H)

                video, new_W, new_H, padding = load_video_cthw_norm(chunk_tensor)
                canny = load_canny_cthw_norm(video, new_W, new_H)
                video = (video / 255.0 - 0.5) / 0.5

                first_img = to_pil_image(recover(video[:, 0].permute(1,2,0)).numpy().astype(np.uint8))
                last_img = to_pil_image(recover(video[:, -1].permute(1,2,0)).numpy().astype(np.uint8))

                if enable_key_frame:
                    first_img, last_img, frame_bytes = flf_process(keyframe_encoder, keyframe_decoder, first_img, last_img, qp_i=DEFAULT_CONFIG["qp_i"], qp_p=DEFAULT_CONFIG["qp_p"])
                    if first_img is None:
                        first_img = prev_img

                # 执行推理
                result = pipe.recon(
                    input_video=video.unsqueeze(0),
                    cn_images=canny.unsqueeze(0), 
                    prompt="",
                    negative_prompt="",
                    input_image=first_img,
                    last_image=last_img,
                    height=new_H,
                    width=new_W,
                    num_frames=DEFAULT_CONFIG['num_frames'],
                    cfg_scale=DEFAULT_CONFIG['cfg_scale'],
                    num_inference_steps=DEFAULT_CONFIG['num_inference_step'],
                    seed=42,
                    has_mask=True,
                )

                vis_images = []
                for k in range(len(result)):
                    vis_image = [to_pil_image(recover(video[:, k].permute(1,2,0)).numpy().astype(np.uint8)).crop((0, 0, ori_W, ori_H)), \
                                result[k].crop((0, 0, ori_W, ori_H))]
                    vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
                    vis_images.append(vis_image)

                    os.makedirs( os.path.join(save_dir, save_name), exist_ok=True)
                    png_path = os.path.join(save_dir, save_name, f"clip{t:02d}_frame{k:02d}.png")
                    result[k].crop((0, 0, ori_W, ori_H)).save(png_path, compress_level=0)
                print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.mimsave(save_path, vis_images, fps=15)
                print(f"[Rank {rank}] 保存成功: {save_path}")
                if enable_key_frame:
                    prev_img = last_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {str(e)}")
    

def run_inference_pipeline(
    video_list: List[str],
    gpu_ids: List[int] = None,
    enable_key_frame=False,
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

    inference_worker(rank=0, world_size=1, inference_configs=inference_config_list, gpu_ids=gpu_ids, enable_key_frame=enable_key_frame)

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
    import argparse
    parser = argparse.ArgumentParser(description='WanVideoI2V Video Processing')
    parser.add_argument('--enable_key_frame', action="store_true")
    args = parser.parse_args()

    run_inference_pipeline(
            video_list=get_video_list(),
            gpu_ids=GPU_IDS,
            enable_key_frame=args.enable_key_frame,
        )
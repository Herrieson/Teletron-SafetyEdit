import torch
import os
import sys
sys.path.insert(0, "/gemini/space/yanjq/Teletron/Teletron_1_3b_v1/")
from PIL import Image
from typing import Dict, List, Tuple
from pipelines import WanVideoI2VWoMaskPipeline
from diffusers.utils import export_to_video
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


CKPT_PATH = "/gemini/space/yifq/teletron-model/Wan2.1-FLF2V-14B-720P/release/mp_rank_00/model_optim_rng.pt"
SAVEDIR = f"/gemini/space/yanjq/VAST/results_testdata_reproduce_vast/"


# 管理参数
DEFAULT_CONFIG = {
    "height": 448,
    "width": 832,
    "num_frames": 45,
    "cfg_scale": 1,
    "num_inference_step": 10,
    "save_fps": 15,
    "tiled":False,
}
NUM_OF_CLIPS = 70
GPU_IDS = [5]

# def get_video_list(num=20):
#     video_list = []
#     data = json.load(open("/gemini/space/yifq/yifq/code/scripts/jsons/crawl_0923.json"))
#     for clip in data["clips"]:
#         if clip["video_length"] >= DEFAULT_CONFIG["num_frames"]:
#             video_list.append([clip["video_path"], clip['canny_path']])

#         if len(video_list) >= num:
#             break
#     return video_list

def get_video_list():
    video_list = []
    video_list.append(["/gemini/space/yifq/xjy/BV1PK411N75n_p63.mp4", "/gemini/space/yifq/xjy/temp_data/BV1PK411N75n_p63_canny.npz"])
    return video_list

PIPELINE_CONFIG = dict(
    model_config=dict(
        dit=dict(
            path=CKPT_PATH, # ema_model.pt
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36+16, # t2v:16 i2v:36, s2v: 16 * numof_s
                dim=5120, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=13824, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=40, # 1.3B:12 10B:40 14B:40
                num_layers=40, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=True,  # be true
                has_compressor={"use":True, "up_T":True},
                has_quantizer=False,
            ),
        ),
        encoder=dict(
            vae=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=True,
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
            ),
            text_encoder=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
        ),
    ),
    torch_dtype=torch.bfloat16,
    device="cuda",
)


def recover(tensor):
    return (tensor * 0.5 + 0.5) * 255

def load_pipeline(
    device: torch.device,
) -> WanVideoI2VWoMaskPipeline:
    pipe = WanVideoI2VWoMaskPipeline(PIPELINE_CONFIG)
    return pipe


def load_video_cthw_norm(path, t, W=DEFAULT_CONFIG["width"], H=DEFAULT_CONFIG["height"], T=DEFAULT_CONFIG["num_frames"]):
    video, _, info = read_video(path, pts_unit='sec')
    frames = video[t*T:(t+1)*T].permute(3, 0, 1, 2)  # T,H,W,C
    width, height = video.shape[2], video.shape[1]
    fps = info['video_fps']
    frames = F.resize(frames, size=[H, W])
    print(f"frames.shape: {frames.shape}")
    frames = (frames / 255.0 - 0.5) / 0.5
    return frames, width, height, fps  # T,H,W,C → C,T,H,W


def fitted_curves(img, n, order=3):
    """
    参数:
        img: 输入二值图像
        n: 处理的轮廓数量
        order: 贝塞尔曲线阶数（默认8阶）
    """
    def bezier_interp_nth(points, order, num=50):
        """
        n阶贝塞尔曲线插值（通用版本）
        """
        t = np.linspace(0, 1, num).reshape(-1, 1)
        curve = np.zeros((num, 2))
        
        for i in range(order + 1):
            # 伯努斯坦基函数
            basis = comb(order, i) * ((1 - t) ** (order - i)) * (t ** i)
            curve += basis * points[i]
        return curve
    
    num_control_points = order + 1
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_points_needed = n * num_control_points
    output = np.zeros_like(img)
    
    if len(contours) > n:
        contour_areas = [(cv2.contourArea(c), c) for c in contours]
        contour_areas.sort(key=lambda x: x[0], reverse=True)
        contours = [c for _, c in contour_areas[:n]]
    
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < num_control_points:
            continue
        
        idx = np.linspace(0, len(cnt) - 1, total_points_needed, dtype=int)
        cnt = cnt[idx]
        
        step = num_control_points - 1
        for i in range(0, len(cnt) - step, step):
            control_points = cnt[i:i+num_control_points]
            curve = bezier_interp_nth(control_points, order, num=10)
            curve = np.round(curve).astype(int)
            
            for j in range(len(curve) - 1):
                pt1 = tuple(curve[j])
                pt2 = tuple(curve[j + 1])
                cv2.line(output, pt1, pt2, 255, 1)
    output = np.stack([output]*3, axis=-1)
    return output

def load_canny_cthw_norm(path, ori_W, ori_H, t, fps, W=DEFAULT_CONFIG["width"], H=DEFAULT_CONFIG["height"], T=DEFAULT_CONFIG["num_frames"]):
    data = load_npz(path).toarray().reshape((-1, ori_H, ori_W))
    sample_interval = max(1, round(fps / 5))
    canny_images = []
    for i, idx in enumerate(range(t*T, (t+1)*T)):
        canvas = np.zeros(shape=(ori_H, ori_W, 3), dtype=np.uint8)
        # if i % sample_interval != 0:
        #     canny_images.append(canvas)
        #     continue
        canvas = fitted_curves(data[idx].astype(np.uint8), n=145, order=1)
        # canvas[data[idx]==1] = 255
        canny_images.append(canvas)

    canny_images = np.stack(canny_images, axis=0)
    canny_images = torch.from_numpy(canny_images).permute(3,0,1,2).contiguous()  # T,H,W,C
    canny_images = F.resize(canny_images, size=[H, W])
    # canny_images = (
    #     torch.from_numpy(canny_images).permute(3, 0, 1, 2).contiguous()
    # )
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
        for t in range(8, 70):
            try:
                save_dir = Path(SAVEDIR)
                save_dir.mkdir(parents=True, exist_ok=True)

                save_name = os.path.splitext(os.path.basename(test_name[0]))[0]
                save_path = os.path.join(save_dir, f"{save_name}_p{t}.mp4")

                if os.path.exists(save_path):
                    print(f"[Rank {rank}] 文件已存在，跳过: {save_path}")
                    continue
                else:
                    print(f"[Rank {rank}] 正在生成: {save_path}")

                video, ori_W, ori_H, fps = load_video_cthw_norm(test_name[0], t)
                canny = load_canny_cthw_norm(test_name[1], ori_W, ori_H, t, fps)
                first_img = to_pil_image(recover(video[:, 0].permute(1,2,0)).numpy().astype(np.uint8))
                last_img = to_pil_image(recover(video[:, -1].permute(1,2,0)).numpy().astype(np.uint8))

                # 执行推理
                result = pipe.recon(
                    input_video=video.unsqueeze(0),
                    cn_images=canny.unsqueeze(0), 
                    prompt="",
                    negative_prompt="",
                    input_image=first_img,
                    last_image=last_img,
                    height=config["height"],
                    width=config["width"],
                    num_frames=config['num_frames'],
                    cfg_scale=config['cfg_scale'],
                    num_inference_steps=config['num_inference_step'],
                    seed=42,
                    add_cn_noise=True,
                )

                vis_images = []
                for k in range(len(result)):
                    vis_image = [to_pil_image(recover(video[:, k].permute(1,2,0)).numpy().astype(np.uint8)), \
                                result[k]]
                    vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
                    vis_images.append(vis_image)
                print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                imageio.mimsave(save_path, vis_images, fps=15)  
                print(f"[Rank {rank}] 保存成功: {save_path}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Rank {rank}] 处理任务 {test_name} 时出错: {str(e)}")
    

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

    #inference_worker(rank=0, world_size=world_size, inference_configs=inference_config_list, gpu_ids=gpu_ids)

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
    # try:
    #     run_inference_pipeline(
    #         video_list=get_video_list(),
    #         gpu_ids=GPU_IDS,
    #     )
    # except Exception as e:
    #     print(f"主程序运行出错: {str(e)}")
    run_inference_pipeline(
            video_list=get_video_list(),
            gpu_ids=GPU_IDS,
        )

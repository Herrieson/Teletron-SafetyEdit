import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import torchvision.io as io
import json
import random
import numpy as np
import os
# ps: pixel value should be in [0, 1]!
import re

all_mean = 0.0
num = 0


files = ["/gemini/space/yanjq/Teletron/gan_outputs_1112/infer_200iters_77frames/BV1PK411N75n_p63_p44.mp4"]

files = ["/gemini/space/yanjq/Teletron/gt_mse_dit_outputs/infer_700iters_77frames/BV1PK411N75n_p63_p44.mp4"]


for file in files:
    if not file.endswith(".mp4"):
        continue
    original_video_path = file
    original_videos = io.read_video(original_video_path)
    
    videos1 = original_videos[0][: ,: ,:832, :] / 255
    videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
    start_col = 1680 - 832
    videos2 = original_videos[0][: ,: ,start_col:, :]  / 255
    videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)
    device = torch.device("cuda")
    result = {}
    only_final = False
    result['fvd'] = calculate_fvd(videos1, videos2, device)
    mean_value = np.mean(result['fvd']['value'])
    all_mean += mean_value
    num += 1
    print(f"{original_video_path} fvd: {mean_value}")
all_mean /= num
print(f"fvd: {all_mean}")

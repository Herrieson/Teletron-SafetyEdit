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

#files = ["/gemini/space/yanjq/Teletron/train_pd8-4_outputs/1106_infer_4steps_500_77frames_weight100/BV1PK411N75n_p63_p46.mp4"]
# 16.30 16.01 46: 16.72
files = ["/gemini/space/yanjq/VAST/results_testdata_reproduce_vast_1_3B/BV1PK411N75n_p63_p46.mp4"]
# 16.24 15.97 46:16.59
#files = ["/gemini/space/yanjq/Teletron/baseline_outputs/infer_4steps_77frames/BV1PK411N75n_p63_p46.mp4"]
# 16.31 16.03 46: 16.66
#files = ["/gemini/space/yanjq/Teletron/train_pd8-4_outputs/1107_infer_50_4steps_77frames/BV1PK411N75n_p63_p43.mp4"]
#16.30
files = ["/gemini/space/yanjq/Teletron/baseline_outputs/infer_8steps_77frames/BV1PK411N75n_p63_p43.mp4"]
#16.24
files = ["/gemini/space/yanjq/Teletron/baseline_outputs/infer_1steps_77frames/BV1PK411N75n_p63_p43.mp4"]

#16.34
for file in files:
    if not file.endswith(".mp4"):
        continue
    original_video_path = file
    original_videos = io.read_video(original_video_path)
    print("original_videos =", original_videos[0].shape)
    
    videos1 = original_videos[0][: ,: ,:832, :] / 255
    videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
    start_col = 1680 - 832
    videos2 = original_videos[0][: ,: ,start_col:, :]  / 255
    videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)
    device = torch.device("cuda")
    result = {}
    only_final = False
    result['psnr'] = calculate_psnr(videos1, videos2, device)
    mean_value = np.mean(result['psnr']['value'])
    all_mean += mean_value
    num += 1
    print(f"{original_video_path} psnr: {mean_value}")
all_mean /= num
print(f"psnr: {all_mean}")

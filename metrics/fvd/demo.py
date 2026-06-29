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


# videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
original_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/HEVC/processed_data"
level_0_h264_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_0_h264_videos"
level_0_h265_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_0_h265_videos"
level_1_h264_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_1_h264_videos"
level_1_h265_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_1_h265_videos"
level_2_h264_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_2_h264_videos"
level_2_h265_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_2_h265_videos"
level_3_h264_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_3_h264_videos"
level_3_h265_prefix = "/nvfile-heatstorage/Text2Video/XJY/data/level_3_h265_videos"
level_0_h264 = []
level_0_h265 = []
level_1_h264 = []
level_1_h265 = []
level_2_h264 = []
level_2_h265 = []
level_3_h264 = []
level_3_h265 = []
prefixes = [level_0_h264_prefix, level_0_h265_prefix, level_1_h264_prefix, level_1_h265_prefix, 
            level_2_h264_prefix, level_2_h265_prefix, level_3_h264_prefix, level_3_h265_prefix]

for file in os.listdir(original_prefix):
    if not file.endswith(".mp4"):
        continue
    original_video_path = os.path.join(original_prefix, file)
    original_videos = io.read_video(original_video_path)
    videos1 = original_videos[0] / 255
    videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
    for i, prefix in enumerate(prefixes):
        video_path = os.path.join(prefix, file)
        eval_videos = io.read_video(video_path)
        videos2 = eval_videos[0] / 255
        videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)
        device = torch.device("cuda")
        result = {}
        only_final = False
        result['fvd'] = calculate_fvd(videos1, videos2, device)
        mean_value = np.mean(result['fvd']['value'])
        print(f"{video_path} fvd: {mean_value}")
        if i == 0:
            level_0_h264.append(mean_value)
        elif i == 1:
            level_0_h265.append(mean_value)
        elif i == 2:
            level_1_h264.append(mean_value)
        elif i == 3:
            level_1_h265.append(mean_value)
        elif i == 4:
            level_2_h264.append(mean_value)
        elif i == 5:
            level_2_h265.append(mean_value)
        elif i == 6:
            level_3_h264.append(mean_value)
        else:
            level_3_h265.append(mean_value)
    

level_0_h264_mean = np.mean(level_0_h264)
level_0_h265_mean = np.mean(level_0_h265)
level_1_h264_mean = np.mean(level_1_h264)
level_1_h265_mean = np.mean(level_1_h265)
level_2_h264_mean = np.mean(level_2_h264)
level_2_h265_mean = np.mean(level_2_h265)
level_3_h264_mean = np.mean(level_3_h264)
level_3_h265_mean = np.mean(level_3_h265)

print("On HEVC dataset:")
print(f"level_0_h264 fvd: {level_0_h264_mean}")
print(f"level_0_h265 fvd: {level_0_h265_mean}")
print(f"level_1_h264 fvd: {level_1_h264_mean}")
print(f"level_1_h265 fvd: {level_1_h265_mean}")
print(f"level_2_h264 fvd: {level_2_h264_mean}")
print(f"level_2_h265 fvd: {level_2_h265_mean}")
print(f"level_3_h264 fvd: {level_3_h264_mean}")
print(f"level_3_h265 fvd: {level_3_h265_mean}")


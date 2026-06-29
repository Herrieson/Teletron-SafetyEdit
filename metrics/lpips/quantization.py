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


original_video_path = "/gemini/space/yifq/xjy/results/fl2v_1.3B_recon_480p_f141_sft/release/BV1PK411N75n_p63_14B.mp4"
original_videos = io.read_video(original_video_path)

videos1 = original_videos[0][: ,: ,:832, :] / 255
videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
start_col = 1680 - 832
videos2 = original_videos[0][: ,: ,start_col:, :]  / 255
videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)
device = torch.device("cuda")
result = {}
only_final = False
result['lpips'] = calculate_lpips(videos1, videos2, device=device, calculate_per_frame=1, calculate_final=only_final)
mean_value = np.mean(result['lpips'])
print(f"{original_video_path} lpips: {mean_value}")
    




import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips
import os
import cv2
from torchvision.transforms import ToTensor

spatial = True         # Return a spatial map of perceptual distance.


loss_fn = lpips.LPIPS(net='alex',spatial=spatial)


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x * 2 - 1

    return x



def calculate_lpips(videos1, videos2, calculate_per_frame, calculate_final, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    print("calculate_lpips...")

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    videos1 = trans(videos1)
    videos2 = trans(videos2)
    
    lpips_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).cuda()
            img2 = video2[clip_timestamp].unsqueeze(0).cuda()
            
            loss_fn.to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    
    lpips_results = np.array(lpips_results)
    
    return lpips_results 

# 视频加载和处理
def load_video_as_tensor(path, max_frames=None, resize_short_side_to=None):
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if resize_short_side_to is not None:
            h, w = frame.shape[:2]
            scale = resize_short_side_to / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_tensor = ToTensor()(frame)  # shape: [C, H, W]
        frames.append(frame_tensor)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError("No frames read from video.")

    video_tensor = torch.stack(frames)  # [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0)         # [1, C, T, H, W]
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]

    return video_tensor  # shape: [1, VIDEO_LENGTH, 3, H, W]

def main():
    NUMBER_OF_VIDEOS = 1
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    CALCULATE_FINAL = True
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    # videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    video_tensor_1= load_video_as_tensor(
    "/nvfile-heatstorage/wrq/teleai_pipe/work_dirs/output/pose/tiaoshui/sport_tiaoshui_clip_0124/6_0_40_cfg.mp4",
    max_frames=None,              # None = 全部帧
    resize_short_side_to=224      # 将短边缩放为224，保留纵横比
)   
 
    video_tensor_2= load_video_as_tensor(
    "/nvfile-heatstorage/wrq/teleai_pipe/work_dirs/output/pose/tiaoshui/sport_tiaoshui_clip_0124/6_1_40_cfg.mp4",
    max_frames=None,              # None = 全部帧
    resize_short_side_to=224      # 将短边缩放为224，保留纵横比
)
    import json
    result = calculate_lpips(videos1, videos2,CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
    
    print(result)

if __name__ == "__main__":
    main()
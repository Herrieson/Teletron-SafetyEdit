# import numpy as np
# import torch
# from tqdm import tqdm
# import math

# import torch
# import lpips
# import cv2
# from torchvision.transforms import ToTensor

# spatial = True         # Return a spatial map of perceptual distance.

# # Linearly calibrated models (LPIPS)
# # loss_fn = lpips.LPIPS(net='alex', 
# #                       spatial=spatial, 
# #                       pretrained=True, 
# #                       model_path='/nvfile-heatstorage/wrq/video_Compression/Compression-Diffusion/fvd_utils/hub/checkpoints/alexnet-owt-7be5be79.pth',
# #                       eval_mode=True,
# #                       verbose=True,
# #                       ) 
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

# def trans(x):
#     # if greyscale images add channel
#     if x.shape[-3] == 1:
#         x = x.repeat(1, 1, 3, 1, 1)

#     # value range [0, 1] -> [-1, 1]
#     x = x * 2 - 1

#     return x

# # def calculate_lpips(videos1, videos2, calculate_per_frame, calculate_final, device):
# #     # image should be RGB, IMPORTANT: normalized to [-1,1]
# #     print("calculate_lpips...")

# #     assert videos1.shape == videos2.shape

# #     # videos [batch_size, timestamps, channel, h, w]

# #     # support grayscale input, if grayscale -> channel*3
# #     # value range [0, 1] -> [-1, 1]
# #     videos1 = trans(videos1)
# #     videos2 = trans(videos2)
    
# #     lpips_results = []

# #     for video_num in tqdm(range(videos1.shape[0])):
# #         # get a video
# #         # video [timestamps, channel, h, w]
# #         video1 = videos1[video_num]
# #         video2 = videos2[video_num]

# #         lpips_results_of_a_video = []
# #         for clip_timestamp in range(len(video1)):
# #             # get a img
# #             # img [timestamps[x], channel, h, w]
# #             # img [channel, h, w] tensor

# #             img1 = video1[clip_timestamp].unsqueeze(0).cuda()
# #             img2 = video2[clip_timestamp].unsqueeze(0).cuda()
            
# #             loss_fn.to(device)

# #             # calculate lpips of a video
# #             lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
# #         lpips_results.append(lpips_results_of_a_video)
    
# #     lpips_results = np.array(lpips_results)
    
# #     lpips = {}
# #     lpips_std = {}

# #     for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
# #         lpips[f'avg[:{clip_timestamp}]'] = np.mean(lpips_results[:,:clip_timestamp])
# #         lpips_std[f'std[:{clip_timestamp}]'] = np.std(lpips_results[:,:clip_timestamp])

# #     if calculate_final:
# #         lpips['final'] = np.mean(lpips_results)
# #         lpips_std['final'] = np.std(lpips_results)

# #     result = {
# #         "lpips": lpips,
# #         "lpips_std": lpips_std,
# #         "lpips_per_frame": calculate_per_frame,
# #         "lpips_video_setting": video1.shape,
# #         "lpips_video_setting_name": "time, channel, heigth, width",
# #     }

# #     return result

# # test code / using example

# def calculate_lpips(videos1, videos2, calculate_per_frame, calculate_final, device):
#     # image should be RGB, IMPORTANT: normalized to [-1,1]
#     print("calculate_lpips...")

#     assert videos1.shape == videos2.shape

#     # videos [batch_size, timestamps, channel, h, w]

#     # support grayscale input, if grayscale -> channel*3
#     # value range [0, 1] -> [-1, 1]
#     videos1 = trans(videos1)
#     videos2 = trans(videos2)
    
#     lpips_results = []

#     for video_num in tqdm(range(videos1.shape[0])):
#         # get a video
#         # video [timestamps, channel, h, w]
#         video1 = videos1[video_num]
#         video2 = videos2[video_num]

#         lpips_results_of_a_video = []
#         for clip_timestamp in range(len(video1)):
#             # get a img
#             # img [timestamps[x], channel, h, w]
#             # img [channel, h, w] tensor

#             img1 = video1[clip_timestamp].unsqueeze(0).cuda()
#             img2 = video2[clip_timestamp].unsqueeze(0).cuda()
            
#             loss_fn.to(device)

#             # calculate lpips of a video
#             lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
#         lpips_results.append(lpips_results_of_a_video)
    
#     lpips_results = np.array(lpips_results)
    
#     # lpips = {}
#     # lpips_std = {}

#     # for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
#     #     lpips[f'avg[:{clip_timestamp}]'] = np.mean(lpips_results[:,:clip_timestamp])
#     #     lpips_std[f'std[:{clip_timestamp}]'] = np.std(lpips_results[:,:clip_timestamp])

#     # if calculate_final:
#     #     lpips['final'] = np.mean(lpips_results)
#     #     lpips_std['final'] = np.std(lpips_results)

#     # result = {
#     #     "lpips": lpips,
#     #     "lpips_std": lpips_std,
#     #     # "lpips_per_frame": calculate_per_frame,
#     #     # "lpips_video_setting": video1.shape,
#     #     # "lpips_video_setting_name": "time, channel, heigth, width",
#     # }
#     # import ipdb;
#     # ipdb.set_trace()
#     return lpips_results 


# # 视频加载和处理
# def load_video_as_tensor(path, max_frames=None, resize_short_side_to=None):
#     cap = cv2.VideoCapture(path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frames = []

#     for i in range(frame_count):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         if resize_short_side_to is not None:
#             h, w = frame.shape[:2]
#             scale = resize_short_side_to / min(h, w)
#             new_w, new_h = int(w * scale), int(h * scale)
#             frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

#         frame_tensor = ToTensor()(frame)  # shape: [C, H, W]
#         frames.append(frame_tensor)

#         if max_frames is not None and len(frames) >= max_frames:
#             break

#     cap.release()

#     if not frames:
#         raise ValueError("No frames read from video.")

#     video_tensor = torch.stack(frames)  # [T, C, H, W]
#     video_tensor = video_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
#     video_tensor = video_tensor.unsqueeze(0)         # [1, C, T, H, W]
#     video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]

#     return video_tensor  # shape: [1, VIDEO_LENGTH, 3, H, W]



# def main():
#     NUMBER_OF_VIDEOS = 1
#     VIDEO_LENGTH = 50
#     CHANNEL = 3
#     SIZE = 64
#     CALCULATE_PER_FRAME = 5
#     CALCULATE_FINAL = True
#     videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     # 构造完整张量
#     video_tensor_1= load_video_as_tensor(
#     "/nvfile-heatstorage/wrq/teleai_pipe/work_dirs/output/pose/tiaoshui/sport_tiaoshui_clip_0124/6_0_40_cfg.mp4",
#     max_frames=None,              # None = 全部帧
#     resize_short_side_to=224      # 将短边缩放为224，保留纵横比
# )
#     video_tensor_2= load_video_as_tensor(
#     "/nvfile-heatstorage/wrq/teleai_pipe/work_dirs/output/pose/tiaoshui/sport_tiaoshui_clip_0124/634_1_40_cfg.mp4",
#     max_frames=None,              # None = 全部帧
#     resize_short_side_to=224      # 将短边缩放为224，保留纵横比
# )
#     images1 = torch.rand(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
#     images2 = torch.rand(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE,  requires_grad=False)
#     device = torch.device("cuda")

#     import json
#     # import ipdb;
#     # ipdb.set_trace()

#     result = calculate_lpips(videos1,videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
#     print(result)

# if __name__ == "__main__":
#     main()



import numpy as np
import torch
from tqdm import tqdm
import math

import torch
import lpips

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

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
    
    lpips = {}
    lpips_std = {}

    for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
        lpips[f'avg[:{clip_timestamp}]'] = np.mean(lpips_results[:,:clip_timestamp])
        lpips_std[f'std[:{clip_timestamp}]'] = np.std(lpips_results[:,:clip_timestamp])

    if calculate_final:
        lpips['final'] = np.mean(lpips_results)
        lpips_std['final'] = np.std(lpips_results)

    result = {
        "lpips": lpips,
        "lpips_std": lpips_std,
        "lpips_per_frame": calculate_per_frame,
        "lpips_video_setting": video1.shape,
        "lpips_video_setting_name": "time, channel, heigth, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 1
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    CALCULATE_FINAL = True
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    import json
    result = calculate_lpips(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
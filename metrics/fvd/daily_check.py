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
import re
# ps: pixel value should be in [0, 1]!


# videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# prefix = "/nvfile-heatstorage/AIGC_H100/XJY/code/vast/projects/wan/results_0724_wanfl2v_multi"
# prefix = "/nvfile-heatstorage/AIGC_H100/XJY/code/vast/projects/wan/results_0822_wanfl2v_canny_UVG"
# prefix = "/nvfile-heatstorage/AIGC_H100/XJY/code/vast/projects/wan/results_0919_0kl_HEVC"
# prefix = "/gemini/space/yifq/xjy/results/14B_vae"
# prefix = "/gemini/space/yanjq/VAST/results_testdata"
#prefix = "/gemini/space/yifq/xjy/results/multi_resolution_iter_0003500"
# prefix = "/gemini/space/yifq/xjy/results/iter_0023000"

# prefix = "/gemini/space/yanjq/Teletron/outputs/baseline_10steps_infer_3500_77frames/"
#prefix = "/gemini/space/yanjq/Teletron/outputs/baseline_3steps_infer_3500_77frames/"
#prefix = "/gemini/space/yanjq/Teletron/outputs/baseline_2steps_infer_3500_77frames/"
#prefix = "/gemini/space/yanjq/Teletron/outputs/baseline_1step_infer_3500_77frames/"
# prefix = "/gemini/space/yanjq/Teletron/outputs/onestep_t1_train_t1_infer_5500_77frames/"
# prefix = "/gemini/space/yanjq/VAST/results_testdata_reproduce_vast_1_3B"
#prefix = "/gemini/space/yanjq/Teletron/train_gan_outputs/1104_3_infer_5560_77frames"

# prefix = "/gemini/space/yanjq/Teletron/pd2_outputs/infer_1step_1100iters_77frames/"

# prefix = "/gemini/space/yifq/xjy/results/multi_resolution_w_canny_mask_iter_0009000_resize_4_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/multi_resolution_w_canny_mask_iter_0006000_resize_4_avs3_all_frames"

# prefix = "/gemini/space/yifq/xjy/results/tae_w_canny_mask_iter_0002000_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/previous_ckpt_vae_w_canny_mask_iter_0012000_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/previous_ckpt_vae_w_canny_mask_iter_0003500_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/tae_w_canny_mask_iter_0002000_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/vae_w_canny_mask_iter_0003500_shuaijiao"
# prefix = "/gemini/space/yifq/xjy/results/previous_ckpt_vae_w_canny_mask_iter_0012000_shuaijiao_new"
# prefix = "/gemini/space/yifq/xjy/results/tae_mask_iter_0003500_shuaijiao_1216"
# prefix = "/gemini/space/yifq/xjy/results/vae_mask_iter_0012000_shuaijiao_1216"
# prefix = "/gemini/space/yifq/xjy/results/1.3B_soccer_1223_with_keyframe"
# prefix = "/gemini/platform/shared/xujingyu/xujy/data/test/hm_hevc_test_sets"
# prefix = "/gemini/platform/shared/yifq1/yifq/test_results_fixframeinterlap_gop29"
# prefix = "/gemini/space/yifq/xjy/h265_hevc"
# prefix = "/gemini/platform/shared/yifq1/yifq/test_results_fixframeinterlap_gop29"
prefix = "/gemini/space/yifq/xjy/h265_mcljcv_new"


all_mean = 0.0
num = 0
for root, dirs, files in os.walk(prefix):
    for file in files:
        if not file.endswith(".mp4"):
            continue

        original_video_path = os.path.join(root, file)
        original_videos = io.read_video(original_video_path)
        
        # videos1 = original_videos[0][: ,: ,:1984, :] / 255
        videos1 = original_videos[0][: ,: ,:832, :] / 255
        videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
        # start_col = 3984 - 1984
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

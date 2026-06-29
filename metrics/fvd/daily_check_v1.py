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
# pred_prefix = [
#     "/gemini/platform/shared/yifq1/yifq/public_benchmarks/HEVC/HEVC_B",
#     "/gemini/platform/shared/yifq1/yifq/public_benchmarks/HEVC/HEVC_C",
#     "/gemini/platform/shared/yifq1/yifq/public_benchmarks/HEVC/HEVC_E",
# ]
pred_prefix = [
    "/gemini/platform/shared/yifq1/yifq/public_benchmarks/MCL_JCV/720P/YUV_source",
]
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_output_path(original_path):
    file_name = os.path.basename(original_path).split("_qp")[0] + ".mp4"
    for prefix in pred_prefix:
        candidate_path = os.path.join(prefix, file_name)
        if os.path.exists(candidate_path):
            return candidate_path
    return None

def get_resolution_from_filename(filename):
    match = re.search(r'_(\d+)x(\d+)_', filename)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width, height
    else:
        return None, None

all_mean = 0.0
num = 0

# original_video_path = "/gemini/space/yifq/xjy/data/results/14B_avs3/MarketPlace_1920x1080_60_recon.mp4"
original_video_path = "/gemini/space/yifq/xjy/data/results/14B_avs_1230_qp_38/RitualDance_1920x1080_60/_p0_recon.mp4"
original_videos = io.read_video(original_video_path)
original_video_path1 = "/gemini/space/yifq/xjy/data/results/14B_avs_1230_qp_38/RitualDance_1920x1080_60/_p1_recon.mp4"
original_videos1 = io.read_video(original_video_path1)
# videos1 = original_videos[0][: ,: ,:1984, :] / 255
# videos1 = original_videos[0][: ,: ,:832, :] / 255
w, h = get_resolution_from_filename("RitualDance_1920x1080_60")
if h == 480:
    videos1 = original_videos[0][:, :480, :, :] / 255
elif h == 720:
    videos1 = original_videos[0][:, :720, :, :] / 255
elif h == 1080:
    videos1 = original_videos[0][:, -1080:, :, :] / 255
    videos3 = original_videos1[0][:, -1080:, :, :] / 255
else:
    print(f"Unsupported height {h}")
# videos1 = original_videos[0] / 255
videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
videos3 = videos3.unsqueeze(0).permute(0, 1, 4, 2, 3)
# 在时间上一起
videos1 = torch.cat([videos1, videos3], dim=1)
# start_col = 3984 - 1984
# start_col = 1680 - 832
# videos2 = original_videos[0][: ,: ,start_col:, :]  / 255

videos2_path = "/gemini/platform/shared/xujingyu/xujy/data/avs3/AVS3_Seq/rgb_video/RitualDance_1920x1080_60.mp4"
reconstructed_videos = io.read_video(videos2_path)
videos2 = reconstructed_videos[0] / 255

videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)

# length = min(videos1.shape[1], videos2.shape[1])
length=77
videos1 = videos1[:, :77, :, :, :]
videos2 = videos2[:, :77, :, :, :]
print(videos1.shape)

print(videos2.shape)

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

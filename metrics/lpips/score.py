import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
# from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import json

# # ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 8#视频的批量大小
VIDEO_LENGTH = 30 #每个视频的帧数
CHANNEL = 3 #视频帧的通道数
SIZE = 64 #视频帧的高度和宽度
# CALCULATE_PER_FRAME = 5
# CALCULATE_FINAL = True
# videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
# device = torch.device("cuda")


# # # 输出 psnr[T],ssim[T],lpips[T],fvd[1]
# import json
# result = {}
# result['fvd'] = calculate_fvd(videos1, videos1, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
# # result['ssim'] = calculate_ssim(videos1, videos1, CALCULATE_PER_FRAME, CALCULATE_FINAL)
# # result['psnr'] = calculate_psnr(videos1, videos1, CALCULATE_PER_FRAME, CALCULATE_FINAL)
# # result['lpips'] = calculate_lpips(videos1, videos1, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
# # print(json.dumps(result, indent=4))
# print(result['fvd'])





def calculate_score(videos1,videos2,calculate_per_frame,calculate_final,device):
    result = {}
    # import ipdb;
    # ipdb.set_trace()
    # result['fvd'] = calculate_fvd(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
    # result['psnr'] = calculate_psnr(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL)
    result['lpips'] = calculate_lpips(videos1, videos2, CALCULATE_PER_FRAME, CALCULATE_FINAL, device)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")
    CALCULATE_PER_FRAME = 7
    CALCULATE_FINAL = True
    calculate_score(videos1,videos2,CALCULATE_PER_FRAME,CALCULATE_FINAL,device)
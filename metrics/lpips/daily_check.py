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
import matplotlib.pyplot as plt
# ps: pixel value should be in [0, 1]!


prefix = "/gemini/space/yifq/yifq/vis_result_public/tae_f29_dynamiccanny_e4000"
video_width = 852

def cal_lpips(video_path, video_width):
    original_videos = io.read_video(video_path)
    videos1 = original_videos[0][: ,: ,:video_width, :] / 255
    videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
    start_col = original_videos[0].shape[2] - video_width
    videos2 = original_videos[0][: ,: ,start_col:, :]  / 255        
    videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)

    device = torch.device("cuda")
    result = {}
    only_final = False
    result['lpips'] = calculate_lpips(videos1, videos2, device=device, calculate_per_frame=1, calculate_final=only_final)
    mean_value = np.mean(result['lpips'])
    return mean_value

def cal_motion(video_path, video_width):
    original_videos = io.read_video(video_path)
    images = original_videos[0][: ,: ,:video_width, :].permute(0,3,1,2)
    T, C, H, W = images.shape
    if C == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], 
                              device=images.device).view(1, 3, 1, 1)
        gray_images = (images.float() * weights).sum(dim=1, keepdim=True)  # T1HW
    else:
        gray_images = images.float().mean(dim=1, keepdim=True)  # 多通道取平均
    
    # 计算帧间绝对差异
    motion_diffs = []
    for t in range(T - 1):
        diff = torch.abs(gray_images[t] - gray_images[t + 1]).sum()
        motion_diffs.append(diff.item())
    return sum(motion_diffs) / len(motion_diffs) / H / W


def lpips_mean(prefix, video_width):
    all_means = []
    for root, dirs, files in os.walk(prefix):
        for file in files:
            if not file.endswith(".mp4"):
                continue

            original_video_path = os.path.join(root, file)
            original_videos = io.read_video(original_video_path)
            
            videos1 = original_videos[0][: ,: ,:video_width, :] / 255
            videos1 = videos1.unsqueeze(0).permute(0, 1, 4, 2, 3)
            start_col = original_videos[0].shape[2] - video_width
            videos2 = original_videos[0][: ,: ,start_col:, :]  / 255
            videos2 = videos2.unsqueeze(0).permute(0, 1, 4, 2, 3)

            device = torch.device("cuda")
            result = {}
            only_final = False
            result['lpips'] = calculate_lpips(videos1, videos2, device=device, calculate_per_frame=1, calculate_final=only_final)
            mean_value = np.mean(result['lpips'])
            all_means.append(mean_value)
    return all_means
    
    
def vis1():
    baseline = lpips_mean("/gemini/space/yifq/yifq/vis_result_public/tae_f29_canny/shuaijiao", video_width)
    baseline = sum(baseline) / len(baseline)

    canny_counts = []
    lpips_values = []

    for folder in os.listdir(prefix):
        canny_cnt = int(folder.split("_")[1])
        folder_path = os.path.join(prefix, folder)
        all_means = lpips_mean(folder_path, video_width)
        avg_mean = sum(all_means) / len(all_means)
        canny_counts.append(canny_cnt)
        lpips_values.append(avg_mean)

    canny_counts, lpips_values = zip(*sorted(zip(canny_counts, lpips_values)))
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(canny_counts, lpips_values, 'b-', marker='o', linewidth=2, markersize=8, label='canny_cnt vs LPIPS')

    # 绘制baseline点
    plt.scatter(27, baseline, color='red', s=150, zorder=5, label=f'Baseline (27, {baseline:.4f})')
    # 添加标签和标题
    plt.xlabel('Canny Count', fontsize=12)
    plt.ylabel('LPIPS', fontsize=12)
    plt.title('LPIPS vs Canny Count (e4000)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 优化布局
    plt.tight_layout()
    plt.savefig('lpips_vs_canny_cnt_e4000.png', dpi=300, bbox_inches='tight')


def vis2():
    from matplotlib.patches import Rectangle
    import matplotlib.cm as cm


    video_names, video_folders = [], []
    for folder in os.listdir(prefix):
        video_folders.append(folder)

    for file in os.listdir(os.path.join(prefix, video_folders[0])):
        if file.endswith(".mp4"):
            video_names.append(file)



    # 首先收集所有视频的motion_avg值
    print("收集和处理数据...")
    all_motions = []
    for video_name in video_names:
        video_path = os.path.join(prefix, video_folders[0], video_name)
        motion_avg = cal_motion(video_path, video_width)
        all_motions.append(motion_avg)

    # 确定分桶边界（25个桶）
    all_motions = np.array(all_motions)
    motion_min = np.min(all_motions)
    motion_max = np.max(all_motions)
    bin_edges = np.linspace(motion_min, motion_max, 26)  # 26个边界点，25个桶
    bucket_ranges = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]
    bucket_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

    # 初始化数据结构
    # bucket_videos: 每个桶包含的视频索引列表
    bucket_videos = [[] for _ in range(len(bucket_ranges))]

    # 第一步：将视频分配到桶中
    for video_idx, video_name in enumerate(video_names):
        video_path = os.path.join(prefix, video_folders[0], video_name)
        motion_avg = cal_motion(video_path, video_width)
        
        # 确定桶索引
        bucket_idx = np.digitize(motion_avg, bin_edges) - 1
        bucket_idx = max(0, min(bucket_idx, len(bucket_ranges)-1))
        
        # 将视频索引添加到对应的桶
        bucket_videos[bucket_idx].append(video_idx)

    # 第二步：收集所有canny_cnt值
    all_canny_cnts = set()
    for folder in video_folders:
        try:
            canny_cnt = int(folder.split("_")[1])
            all_canny_cnts.add(canny_cnt)
        except (IndexError, ValueError):
            print(f"跳过无法解析的文件夹: {folder}")
            continue

    all_canny_cnts = sorted(list(all_canny_cnts))

    # 第三步：为每个桶计算每个canny_cnt的LPIPS平均值
    # bucket_canny_avgs: 每个桶 -> {canny_cnt: 平均值}
    bucket_canny_avgs = [{} for _ in range(len(bucket_ranges))]
    # bucket_baseline_avgs: 每个桶的baseline平均值
    bucket_baseline_avgs = [0 for _ in range(len(bucket_ranges))]

    # 初始化存储结构
    for bucket_idx in range(len(bucket_ranges)):
        for canny_cnt in all_canny_cnts:
            bucket_canny_avgs[bucket_idx][canny_cnt] = []

    # 遍历所有文件夹（canny_cnt）
    for folder in video_folders:
        try:
            canny_cnt = int(folder.split("_")[1])
        except (IndexError, ValueError):
            continue
        
        print(f"处理canny_cnt={canny_cnt}的文件夹: {folder}")
        
        # 遍历所有视频
        for video_idx, video_name in enumerate(video_names):
            video_path = os.path.join(prefix, folder, video_name)
            lpips_val = cal_lpips(video_path, video_width)
            
            # 找到视频所在的桶
            for bucket_idx, video_indices in enumerate(bucket_videos):
                if video_idx in video_indices:
                    bucket_canny_avgs[bucket_idx][canny_cnt].append(lpips_val)
                    break

    # 第四步：计算每个视频的baseline LPIPS，并分配到桶中
    baseline_values_by_bucket = [[] for _ in range(len(bucket_ranges))]
    for video_idx, video_name in enumerate(video_names):
        baseline_path = os.path.join("/gemini/space/yifq/yifq/vis_result_public/tae_f29_canny/", video_name)
        lpips_val_baseline = cal_lpips(baseline_path, video_width)
        
        # 找到视频所在的桶
        for bucket_idx, video_indices in enumerate(bucket_videos):
            if video_idx in video_indices:
                baseline_values_by_bucket[bucket_idx].append(lpips_val_baseline)
                break

    # 第五步：计算平均值
    for bucket_idx in range(len(bucket_ranges)):
        # 计算每个canny_cnt的平均值
        for canny_cnt in all_canny_cnts:
            values = bucket_canny_avgs[bucket_idx][canny_cnt]
            if values:  # 如果有数据
                bucket_canny_avgs[bucket_idx][canny_cnt] = np.mean(values)
            else:
                bucket_canny_avgs[bucket_idx][canny_cnt] = np.nan
        
        # 计算baseline平均值
        if baseline_values_by_bucket[bucket_idx]:
            bucket_baseline_avgs[bucket_idx] = np.mean(baseline_values_by_bucket[bucket_idx])
        else:
            bucket_baseline_avgs[bucket_idx] = np.nan

    # 第六步：创建图形
    plt.figure(figsize=(20, 10))

    # 为canny_cnt创建颜色映射
    if all_canny_cnts:
        min_cnt = min(all_canny_cnts)
        max_cnt = max(all_canny_cnts)
        norm = plt.Normalize(vmin=min_cnt, vmax=max_cnt)
        colormap = cm.viridis
    else:
        norm = None
        colormap = None

    # 绘制每个桶内的散点
    for bucket_idx in range(len(bucket_ranges)):
        x_pos = bucket_idx  # 使用桶索引作为x位置
        
        # 绘制每个canny_cnt的散点
        for canny_cnt in all_canny_cnts:
            avg_lpips = bucket_canny_avgs[bucket_idx][canny_cnt]
            if not np.isnan(avg_lpips):  # 只绘制有数据的点
                # 确定颜色
                if norm and colormap:
                    color = colormap(norm(canny_cnt))
                else:
                    color = 'blue'
                
                # 绘制散点
                plt.scatter(x_pos, avg_lpips, 
                        color=color, 
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=3)
        
        # 绘制baseline散点
        if not np.isnan(bucket_baseline_avgs[bucket_idx]):
            plt.scatter(x_pos, bucket_baseline_avgs[bucket_idx],
                    color='red',
                    s=150,
                    marker='s',  # 方形标记
                    edgecolors='black',
                    linewidths=1,
                    zorder=5,
                    label='Baseline' if bucket_idx == 0 else "")

    # 设置x轴
    plt.xticks(
        range(len(bucket_ranges)), 
        [f"{bucket_range[0]:.2f}\n-\n{bucket_range[1]:.2f}" for bucket_range in bucket_ranges],
        rotation=45,
        ha='right',
        fontsize=9
    )

    # 添加网格
    plt.grid(True, alpha=0.3, axis='y')

    # 添加标签和标题
    plt.xlabel('Motion Average Buckets', fontsize=12)
    plt.ylabel('Average LPIPS', fontsize=12)
    plt.title('Average LPIPS by Motion Average Buckets and Canny Count (Baseline in Red)', fontsize=14)

    # 创建颜色条
    if norm and colormap:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
        cbar.set_label('Canny Count', fontsize=12)

    # 添加图例
    plt.legend(['Baseline'], loc='upper right')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.savefig('lpips_scatter_by_motion_buckets.png', dpi=300, bbox_inches='tight')

    


if __name__ == '__main__':
    vis1()
    # vis2()
    # results_1080 = lpips_mean("/gemini/space/yifq/yifq/vis_result_public/tae_f29_canny_0106_model/MCJCV_1080p", video_width=1920)
    # results_720 = lpips_mean("/gemini/space/yifq/yifq/vis_result_public/tae_f29_canny_0106_model/MCJCV_720p", video_width=1280)

    # print(sum(results_1080) / len(results_1080))
    # print(sum(results_720) / len(results_720))


    

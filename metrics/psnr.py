import cv2
import numpy as np
import os

def calculate_psnr_with_metrics(video_path):
    """
    增强版本：计算PSNR并返回更多统计信息
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_width = width // 2
    
    psnr_values = []
    mse_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 分割视频
        left_frame = frame[:, :832]
        right_frame = frame[:, -832:]
        
        # 使用Y通道计算
        left_y = cv2.cvtColor(left_frame, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float64)
        right_y = cv2.cvtColor(right_frame, cv2.COLOR_BGR2YUV)[:,:,0].astype(np.float64)
        
        # 计算MSE和PSNR
        mse = np.mean((left_y - right_y) ** 2)
        psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        
        psnr_values.append(psnr)
        mse_values.append(mse)
    
    cap.release()
    
    if not psnr_values:
        return None
    
    # 计算统计信息
    stats = {
        'average_psnr': np.mean(psnr_values),
        'min_psnr': np.min(psnr_values),
        'max_psnr': np.max(psnr_values),
        'std_psnr': np.std(psnr_values),
        'average_mse': np.mean(mse_values),
        'total_frames': len(psnr_values),
        'psnr_values': psnr_values,
        'mse_values': mse_values
    }
    
    return stats

# 使用示例
if __name__ == "__main__":
    prefix = "/gemini/space/yifq/xjy/results/multi_resolution_w_canny_mask_iter_0006000_resize_4_avs3"
    psnr_values = 0.0
    num = 0
    for file in os.listdir(prefix):
        if not file.endswith(".mp4"):
            continue
        video_path = os.path.join(prefix, file)
    
        # 增强版本（获取更多统计信息）
        print("\n计算详细统计信息...")
        stats = calculate_psnr_with_metrics(video_path)
        
        if stats:
            print(f"\n详细统计结果:")
            print(f"平均PSNR: {stats['average_psnr']:.2f} dB")
            print(f"最小PSNR: {stats['min_psnr']:.2f} dB")
            print(f"最大PSNR: {stats['max_psnr']:.2f} dB")
            print(f"PSNR标准差: {stats['std_psnr']:.2f} dB")
            print(f"平均MSE: {stats['average_mse']:.2f}")
            print(f"总帧数: {stats['total_frames']}")
        psnr_values += stats['average_psnr']
        num += 1
    print(f"平均PSNR: {psnr_values/num:.2f} dB")
    
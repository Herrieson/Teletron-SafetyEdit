import numpy as np
import sys
import os
from pathlib import Path
import argparse

def read_yuv420_file(yuv_path, width, height, frame_index=0):
    """
    读取指定帧的 YUV420 数据
    
    Args:
        yuv_path: YUV 文件路径
        width: 视频宽度
        height: 视频高度
        frame_index: 帧索引（从0开始）
    
    Returns:
        tuple: (Y, U, V) 分量的 numpy 数组
    """
    # 计算每帧的大小（字节）
    frame_size = width * height * 3 // 2  # YUV420
    
    # 打开文件并定位到指定帧
    with open(yuv_path, 'rb') as f:
        # 跳转到指定帧
        f.seek(frame_index * frame_size)
        
        # 读取整帧数据
        yuv_data = np.frombuffer(f.read(frame_size), dtype=np.uint8)
        
        if len(yuv_data) < frame_size:
            raise ValueError(f"文件大小不足，无法读取第 {frame_index} 帧")
    
    # 分离 YUV 分量
    y_size = width * height
    uv_size = y_size // 4
    
    # Y 分量
    Y = yuv_data[:y_size].reshape(height, width)
    
    # U 分量（在 Y 之后）
    u_start = y_size
    U = yuv_data[u_start:u_start+uv_size].reshape(height//2, width//2)
    
    # V 分量（在 U 之后）
    v_start = y_size + uv_size
    V = yuv_data[v_start:v_start+uv_size].reshape(height//2, width//2)
    
    return Y, U, V

def calculate_psnr(gt, pred, max_val=255.0):
    """
    计算 PSNR
    
    Args:
        gt: 原始图像（ground truth）
        pred: 生成图像
        max_val: 最大像素值（默认255）
    
    Returns:
        float: PSNR 值（单位：dB）
    """
    # 确保数据类型一致
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    
    # 计算 MSE
    mse = np.mean((gt - pred) ** 2)
    
    # 避免除零
    if mse == 0:
        return float('inf')
    
    # 计算 PSNR
    psnr = 10 * np.log10((max_val ** 2) / mse)
    
    return psnr

def calculate_yuv_psnr(gt_yuv_path, pred_yuv_path, width, height, frame_indices=None, bit=8):
    """
    计算 YUV 分量的 PSNR
    
    Args:
        gt_yuv_path: GT YUV 文件路径
        pred_yuv_path: 生成的 YUV 文件路径
        width: 视频宽度
        height: 视频高度
        frame_indices: 要计算的帧索引列表，None 表示计算所有帧
    
    Returns:
        dict: 包含各种 PSNR 值的字典
    """
    # 获取文件大小确定总帧数
    gt_size = os.path.getsize(gt_yuv_path)
    pred_size = os.path.getsize(pred_yuv_path)
    
    frame_size = width * height * 3 // 2
    
    # 计算总帧数
    total_frames = min(gt_size, pred_size) // frame_size
    
    if total_frames == 0:
        raise ValueError("YUV 文件大小不足以包含至少一帧")
    
    # 如果没有指定帧索引，计算所有帧
    if frame_indices is None:
        frame_indices = range(total_frames)
    else:
        # 确保帧索引在有效范围内
        frame_indices = [i for i in frame_indices if i < total_frames]
    
    if not frame_indices:
        raise ValueError("没有有效的帧可用于计算")
    
    print(f"计算 {len(frame_indices)} 帧的 PSNR...")
    
    # 初始化统计变量
    y_psnr_list = []
    u_psnr_list = []
    v_psnr_list = []
    weighted_psnr_list = []
    
    for frame_idx in frame_indices:
        try:
            # 读取 GT 帧
            gt_Y, gt_U, gt_V = read_yuv420_file(gt_yuv_path, width, height, frame_idx)
            
            # 读取生成帧
            pred_Y, pred_U, pred_V = read_yuv420_file(pred_yuv_path, width, height, frame_idx)
            
            # 计算各分量 PSNR
            y_psnr = calculate_psnr(gt_Y, pred_Y, max_val=2**bit - 1)
            u_psnr = calculate_psnr(gt_U, pred_U, max_val=2**bit - 1)
            v_psnr = calculate_psnr(gt_V, pred_V, max_val=2**bit - 1)
            
            # 计算加权 PSNR: (6*Y + U + V) / 8
            weighted_psnr = (6 * y_psnr + u_psnr + v_psnr) / 8
            
            y_psnr_list.append(y_psnr)
            u_psnr_list.append(u_psnr)
            v_psnr_list.append(v_psnr)
            weighted_psnr_list.append(weighted_psnr)
            
            if len(frame_indices) <= 10:  # 如果帧数不多，显示每帧结果
                print(f"帧 {frame_idx:3d}: Y-PSNR={y_psnr:6.2f}dB, "
                      f"U-PSNR={u_psnr:6.2f}dB, V-PSNR={v_psnr:6.2f}dB, "
                      f"加权PSNR={weighted_psnr:6.2f}dB")
        
        except Exception as e:
            print(f"警告: 计算帧 {frame_idx} 时出错: {e}")
            continue
    
    if not y_psnr_list:
        raise ValueError("没有成功计算任何帧的 PSNR")
    
    # 计算平均值
    avg_y_psnr = np.mean(y_psnr_list)
    avg_u_psnr = np.mean(u_psnr_list)
    avg_v_psnr = np.mean(v_psnr_list)
    avg_weighted_psnr = np.mean(weighted_psnr_list)
    
    # 计算标准差（可选）
    std_y_psnr = np.std(y_psnr_list)
    std_u_psnr = np.std(u_psnr_list)
    std_v_psnr = np.std(v_psnr_list)
    std_weighted_psnr = np.std(weighted_psnr_list)
    
    return {
        'frame_count': len(y_psnr_list),
        'y_psnr': {
            'mean': avg_y_psnr,
            'std': std_y_psnr,
            'min': np.min(y_psnr_list),
            'max': np.max(y_psnr_list)
        },
        'u_psnr': {
            'mean': avg_u_psnr,
            'std': std_u_psnr,
            'min': np.min(u_psnr_list),
            'max': np.max(u_psnr_list)
        },
        'v_psnr': {
            'mean': avg_v_psnr,
            'std': std_v_psnr,
            'min': np.min(v_psnr_list),
            'max': np.max(v_psnr_list)
        },
        'weighted_psnr': {
            'mean': avg_weighted_psnr,
            'std': std_weighted_psnr,
            'min': np.min(weighted_psnr_list),
            'max': np.max(weighted_psnr_list)
        },
        'per_frame': {
            'y_psnr': y_psnr_list,
            'u_psnr': u_psnr_list,
            'v_psnr': v_psnr_list,
            'weighted_psnr': weighted_psnr_list
        }
    }

def print_psnr_results(results, verbose=False):
    """
    打印 PSNR 计算结果
    """
    print("\n" + "="*80)
    print("YUV PSNR 计算结果")
    print("="*80)
    print(f"总帧数: {results['frame_count']}")
    print("-"*80)
    
    print("分量 PSNR (dB):")
    print(f"  Y-PSNR:  {results['y_psnr']['mean']:6.2f} ± {results['y_psnr']['std']:5.2f} "
          f"[{results['y_psnr']['min']:6.2f}, {results['y_psnr']['max']:6.2f}]")
    print(f"  U-PSNR:  {results['u_psnr']['mean']:6.2f} ± {results['u_psnr']['std']:5.2f} "
          f"[{results['u_psnr']['min']:6.2f}, {results['u_psnr']['max']:6.2f}]")
    print(f"  V-PSNR:  {results['v_psnr']['mean']:6.2f} ± {results['v_psnr']['std']:5.2f} "
          f"[{results['v_psnr']['min']:6.2f}, {results['v_psnr']['max']:6.2f}]")
    
    print("-"*80)
    print(f"加权 PSNR (6*Y+U+V)/8: {results['weighted_psnr']['mean']:6.2f} ± "
          f"{results['weighted_psnr']['std']:5.2f} dB")
    print("="*80)
    
    if verbose and results['frame_count'] > 1:
        print("\n每帧详细结果:")
        for i in range(results['frame_count']):
            print(f"帧 {i:3d}: Y={results['per_frame']['y_psnr'][i]:6.2f}dB, "
                  f"U={results['per_frame']['u_psnr'][i]:6.2f}dB, "
                  f"V={results['per_frame']['v_psnr'][i]:6.2f}dB, "
                  f"加权={results['per_frame']['weighted_psnr'][i]:6.2f}dB")

def save_psnr_results(results, output_path):
    """
    保存 PSNR 结果到文件
    """
    with open(output_path, 'w') as f:
        f.write("# YUV PSNR 计算结果\n")
        f.write(f"总帧数: {results['frame_count']}\n\n")
        
        f.write("平均值 ± 标准差 [最小值, 最大值] (单位: dB)\n")
        f.write("-"*60 + "\n")
        
        f.write(f"Y-PSNR:  {results['y_psnr']['mean']:6.2f} ± {results['y_psnr']['std']:5.2f} "
                f"[{results['y_psnr']['min']:6.2f}, {results['y_psnr']['max']:6.2f}]\n")
        f.write(f"U-PSNR:  {results['u_psnr']['mean']:6.2f} ± {results['u_psnr']['std']:5.2f} "
                f"[{results['u_psnr']['min']:6.2f}, {results['u_psnr']['max']:6.2f}]\n")
        f.write(f"V-PSNR:  {results['v_psnr']['mean']:6.2f} ± {results['v_psnr']['std']:5.2f} "
                f"[{results['v_psnr']['min']:6.2f}, {results['v_psnr']['max']:6.2f}]\n")
        
        f.write("-"*60 + "\n")
        f.write(f"加权 PSNR (6*Y+U+V)/8: {results['weighted_psnr']['mean']:6.2f} ± "
                f"{results['weighted_psnr']['std']:5.2f} dB\n\n")
        
        if results['frame_count'] > 1:
            f.write("每帧详细结果:\n")
            f.write("帧号, Y-PSNR, U-PSNR, V-PSNR, 加权PSNR\n")
            for i in range(results['frame_count']):
                f.write(f"{i}, {results['per_frame']['y_psnr'][i]:.2f}, "
                        f"{results['per_frame']['u_psnr'][i]:.2f}, "
                        f"{results['per_frame']['v_psnr'][i]:.2f}, "
                        f"{results['per_frame']['weighted_psnr'][i]:.2f}\n")
    
    print(f"结果已保存到: {output_path}")

def main():
    """
    命令行接口
    """
    parser = argparse.ArgumentParser(description='计算 YUV420 文件的 PSNR')
    parser.add_argument('--gt', required=False, help='GT (ground truth) YUV 文件路径')
    parser.add_argument('--pred', required=False, help='生成 (predicted) YUV 文件路径')
    parser.add_argument('--width', type=int, required=True, help='视频宽度')
    parser.add_argument('--height', type=int, required=True, help='视频高度')
    parser.add_argument('--frames', type=str, default='all', 
                       help='要计算的帧范围，如 "0-49" 或 "0,10,20" 或 "all" (默认)')
    parser.add_argument('--output', type=str, default='psnr_results.txt',
                       help='输出结果文件路径 (默认: psnr_results.txt)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    parser.add_argument('--bit', type=int, choices=[8, 10], default=8,
                       help='YUV 位深 (8 或 10，默认: 8)')
    
    args = parser.parse_args()
    
    # 解析帧范围
    frame_indices = None
    if args.frames != 'all':
        if '-' in args.frames:
            # 格式: "start-end"
            start, end = map(int, args.frames.split('-'))
            frame_indices = range(start, end + 1)
        elif ',' in args.frames:
            # 格式: "frame1,frame2,frame3"
            frame_indices = list(map(int, args.frames.split(',')))
        else:
            # 单个帧
            frame_indices = [int(args.frames)]


    prefix = "/gemini/space/yifq/xjy/results/all_frames_yuv_original"
    recon_prefix = "/gemini/space/yifq/xjy/results/all_frames_yuv_generated"
    for item in os.listdir(prefix):
        print(f"\n处理文件: {item}")
        gt_path = os.path.join(prefix, item)
        pred_path = os.path.join(recon_prefix, item)
        try:
            # 计算 PSNR
            results = calculate_yuv_psnr(
                gt_path,
                pred_path,
                args.width,
                args.height,
                frame_indices,
                args.bit
            )
            
            # 打印结果
            print_psnr_results(results, args.verbose)
            
            # 保存结果
            save_psnr_results(results, args.output)            
        except Exception as e:
            print(f"错误: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
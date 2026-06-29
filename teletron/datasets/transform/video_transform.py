import copy
import math
import random
import json
import scipy
import numpy as np
import torch
import os
from teletron.utils import video_utils
import torch.nn.functional as F
from einops import rearrange
from math import floor, ceil
from func_timeout import func_set_timeout
import cv2
import av


def random_dropout_canny(canny_array, dropout_rate):
    """
    在canny数组中对值为1的元素随机置零
    
    参数:
        canny_array: 形状为(T, height, width)的数组, 只包含0和1
        dropout_rate: 要置零的比例(0到1之间)
        
    返回:
        处理后的数组
    """
    result = canny_array.copy()
    ones_mask = (result == 1)
    random_values = np.random.random(result.shape)
    to_drop = (ones_mask & (random_values < dropout_rate))
    result[to_drop] = 0
    return result

def resize_canny_direct(canny_array, target_size=(448, 832)):
    """
    直接调整canny数组大小
    Args:
        canny_array: (t, h, w) 的numpy数组
        target_size: 目标大小 (height, width)
    Returns:
        resized_canny: (t, target_h, target_w) 的numpy数组
    """
    canny_tensor = torch.from_numpy(canny_array).float()
    canny_tensor = canny_tensor.unsqueeze(1)
    # 使用最邻近插值调整大小
    resized = torch.nn.functional.interpolate(canny_tensor, size=target_size, mode='nearest')
    resized_canny = resized.squeeze(1).numpy()
    # 重新二值化
    resized_canny = (resized_canny > 0.4).astype(np.uint8)
    
    return resized_canny

def resize_canny_with_pyav(frames_array, target_size=(448, 832)):
    """
    对视频帧进行缩放操作
    
    Args:
        frames_array: numpy数组, 形状为 (T, H, W, 3)
        target_size: 目标尺寸 (height, width)
    
    Returns:
        resized_frames: 缩放后的视频帧，形状为 (T, new_height, new_width, 3)
    """

    T, orig_height, orig_width, _ = frames_array.shape
    resized_frames = []
    for i in range(T):
        frame = av.VideoFrame.from_ndarray(frames_array[i], format='rgb24')
        resized_frame = frame.reformat(width=target_size[1], height=target_size[0], interpolation=av.video.reformatter.Interpolation.LANCZOS)
        resized_frame_array = resized_frame.to_ndarray(format='rgb24')
        resized_frames.append(resized_frame_array)
    resized_frames = np.stack(resized_frames, axis=0)
    return resized_frames


class ConditionGenerator:
    def __init__(
        self,
        cn_keys=[],
    ):
        self.cn_keys = cn_keys
    
    def get_canny(self, data_dict):
        def binary_array_from_rgb(canny_images):
            """
            对缩放后的视频帧进行重新二值化
            
            Args:
                canny_images: numpy数组，形状为 (T, new_height, new_width, 3)
            
            Returns:
                binary: 二值化后的canny数组，形状为 (T, new_height, new_width)
            """
            gray_images = np.mean(canny_images, axis=3)
            gray_uint8 = gray_images.astype(np.uint8)
            binary = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for gray in gray_uint8]
            binary = np.stack(binary).astype(bool)
            return binary
        
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        canny = data_dict['canny'].reshape((-1, height, width))

        canny_images = []
        
        # TODO: 抽样部分
        # gaussian_indexes = [14, 22, 26, 30, 33, 36, 39, 42, 45, 49, 53, 61]
        gaussian_indexes = range(0, len(sample_indexes))

        for i, idx in enumerate(sample_indexes):
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if i in gaussian_indexes:
                canvas[canny[idx] == 1] = 255
            canny_images.append(canvas)
        canny_images = np.stack(canny_images, axis=0)
        del data_dict["canny"]

        # 先下采样(传输用)
        downsample_stride = round(height / 480 * 4)
        canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(height//downsample_stride, width//downsample_stride))

        binary_array = binary_array_from_rgb(canny_images)
        canny_images = np.stack([np.stack([binary.astype(np.uint8)*255]*3, axis=-1) for binary in binary_array], axis=0)

        # 再上采样回去
        canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(height, width))
        
        canny_images = (
            torch.from_numpy(canny_images).permute(0, 3, 1, 2).contiguous()
        )
        data_dict["canny_images"] = canny_images
        return data_dict
    
    def get_masked_canny(self, data_dict):
        def binary_array_from_rgb(canny_images):
            """
            对缩放后的视频帧进行重新二值化
            
            Args:
                canny_images: numpy数组，形状为 (T, new_height, new_width, 3)
            
            Returns:
                binary: 二值化后的canny数组，形状为 (T, new_height, new_width)
            """
            gray_images = np.mean(canny_images, axis=3)
            gray_uint8 = gray_images.astype(np.uint8)
            binary = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for gray in gray_uint8]
            binary = np.stack(binary).astype(bool)
            return binary

        # def motion_abs_simple(images, num_segments):
        #     if num_segments == 0:
        #         return [0, len(images) - 1]
            
        #     T, C, H, W = images.shape
        #     weights = torch.tensor([0.299, 0.587, 0.114], device=images.device).view(1, 3, 1, 1)
        #     gray_images = (images.float() * weights).sum(dim=1, keepdim=True)  # T1HW
        #     motion_diffs = []
        #     for t in range(T - 1):
        #         diff = torch.abs(gray_images[t] - gray_images[t + 1]).sum()
        #         motion_diffs.append(diff.item())
        #     cumulative_motion = np.cumsum([0] + motion_diffs)
        #     total_diff = cumulative_motion[-1]

        #     segment_target = total_diff / num_segments
        #     segment_indices = [0,  len(images) - 1]
        #     current_sum = 0
            
        #     for i in range(T - 1):
        #         current_sum += motion_diffs[i]
                
        #         # 判断是否需要分割
        #         if current_sum >= segment_target:
        #             # 找到使当前段最接近目标值的位置
        #             if i == 0 or (abs(current_sum - segment_target) < 
        #                         abs((current_sum - motion_diffs[i-1]) - segment_target)):
        #                 segment_indices.append(i + 1)
        #             else:
        #                 segment_indices.append(i)
        #             current_sum = 0
                    
        #         # 如果已经找到足够的分段，跳出
        #         if len(segment_indices) == num_segments:
        #             break
        #     segment_indices = sorted(set(segment_indices))
        #     return segment_indices
        

        def motion_abs_simple(images, num_segments):
            """
            基于累积运动量的视频分段
            Args:
                images: torch.Tensor, shape (T, C, H, W) 或 (T, H, W, C)
                num_segments: int, 中间需要选择的分段点数量（总段数 = num_segments + 1）
            Returns:
                List[int]: 分段索引，包含首尾帧索引 [0, ..., T-1]
            """
            # 统一转换为 (T, C, H, W) 格式
            if images.dim() == 4:
                if images.shape[-1] == 3:  # THWC格式
                    images = images.permute(0, 3, 1, 2)  # 转换为TCHW
            
            T, C, H, W = images.shape
            
            # 特殊情况处理
            if T <= 2 or num_segments <= 0:
                return [0, T-1]
            
            if num_segments >= T - 1:
                # 如果要分的段数超过帧数，返回所有帧索引
                return list(range(T))
            
            # 转换为灰度图
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
            
            # 计算累积运动量
            cumulative_motion = np.cumsum([0] + motion_diffs)
            total_motion = cumulative_motion[-1]
            
            # 如果总运动量太小，均匀分段
            if total_motion < 1e-6:
                step = (T - 1) / (num_segments + 1)
                segment_indices = [0] + [int(i * step) for i in range(1, num_segments + 1)] + [T-1]
                return [int(idx) for idx in segment_indices]
            
            # 根据累积运动量均匀选择中间分段点
            segment_indices = [0]  # 固定包含首帧
            
            # 计算每个分段点应达到的累积运动量目标
            # 注意：总段数 = num_segments + 1，所以有num_segments个分段点需要选择
            # 将总运动量均匀分为(num_segments + 1)段，选择中间的num_segments个分割点
            segment_targets = [(i + 1) * total_motion / (num_segments + 1) 
                            for i in range(num_segments)]
            
            # 为每个目标值找到最接近的帧索引
            current_idx = 0
            for target in segment_targets:
                # 在累积运动量中找到第一个超过目标的索引
                while current_idx < T and cumulative_motion[current_idx] < target:
                    current_idx += 1
                
                # 确保索引在有效范围内且不重复
                current_idx = min(current_idx, T - 1)
                if current_idx <= segment_indices[-1]:
                    current_idx = segment_indices[-1] + 1
                
                # 如果还有空间，添加该索引
                if current_idx < T and current_idx not in segment_indices:
                    segment_indices.append(current_idx)
            
            # 确保包含尾帧
            if segment_indices[-1] != T - 1:
                segment_indices.append(T - 1)
            
            # 去重并确保排序
            segment_indices = sorted(set(segment_indices))
            
            return segment_indices


        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        canny = data_dict['canny'].reshape((-1, height, width))
        canny_cnt = data_dict["canny_cnt"]
        canny_images = []
        
        for idx in sample_indexes:
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            canvas[canny[idx] == 1] = 255
            canny_images.append(canvas)
        canny_images = np.stack(canny_images, axis=0)
        if canny_cnt == -1:
            gaussian_indexes = []
        else:
            gaussian_indexes = motion_abs_simple(data_dict["images"], canny_cnt)
        for idx in range(0, len(canny_images)): 
            if idx not in gaussian_indexes: canny_images[idx] = 0
        
        del data_dict["canny"]

        # 先下采样(传输用)
        downsample_stride = round(height / 480 * 4)
        canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(height//downsample_stride, width//downsample_stride))

        binary_array = binary_array_from_rgb(canny_images)
        canny_images = np.stack([np.stack([binary.astype(np.uint8)*255]*3, axis=-1) for binary in binary_array], axis=0)

        # 再上采样回去
        canny_images = resize_canny_with_pyav(frames_array=canny_images, target_size=(height, width))
        
        canny_images = (
            torch.from_numpy(canny_images).permute(0, 3, 1, 2).contiguous()
        )
        data_dict["canny_images"] = canny_images
        return data_dict

    def get_masked_video(self, data_dict):
        images = data_dict["images"]  # [T, C, H, W]
        T, C, H, W = images.shape
        
        # 初始化mask为全1（不遮挡）
        mask = torch.ones((T, 1, H, W), device=images.device)
        mask[[0, -1]] = 0
        data_dict["masked_images"] = images * mask
        return data_dict
    

    def __call__(self, data_dict):
        if "canny" in self.cn_keys:
            data_dict = self.get_canny(data_dict)
        if "masked_canny" in self.cn_keys:
            data_dict = self.get_masked_canny(data_dict)
        if "masked_images" in self.cn_keys:
            data_dict = self.get_masked_video(data_dict)
        # print("##########################", data_dict["canny_images"].shape, data_dict["masked_images"].shape)
        return data_dict


class MaskGenerator:
    def __init__(self, mask_ratios, min_clear_ratio=0.0, max_clear_ratio=1.0):
        valid_mask_names = [
            "t2v",
            "i2v",
            "clear",
            "transition",
            "continuation",
            "random",
            "f1fn2v"
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        self.mask_ratios = mask_ratios
        self.min_clear_ratio = min_clear_ratio
        self.max_clear_ratio = max_clear_ratio

    def get_mask(self, num_frames, height=None, width=None):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break
        num_select = random.randint(floor(num_frames * self.min_clear_ratio),
                                    ceil(num_frames * self.max_clear_ratio))

        if height is not None and width is not None:
            mask = torch.ones(size=(num_frames, 1, height, width), dtype=torch.float32)
        else:
            mask = torch.ones(num_frames, dtype=torch.float32)

        if num_frames <= 1:
            return mask
        if mask_name == "t2v":
            return mask
        elif mask_name == "i2v":
            mask[0] = 0
        elif mask_name == "clear":
            mask[:] = 0
        elif mask_name == "transition":
            mask[0] = 0
            mask[-1] = 0
        elif mask_name == "continuation":
            mask[:num_select] = 0
        elif mask_name == "random":
            selected_indices = random.sample(range(num_frames), num_select)
            mask[selected_indices] = 0
        elif mask_name == "f1fn2v":
            selected_indices = random.sample(range(num_frames), num_select)
            mask[selected_indices] = 0
            mask[0] = 0
        return mask

class MaskProcesser:
    '''
    modified from open-sora-plan
    https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/utils/mask_utils.py
    '''
    def __init__(self, ae_stride_h=8, ae_stride_w=8, ae_stride_t=4, **kwargs):
        self.ae_stride_h = ae_stride_h
        self.ae_stride_w = ae_stride_w
        self.ae_stride_t = ae_stride_t
    
    def __call__(self, mask):
        T, _, H, W = mask.shape
        new_H, new_W = H // self.ae_stride_h, W // self.ae_stride_w
        mask = rearrange(mask, 't c h w -> (t c) 1 h w')
        mask = F.interpolate(mask, size=(new_H, new_W), mode='bilinear')
        mask = rearrange(mask, '(t c) 1 h w -> t c h w', t=T)
        # align with wan vae
        new_T = (T + 3) // self.ae_stride_t
        mask_first_frame = mask[0:1].repeat(self.ae_stride_t, 1, 1, 1).contiguous() 
        mask = torch.cat([mask_first_frame, mask[1:]], dim=0)
        # if T % 2 == 1:
        #     new_T = T // self.ae_stride_t + 1
        #     mask_first_frame = mask[0:1].repeat(self.ae_stride_t, 1, 1, 1).contiguous() 
        #     mask = torch.cat([mask_first_frame, mask[1:]], dim=0)
        # else:
        #     new_T = T // self.ae_stride_t
        mask = mask.view(new_T, self.ae_stride_t, new_H, new_W).contiguous()
        return mask


class GenerateRefImages:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)[:, None, None, None]
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images
        return data_dict
    

class GenerateRefImagesWithMask:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)
        self.mask_processer = MaskProcesser()

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames, height, width = ref_images.shape[0], ref_images.shape[-2], ref_images.shape[-1]
        mask = self.mask_generator.get_mask(num_frames, height=height, width=width)
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_mask"] = self.mask_processer((mask < 0.5).float())
        data_dict["ref_images"] = ref_images
        return data_dict
    

class GenerateRefImagesWithTimeMask:
    def __init__(self, mask_cfg=dict(), min_clear_ratio=0.0, max_clear_ratio=1.0):
        self.mask_generator = MaskGenerator(mask_cfg, min_clear_ratio, max_clear_ratio)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images
        data_dict["time_mask"] = mask
        return data_dict


class GenerateFirstRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict

class GenerateFirstAndLastRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        last_ref_image = copy.deepcopy(data_dict["images"][-1:, ...])
        data_dict["last_ref_image"] = last_ref_image
        return data_dict

class GenerateRepeatedFirstImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict


class GeneratePoseControlImages:
    def __init__(self):
        pass

class GenerateRawFirstRefImage:
    def __call__(self, data_dict):
        raw_first_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["raw_first_image"] = raw_first_image
        return data_dict

class GenerateRawFirstLastRefImage:
    def __call__(self, data_dict):
        raw_first_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["raw_first_image"] = raw_first_image
        raw_last_image = copy.deepcopy(data_dict["images"][-1:, ...])
        data_dict["raw_last_image"] = raw_last_image
        return data_dict
    
@func_set_timeout(60)
class SampleImages:
    def __init__(
        self,
        num_frames=1,
    ):
        self.num_frames = num_frames
    
    def sample_video(self, video, indexes, method=2):
        if method == 1:
            frames = video.get_batch(indexes)
            frames = (
                frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
            )
        elif method == 2:
            max_idx = indexes.max() + 1
            all_indexes = np.arange(max_idx, dtype=int)
            frames = video.get_batch(all_indexes)
            frames = (
                frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
            )
            frames = frames[indexes]
        else:
            assert False
        return frames

    @func_set_timeout(60)
    def __call__(self, data_dict):
        video = data_dict["video"]
        if self.num_frames > 1:
            sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
            # images = video.get_frames_at(sample_indexes.tolist()).data # 
            data_dict["sample_indexes"] = sample_indexes
            images = self.sample_video(video, sample_indexes, method=1)
            images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        else:
            images = np.array(video)
            images = torch.from_numpy(images).permute(2,0,1).unsqueeze(0).contiguous()
        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]

        frame_interval = data_dict["frame_interval"]
        sample_length = (num_frames - 1) * frame_interval + 1
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length - 1)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes
    
@func_set_timeout(60)
class SampleDynamicFPSVideo:
    def __init__(
        self,
        num_frames=1,
        max_frames=201,
        fps_config={"24": 1.0},
        default_fps=24,
    ):
        self.num_frames = num_frames
        self.fps_config = {}
        self.default_fps = default_fps
        self.max_frames = max_frames

        for k, v in fps_config.items():
            self.fps_config[int(k)] = v

        assert all(
            fps_ratio >= 0 for fps_ratio in self.fps_config.values()
        ), f"mask_ratio should be greater than or equal to 0, got {self.fps_config.values()}"
        assert all(
            fps_ratio <= 1 for fps_ratio in self.fps_config.values()
        ), f"mask_ratio should be less than or equal to 1, got {self.fps_config.values()}"
        # sum of mask_ratios should be 1
        assert math.isclose(
            sum(self.fps_config.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(self.fps_config.values())}"

    def __call__(self, data_dict):
        video = data_dict["video"]

        sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
        images = video.get_frames_at(sample_indexes.tolist()).data

        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        
        fps_type = random.random()
        prob_acc = 0.
        dst_fps = self.default_fps
        for fps, ratio in self.fps_config.items():
            prob_acc = prob_acc + ratio
            if fps_type < prob_acc:
                dst_fps = fps
                break
        
        
        data_dict["dst_fps"] = dst_fps # inject fps
        data_dict["frame_interval"] = int(self.default_fps // dst_fps) # inject frame interval
        
        native_fps = data_dict['fps']
        extract_frame_interval = native_fps / dst_fps
        this_video_length = valid_range[1] - valid_range[0]
        
        data_dict["dst_fps"] = dst_fps # inject fps

        num_frames = int(1 + (this_video_length - 1) / extract_frame_interval)
        num_frames = max(1, (num_frames // 4 * 4) - 3)

        start_idx = valid_range[0]

        indexes = [start_idx + round(i * extract_frame_interval) for i in range(num_frames)]
        if len(indexes) > self.max_frames:
            start = random.randint(0, len(indexes) - self.max_frames)
            indexes = indexes[start: start + self.max_frames]
        sample_indexes = np.array(indexes, dtype=int)        

        return sample_indexes
    
@func_set_timeout(60)
class SampleWholeVideo:
    def __init__(
        self,
        max_frames=145,
        base_fps=24,
        fps_list=[24, 12, 6]
    ):
        self.max_frames = max_frames
        self.base_fps = base_fps
        self.fps_list = sorted(fps_list, reverse=True)
        
    def __call__(self, data_dict):
        video = data_dict["video"]

        sample_indexes, frame_interval = self.get_sample_indexes(data_dict)
        images = video.get_frames_at(sample_indexes.tolist()).data
        
        data_dict["images"] = images
        data_dict["frame_interval"] = frame_interval
        return data_dict

    def get_sample_indexes(self, data_dict):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        
        length = valid_range[1] - valid_range[0]
        
        native_fps = data_dict['fps']
        native_seconds = length / native_fps
        
        dst_fps = self.fps_list[-1]
        for fps in self.fps_list:
            if (self.max_frames / fps) > native_seconds:
                dst_fps = fps
                break

        if dst_fps > self.base_fps:
            dst_fps = self.base_fps
        
        frame_interval = native_fps / dst_fps
        
        data_dict["dst_fps"] = dst_fps # inject fps

        num_frames = int(1 + (length - 1) / frame_interval)
        num_frames = max(1, (num_frames // 4 * 4) - 3)

        start_idx = valid_range[0]

        indexes = [start_idx + round(i * frame_interval) for i in range(num_frames)]
        sample_indexes = np.array(indexes, dtype=int)        

        return sample_indexes, max(1, int(self.base_fps // dst_fps))
    
    
@func_set_timeout(60)
class SampleImageVideo:
    def __call__(self, data_dict):
        video = data_dict["video"]
        video_length = data_dict["video_info"][0]
        if video_length > 1:
            sample_indexes = self.get_sample_indexes(data_dict, video_length)
            images = video.get_frames_at(sample_indexes.tolist()).data
        else:
            images = np.array(video)
            images = torch.from_numpy(images).permute(2,0,1).unsqueeze(0).contiguous()
        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]

        frame_interval = data_dict["frame_interval"]
        sample_length = (num_frames - 1) * frame_interval + 1
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length - 1)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes

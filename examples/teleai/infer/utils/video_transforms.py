import copy
import random
import numpy as np
import torch
import os
from utils.utils import sample_video
from utils.pipeline_pose import draw_smpl_body
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import cv2

class ConditionGenerator:
    def __init__(
        self,
        type,
        cn_keys=[],
        mask_ratios=dict(),
    ):
        self.cn_keys = cn_keys

    def get_poses(self, data_dict):
        rand_number = data_dict['rand_number']
        
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        joints2ds_scores = data_dict["joint_uncertainties"]
        input_cn_images = []
        for i, idx in enumerate(sample_indexes):
            filtered_bboxes = []
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if i == 0 or i == len(sample_indexes) - 1:
                input_cn_images.append(Image.fromarray(canvas))
                continue
            try:
                for person_idx in range(len(joints2ds[idx])):
                    joints2d = joints2ds[idx][person_idx]
                    joints2d_score = joints2ds_scores[idx][person_idx]
                    if np.isnan(joints2d).any() or (joints2d == -1).all():
                        continue
                    if rand_number <= 0.5:
                        if (box[3] - box[1]) < height / 8:
                            continue  # 高度不足H/8，直接过滤
                    else:
                        if (box[3] - box[1]) < height / 5:
                            continue  # 高度不足H/5，直接过滤
                    canvas = draw_smpl_body(
                        canvas,
                        joints2d,
                        stickwidth=int(height / 350),
                        box=None,
                        scores=joints2d_score,
                        score_thre=600,
                    )
                input_cn_images.append(Image.fromarray(canvas))
            except Exception as e:
                input_cn_images.append(Image.fromarray(canvas))
        input_cn_images = np.stack(input_cn_images, axis=0)
        input_cn_images = (
            torch.from_numpy(input_cn_images).permute(0, 3, 1, 2).contiguous()
        )
        data_dict["poses_images"] = input_cn_images

    def get_canny(self, data_dict):
        rand_number = data_dict['rand_number']
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        try:
            canny = data_dict['canny'].reshape((-1, height, width))
        except Exception as e:
            print(f"error while loading canny from data dict, {e}")
            canny = np.zeros((data_dict["video_length"], height, width), dtype=bool)
        canny_images = []
        for i, idx in enumerate(sample_indexes):
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if i == 0 or i == len(sample_indexes) - 1:
                canny_images.append(canvas)
                continue
            try:
                if rand_number <= 0.5: # level 2, 压缩率更小
                    curve = get_curve(canny[idx], 100)
                else:
                    curve = get_curve(canny[idx], 150)
                canvas[curve == 1] = 255
            except Exception as e:
                print(f"error while putting canny on the canvas, {e}")
            canny_images.append(canvas)
        
        canny_images = np.stack(canny_images, axis=0)
        canny_images = (
            torch.from_numpy(canny_images).permute(0, 3, 1, 2).contiguous()
        )
        data_dict["canny_images"] = canny_images

    def get_flow(self, data_dict):
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        try:
            flow = data_dict['flow'].astype(np.float16, copy = False)
            flow_indexes = data_dict['flow_indexes']
        except Exception as e:
            print(f"error while loading flow from data dict, {e}")
            pass
        flow_images = []
        for i, idx in enumerate(sample_indexes):
            if i == 0 or i == len(sample_indexes) - 1:
                canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
                flow_images.append(canvas)
                continue
            canvas = np.ones(shape=(height, width, 3), dtype=np.uint8)
            canvas *= 255 # 白色底图            
            try:
                canvas = flow_to_arrow_grid(canvas, flow[idx], flow_indexes)
            except Exception as e:
                print(f"error while putting flow on the canvas, {e}")
                pass
            flow_images.append(canvas)
        del data_dict['flow']
        del flow
        flow_images = np.stack(flow_images, axis=0)
        flow_images = (
            torch.from_numpy(flow_images).permute(0, 3, 1, 2).contiguous()
        )
        data_dict["flow_images"] = flow_images

    def apply_random_size_mask(self, data_dict, min_ratio=0.5, max_ratio=0.9):
        """
        对视频序列随机选择一个可变大小的区域进行掩码（置零），并以0.5的概率决定是否对当前帧进行mask
        
        Args:
            data_dict (dict): 包含视频序列的字典，键 "images" 的形状为 [T, C, H, W]
            min_ratio (float): 最小掩码区域相对于原图的比例
            max_ratio (float): 最大掩码区域相对于原图的比例
        
        Returns:
            dict: 更新后的 data_dict，新增 "masked_images" 和 "mask" 字段
        """
        images = data_dict["images"]  # [T, C, H, W]
        T, C, H, W = images.shape
        
        # 初始化mask为全1（不遮挡）
        mask = torch.ones((T, 1, H, W), device=images.device)
        
        for t in range(T):  # 遍历时间步
            # 以0.5的概率决定是否对该帧进行mask v3 除首尾帧全给mask
            if t==0 or t==T-1:
                continue
                
            # 随机确定mask区域的大小（原图的0.5-0.9倍）
            mask_h = H
            mask_w = W
            
            # 随机选择掩码的左上角坐标（确保不越界）
            x = random.randint(0, W - mask_w)
            y = random.randint(0, H - mask_h)
            
            # 遮挡区域置0
            mask[t, :, y:y+mask_h, x:x+mask_w] = 0
        
        # 应用掩码
        masked_images = images * mask
        
        # 更新data_dict
        data_dict["masked_images"] = masked_images
        

    def __call__(self, data_dict, use_vae=True):
        if "poses" in self.cn_keys:
            self.get_poses(data_dict)
        if "canny" in self.cn_keys:
            self.get_canny(data_dict)
        if "flow" in self.cn_keys:
            self.get_flow(data_dict)
        if use_vae:
            self.apply_random_size_mask(data_dict)
        return data_dict


def fitted_curves(img, n):
    def bezier_interp(P0, P1, P2, P3, num=20):
        t = np.linspace(0, 1, num).reshape(-1, 1)
        return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
    # 提取所有轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total_points_needed = n * 4
    output = np.zeros_like(img)
    if len(contours) > n:
        contour_areas = [(cv2.contourArea(c), c) for c in contours]
        contour_areas.sort(key=lambda x: x[0], reverse=True)
        contours = [c for _, c in contour_areas[:n]]
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 4:
            continue
        
        idx = np.linspace(0, len(cnt) - 1, total_points_needed, dtype=int)
        cnt = cnt[idx]
        # 将轮廓分成连续段，每4个点拟合一段贝塞尔曲线
        for i in range(0, len(cnt) - 3, 3):  # 步长3，保证段落连接
            P0, P1, P2, P3 = cnt[i:i+4]
            curve = bezier_interp(P0, P1, P2, P3, num=30)
            curve = np.round(curve).astype(int)

            for j in range(len(curve) - 1):
                pt1 = tuple(curve[j])
                pt2 = tuple(curve[j + 1])
                cv2.line(output, pt1, pt2, 255, 1)
    return output

def get_curve(canny, n):
    output = np.zeros_like(canny)
    try:
        output = fitted_curves(canny.astype(np.uint8), n)
    except Exception as e:
        print(e)
    return output

def flow_to_arrow_grid(image, flow, flow_indexes, arrow_scale = 5):
    # image: [H, W, 3]
    # flow: [H, W, 2]
    h, w = image.shape[:2]
    num_y, num_x = flow.shape[:2]
    
    # 创建图形 (无边框)
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)  # 精确控制输出尺寸
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # 图像坐标系y轴向下
    ax.axis('off')
    
    # 显示背景图像
    ax.imshow(image, extent=[0, w, h, 0])
    
    # 生成网格点
    y_coords = np.arange(num_y)
    x_coords = np.arange(num_x)
    
    # 绘制箭头
    for y in y_coords:
        for x in x_coords:
            dx, dy = flow[y, x]
            ax.arrow(flow_indexes[1][x], flow_indexes[0][y], dx*arrow_scale, dy*arrow_scale, 
                    color='black', head_width=5, head_length=20, width=2)
    
    # 转换为numpy数组
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    img_array = img_array.reshape((h, w, 4))
    a = img_array[:,:,0]
    r = img_array[:,:,1]
    g = img_array[:,:,2]
    b = img_array[:,:,3]
    rgb_array = np.dstack((r, g, b))
    plt.close(fig)
    return rgb_array


class SampleImages:
    def __init__(
        self,
        type,
        num_frames=1,
    ):
        self.num_frames = num_frames

    def __call__(self, data_dict):
        video = data_dict["video"]
        sample_indexes = self.get_sample_indexes(data_dict, self.num_frames)
        data_dict["sample_indexes"] = sample_indexes
        images = sample_video(video, sample_indexes, method=2)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        data_dict["images"] = images
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])

        valid_range[0] = valid_range[0]
        valid_range[1] = valid_range[1]

        video_length = valid_range[1] - valid_range[0]

        frame_interval = data_dict["frame_interval"]
        sample_length = (num_frames - 1) * frame_interval + 1
        start_idx = valid_range[0]
        sample_indexes = np.linspace(
            start_idx, min(start_idx + sample_length, valid_range[1]), num_frames, dtype=int
        )
        print(sample_indexes)
        return sample_indexes


class GenerateFirstAndLastRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["raw_first_image"] = first_ref_image
        last_ref_image = copy.deepcopy(data_dict["images"][-1:, ...])
        data_dict["raw_last_image"] = last_ref_image
        return data_dict


class PackInputs:
    def __init__(self, type, image_keys, embedding_keys, dst_size, mean=0.5, std=0.5, crop_keys=[]) -> None:
        self.dst_size = dst_size
        self.image_keys = image_keys
        self.embedding_keys = embedding_keys
        self.crop_keys = crop_keys
        self.mean = mean
        self.std = std
        

    def __call__(self, data_dict, inference=False):
        data_dict = self.resize_and_crop(data_dict) # crop_keys
        for image_key in self.image_keys: # norm
            data_dict[image_key] = (
                (data_dict[image_key] / 255.0) - self.mean
            ) / self.std
        
        input_dict = {}
        for embed_key in self.embedding_keys + self.crop_keys + self.image_keys:
            input_dict[embed_key] = data_dict[embed_key]
        input_dict["sample_indexes"] = data_dict["sample_indexes"]
        return input_dict

    def resize_and_crop(self, data_dict):
        new_height, new_width, dst_height, dst_width = self.get_new_height_width(
            data_dict
        )
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        for image_key in self.crop_keys:
            images = data_dict[image_key]
            images = F.resize(
                images, (new_height, new_width), InterpolationMode.BILINEAR
            )
            images = F.crop(images, y1, x1, dst_height, dst_width)
            data_dict[image_key] = images
        return data_dict

    def get_new_height_width(self, data_dict):
        height = data_dict["video_height"]
        width = data_dict["video_width"]
        dst_width, dst_height = self.dst_size
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        return new_height, new_width, dst_height, dst_width

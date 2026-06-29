from typing import Sequence, Union
import numpy as np
import torch
import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from PIL import Image
from io import BytesIO



def is_seq_of(seq, expected_type):
    """
    检查给定的序列是否包含特定类型的元素。

    :param seq: 要检查的序列
    :param expected_type: 期望的元素类型
    :return: 如果所有元素都是指定类型，则返回True，否则返回False
    """
    if not isinstance(seq, (list, tuple)):
        return False
    return all(isinstance(item, expected_type) for item in seq)

class PackInputs:
    def __init__(self, image_keys, embedding_keys, dst_size, mean=0.5, std=0.5, crop_keys=[]) -> None:
        self.dst_size = dst_size
        self.image_keys = image_keys
        self.embedding_keys = embedding_keys
        self.crop_keys = crop_keys
        self.mean = mean
        self.std = std
        

    def __call__(self, data_dict):
        data_dict = self.resize_and_crop(data_dict) # crop_keys
        for image_key in self.image_keys: # norm
            data_dict[image_key] = (
                (data_dict[image_key] / 255.0) - self.mean
            ) / self.std
        
        input_dict = {}
        input_dict["struct_prompt"] = data_dict["struct_prompt"]
        input_dict["short_prompt"] = data_dict["short_prompt"]
        input_dict["dense_prompt"] = data_dict["dense_prompt"]
        input_dict["frame_interval"] = data_dict["frame_interval"]
        for embed_key in self.embedding_keys + self.crop_keys + self.image_keys:
            input_dict[embed_key] = data_dict[embed_key]
        return input_dict

    def resize_and_crop(self, data_dict):
        new_height, new_width, dst_height, dst_width = self.get_new_height_width(
            data_dict
        )
        # import ipdb;ipdb.set_trace()
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
        if "bucket_index" in data_dict.keys():
            dst_width, dst_height = data_dict["video_info"]
        else:
            dst_width, dst_height = self.dst_size
            
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        return new_height, new_width, dst_height, dst_width


# 临时， 没想好怎么改
# 返回 低分辨率+高分辨率的 输入
# 同时加入关键帧编解码aug
class PackInputs_TMP:
    def __init__(self, image_keys, embedding_keys, dst_size, mean=0.5, std=0.5, crop_keys=[]) -> None:
        self.dst_size = dst_size
        self.image_keys = image_keys
        self.embedding_keys = embedding_keys
        self.crop_keys = crop_keys
        self.mean = mean
        self.std = std    

    def __call__(self, data_dict):
        data_dict = self.resize_and_crop(data_dict) # crop_keys
        for image_key in self.image_keys: # norm
            data_dict[image_key] = (
                (data_dict[image_key] / 255.0) - self.mean
            ) / self.std

            if image_key + '_ds' in data_dict.keys():
                data_dict[image_key + '_ds'] = (
                (data_dict[image_key + '_ds'] / 255.0) - self.mean
            ) / self.std
                
        
        input_dict = {}
        input_dict["struct_prompt"] = data_dict["struct_prompt"]
        input_dict["short_prompt"] = data_dict["short_prompt"]
        input_dict["dense_prompt"] = data_dict["dense_prompt"]
        input_dict["frame_interval"] = data_dict["frame_interval"]
        for embed_key in self.embedding_keys + self.crop_keys + self.image_keys:
            input_dict[embed_key] = data_dict[embed_key]
            if embed_key + "_ds" in data_dict.keys():
                input_dict[embed_key + "_ds"] = data_dict[embed_key + "_ds"]
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
            images_downsample = F.resize(
                images, (dst_height // 2, dst_width // 2), InterpolationMode.BILINEAR
            )
            data_dict[image_key] = images
            data_dict[image_key + "_ds"] = images_downsample
        
        return data_dict

    def get_new_height_width(self, data_dict):
        height = data_dict["video_height"]
        width = data_dict["video_width"]
        if "bucket_index" in data_dict.keys():
            dst_width, dst_height = data_dict["video_info"]
        else:
            dst_width, dst_height = self.dst_size
            
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        return new_height, new_width, dst_height, dst_width

from typing import List
import numpy as np
from .utils import image_utils
from .base_dataset import BaseDataset
from teleai_data_tool.schema.dataset import ClipsDataset as _ClipsDataset
from teleai_data_tool.schema.clip import Clip
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
import json
from collections import defaultdict
from scipy.sparse import load_npz
from tqdm import tqdm
import random


class ClipDatasetEasy(BaseDataset):
    def __init__(
        self,
        data_path_list,
        transforms,
        filter_cfg=dict(),
        enable_bucket_index=False,
        serialize_data=True,
    ) -> None:
        self.data_path_list = data_path_list
        self.enable_bucket_index = enable_bucket_index
        self.bucket_index_list = None
        super().__init__(
            ann_file="",
            serialize_data=serialize_data,
            test_mode=False,
            lazy_init=False,
            max_refetch=10,
            pipeline=transforms,
            filter_cfg=filter_cfg,
        )
        self.file_client = FileClient()


    def load_data_list(self) -> List[dict]:
        data_list = []
        for data_path in tqdm(self.data_path_list):
            with open(data_path) as f:
                dataset = json.load(f)
            for clip in dataset["clips"]:
                # no structure
                data_list.append(clip)
        return data_list

    def get_bucket_index_list(self):
        return self.bucket_index_list


    def filter_data(self):
        dst_size = self.filter_cfg.get("dst_size", (720, 480))
        dst_num_frames = self.filter_cfg.get("dst_num_frames", 81)
        dst_fps = self.filter_cfg.get("dst_fps", [10, 15, 25, 30])

        self.buckets_size = self.filter_cfg.get("buckets_size", [(640, 368), (864, 480), (960, 528)])
        self.buckets_size_ratio = self.filter_cfg.get("buckets_size_ratio", [0.3, 0.4, 0.3])
        shape_list = []
        shape_num_map = defaultdict(int)
        for shape in self.buckets_size:
            shape_list.append(f"{shape[0]}__{shape[1]}")

        
        valid_data_list =  []
        too_short, too_small = 0, 0
        for clip in self.data_list:
            if clip["video_length"] < max(1, round(clip['fps'] / min(dst_fps))) * dst_num_frames:
                too_short += 1
                continue
            # size
            if clip["video_height"] < dst_size[1] or clip["video_width"] < dst_size[0]:
                too_small += 1
                continue
                
            if self.enable_bucket_index:
                # 根据满足的分辨率，随机选择bucket
                valid_bucket_size, valid_bucket_ratio = [], []
                for id_bucket, bucket_size in enumerate(self.buckets_size):
                    if clip["video_height"] < bucket_size[1] or clip["video_width"] < bucket_size[0]:
                        continue
                    else:
                        valid_bucket_size.append(bucket_size)
                        valid_bucket_ratio.append(self.buckets_size_ratio[id_bucket])
                
                if len(valid_bucket_size) == 0:
                    too_small += 1
                    continue
                
                # 不需要对valid_bucket_ratio归一化，random.choices 会做    
                dst_width, dst_height = random.choices(valid_bucket_size, weights=valid_bucket_ratio)[0]
                # dst_width, dst_height = image_utils.get_image_size(
                #     (clip["video_width"], clip["video_height"]),
                #     (sampler_size[0], sampler_size[1]),
                #     mode="area",
                #     multiple=16,
                # )
                video_info = f"{dst_width}__{dst_height}"
                clip["bucket_index"] = shape_list.index(video_info)
                shape_num_map[video_info] += 1
                clip["video_info"] = (dst_width, dst_height)
            valid_data_list.append(clip)
        
        if self.enable_bucket_index:
            bucket_index_list = defaultdict(list)
            for i, clip in enumerate(valid_data_list):
                bucket_index_list[clip["bucket_index"]].append(i)
            self.bucket_index_list = [
                np.array(item) for item in bucket_index_list.values()
            ]
        
        print(
            f"finish filter dataset, from {len(self.data_list)} to {len(valid_data_list)} \n"
            f"too short data {too_short} \n"
            f"too small data {too_small} \n"
            f"bucket shape: {shape_num_map}" 
        )
        return valid_data_list

    def get_data_info(self, idx):
        clip = super().get_data_info(idx)
        dst_fps = self.filter_cfg.get("dst_fps", [10, 15, 25, 30])

        try:
            video = self.file_client.get(clip["video_path"])
        except Exception as e:
            print("Failed to load video {}: {}".format(clip['video_path'], e))
            return None
        
        clip["video"] = video

        clip["frame_interval"] = max(1, round(clip['fps'] / random.choice(dst_fps)))
        
        random_number = np.random.rand()
        clip["random_number"] = random_number
        
        if "canny_path" in clip.keys():
            try:
                clip["canny"] = load_npz(clip['canny_path']).toarray()
                clip["canny_cnt"] = random.choice(self.filter_cfg.get("canny_cnt_buckets", [0])) # 首尾帧本身就有canny
            except Exception as e:
                return None
                # print("Failed to load canny {}, using default value: {}".format(clip['canny_path'], e))
                # # default value: [T, H, W], dtype = bool
                # video_length = len(clip["video"])
                # video_height = clip["video"].shape[1]
                # video_width = clip["video"].shape[2]
                # clip["canny"] = np.zeros((video_length, video_height, video_width), dtype=bool)
                # clip["canny_cnt"] = -1 # means no canny
        
        return clip

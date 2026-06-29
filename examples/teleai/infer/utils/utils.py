from torch import nn
import copy
import re
import base64
from io import BytesIO
from dataclasses import field, dataclass
from typing import List
import json
import torch
import numpy as np

def process_model(model, num_cns):
     # process model
    ori_layer_state_dict = model.patch_embedding.state_dict()
    weight_shape = list(ori_layer_state_dict["weight"].shape)
    weight_shape[1] = 36 + 16 * num_cns # plus control
    new_weight = ori_layer_state_dict["weight"].new_zeros(weight_shape)
    for i in range(num_cns):
        new_weight[:, 36 + i*16: 36 + (i + 1)*16] = copy.deepcopy(ori_layer_state_dict["weight"][:, 16:32]) # init with original weight
    ori_layer_state_dict["weight"] = new_weight
    model.patch_embedding = nn.Conv3d(in_channels=36 + 16 * num_cns, out_channels=5120, kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 14B
    # model.patch_embedding = nn.Conv3d(in_channels=36 + 16 * num_cns, out_channels=1536, kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 1.3B
    model.patch_embedding.load_state_dict(ori_layer_state_dict, strict=True)
    return model

def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)

def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict

def extract_json_from_markdown(markdown_text):
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    try:
        matches = re.findall(json_pattern, markdown_text)
        if not matches:
            matches = [markdown_text]
    except Exception as e:
        print(f"Error occurred while matching JSON pattern: {e}")
        matches = [markdown_text]

    json_data = []
    max_repair_rounds = 10

    for match in matches:
        repair_rounds = 0
        current_match = match
        while repair_rounds <= max_repair_rounds:
            try:
                data = json.loads(current_match)
                json_data.append(data)
                break
            except json.JSONDecodeError as e:
                repair_rounds += 1
                if repair_rounds <= max_repair_rounds:
                    current_match = current_match[: e.pos] + current_match[e.pos + 1 :]
                else:
                    break

    if json_data:
        return json_data[0]
    return None

def images_to_base_64(images):
    base64_images = []
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        base64_images.append(base64_encoded)
    return base64_images

@dataclass
class Caption:
    short_caption: List[str] = field(
        default_factory=list
    )  # the main content of the scene
    dense_caption: List[str] = field(
        default_factory=list
    )  # detail of scene
    frame_range: List[List[int]] = field(
        default_factory=list
    )  # the start and end frame id of this caption


def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = (
            frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        )
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        # frames = video.get_batch(all_indexes)
        # frames = (
        #     frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        # )
        frames = video[indexes]
    else:
        assert False
    return frames
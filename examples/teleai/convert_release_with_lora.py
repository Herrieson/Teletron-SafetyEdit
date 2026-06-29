import os
import re
import torch
import safetensors.torch
from collections import OrderedDict
import copy
import torch.nn as nn

def process_patch_embedding_state_dict(state_dict, lora_path):
    """直接在state_dict上修改patch_embedding的权重"""
    lora_state = torch.load(lora_path, map_location="cpu")
    if "state_dict" in lora_state:  # 兼容某些保存格式
        lora_state = lora_state["state_dict"]
    if "patch_embedding.weight" in lora_state:
        state_dict["patch_embedding.weight"] = lora_state["patch_embedding.weight"]
        state_dict["patch_embedding.bias"] = lora_state["patch_embedding.bias"]
    return state_dict

# model_path 是str或者list[str]
def get_normal_state_dict(model_path: str | list[str]):
    if isinstance(model_path, str):
        print(f"loading model from {model_path}")
        if model_path.endswith(".safetensors"):
            # 检查文件完整性
            with open(model_path, "rb") as f:
                header = f.read(20)
                if header == b'\x00' * 20:  # 全0头部表示文件损坏
                    raise ValueError(f"Corrupted safetensors file detected: {model_path}")
                if len(header) < 8:
                    raise ValueError(f"Truncated safetensors file: {model_path}")
            
            return safetensors.torch.load(open(model_path, "rb").read())
        else:
            return torch.load(model_path, map_location="cpu", weights_only=False)
    else:
        assert isinstance(model_path, list)
        state_dict = OrderedDict()
        for path in model_path:
            try:
                state_dict.update(get_normal_state_dict(path))
            except ValueError as e:
                print(f"⚠️  Skipping corrupted file: {e}")
                continue
        return state_dict

def update_state_dict(state_dict):
    output_state_dict = OrderedDict()
    replacement_rules = [
        (r'\.k\.','.key.'),
        (r'\.q\.','.query.'),
        (r'\.v\.','.value.'),
        (r'\.o\.','.out_proj.'),
        (r'\.norm_q\.','.norm_query.'),
        (r'\.norm_k\.','.norm_key.'),
        (r'\.k_img\.','.img_key.'),
        (r'\.v_img\.','.img_value.'),
        (r'\.norm_k_img\.','.norm_image_key.'),
        (r'^patch_embedding\.','patch_emb.'),
        (r'^time_embedding\.','time_emb.'),
        (r'^text_embedding\.','text_emb.'),
        (r'^time_projection\.','time_proj.'),
    ]
    for key, value in state_dict.items():
        for old, new in replacement_rules:
            if not key.startswith("compressor_up") and not key.startswith("compressor_down"):
                key = re.sub(old, new, key)
        output_state_dict[key] = value
    return output_state_dict

def save_teletron_release(state_dict, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
    with open(latest_file, "w") as f:
        f.write("release")
    release_dir = os.path.join(checkpoint_dir, "release")
    os.makedirs(release_dir, exist_ok=True)
    checkpoint_dir = os.path.join(release_dir, "mp_rank_00")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = "model_optim_rng.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    output_state_dict = OrderedDict({
            "args": None,  
            "checkpoint_version": 0.0,
            "model":  {k: v for k, v in state_dict.items()},
        })
    torch.save(output_state_dict, checkpoint_path)

def merge_lora_into_state_dict(state_dict, lora_path, rank=16, alpha=16):
    lora_state = torch.load(lora_path, map_location="cpu")
    if "state_dict" in lora_state:  # 兼容某些保存格式
        lora_state = lora_state["state_dict"]

    scale = alpha / rank

    # 收集 A, B
    lora_As, lora_Bs = {}, {}
    for k, v in lora_state.items():
        if k.endswith("lora_A.default.weight"):
            prefix = k[:-len("lora_A.default.weight")]
            lora_As[prefix] = v
        elif k.endswith("lora_B.default.weight"):
            prefix = k[:-len("lora_B.default.weight")]
            lora_Bs[prefix] = v

    for prefix, A in lora_As.items():
        if prefix not in lora_Bs:
            print(f"⚠️ Skip {prefix}, missing lora_B")
            continue

        B = lora_Bs[prefix]

        # 自动检测 A/B 组合方式
        merged = None
        if A.shape[0] == rank:  # [r, in]
            if B.shape[1] == rank:  # [out, r]
                merged = torch.matmul(B, A) * scale  # [out, in]
        if merged is None and B.shape[0] == rank:  # B 是 [r, in]
            if A.shape[1] == rank:  # A 是 [out, r]
                merged = torch.matmul(A, B) * scale  # [out, in]

        if merged is None:
            print(f"❌ Shape mismatch for {prefix}: A {A.shape}, B {B.shape}")
            continue

        base_key = prefix + "weight"
        if base_key not in state_dict:
            print(f"⚠️ Skip {prefix}, base weight not found")
            continue

        print(f"🔧 Merging LoRA into {base_key}, base {state_dict[base_key].shape}, delta {merged.shape}")
        state_dict[base_key] = state_dict[base_key] + merged

    return state_dict

if __name__ == "__main__":
    update_key_from_wan_to_teleai = True
    model_path = [
        os.path.join("/gemini/space/yifq/Wan-AI/Wan2.1-FLF2V-14B-720P", f) for f in os.listdir("/gemini/space/yifq/Wan-AI/Wan2.1-FLF2V-14B-720P") if f.endswith(".safetensors")
    ]
    model_path.sort()
    lora_path = "/gemini/space/yifq/Wan-AI/vae+canny/transformer_lora.pt"
    target_path = "/gemini/space/yifq/teletron-model/Wan2.1-FLF2V-14B-720P"
    
    # Compressor模块路径
    downsample_model_path = "/gemini/space/yifq/Wan-AI/vae+canny/downsample_model.pt"
    upsample_model_path = "/gemini/space/yifq/Wan-AI/vae+canny/upsample_model.pt"

    # 加载主模型权重
    state_dict = get_normal_state_dict(model_path)
    print(f"✅ successfully load model from {model_path}")

    # 修改patch_embedding权重
    print("🔧 Processing patch_embedding...")
    state_dict = process_patch_embedding_state_dict(state_dict, lora_path)

    # 加载compressor模块
    print("🔧 Loading compressor modules...")
    if os.path.exists(downsample_model_path):
        downsample_state = torch.load(downsample_model_path, map_location="cpu", weights_only=False)
        # 将downsample模块的权重添加到state_dict中
        for k, v in downsample_state.items():
            state_dict[f"compressor_down.{k}"] = v
        print(f"✅ successfully loaded downsample model from {downsample_model_path}")
    
    if os.path.exists(upsample_model_path):
        upsample_state = torch.load(upsample_model_path, map_location="cpu", weights_only=False)
        # 将upsample模块的权重添加到state_dict中
        for k, v in upsample_state.items():
            state_dict[f"compressor_up.{k}"] = v
        print(f"✅ successfully loaded upsample model from {upsample_model_path}")

    # 合并LoRA
    state_dict = merge_lora_into_state_dict(state_dict, lora_path, rank=16, alpha=16)
    print(f"✅ successfully merged lora {lora_path}")

    # 更新键名
    if update_key_from_wan_to_teleai:
        state_dict = update_state_dict(state_dict)
        print(f"✅ successfully updated state_dict")

    # 保存模型
    save_teletron_release(state_dict, target_path)
    print(f"✅ successfully saved merged model to {target_path}")
    print(f"📊 Final state_dict contains {len(state_dict)} parameters")
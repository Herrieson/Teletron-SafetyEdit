import os
import math

dst_size = (704, 384) # (dst // 8) % 8 == 0
buckets_size = [(704, 384), (832, 448), (1408, 768), (1984, 1088)] 
dst_num_frames = 77 # ((dst_num_frames - 1) // 4 + 1) % 4 == 0

config = dict(
    dataset=dict(
        type="ClipDatasetEasy",
        serialize_data=False,
        enable_bucket_index=True,
        data_path_list=[
            os.path.join("/gemini/space/yifq/yifq/code/scripts/jsons/crawl_0923_split", x) for x in os.listdir("/gemini/space/yifq/yifq/code/scripts/jsons/crawl_0923_split")
        ] + [
            os.path.join("/gemini/space/yifq/yifq/code/scripts/jsons/istock", x) for x in os.listdir("/gemini/space/yifq/yifq/code/scripts/jsons/istock")
        ],
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames = dst_num_frames,
            dst_fps = [10, 15, 25, 30],
            buckets_size = buckets_size,
            buckets_size_ratio = [0.2, 0.3, 0.3, 0.2],
        ),
        transforms=[
            dict(
                type="SampleImages",
                num_frames=dst_num_frames,
            ),
            dict(
                type="PromptGenerator",
                clean_prompt=True,
                default_prompt_prob=1.0, # all with default empty prompt for recon task
            ),
            dict(
                type="GenerateRawFirstLastRefImage",
            ),
            dict(
                type="ConditionGenerator",
                cn_keys=["canny", "masked_images"],
            ),
            dict(
                type="PackInputs",
                crop_keys=[ # 做crop
                    "images",
                    "canny_images",
                    "masked_images",
                    "raw_first_image", 
                    "raw_last_image"
                ],
                image_keys=[ # 做norm
                    "images",
                    "canny_images",
                    "masked_images",
                ],
                embedding_keys=[],
                dst_size=None, # 当bucket 打开后这条无用
            ),
        ],
    ),
    eval=dict(
        data_path_list=[
            os.path.join("/gemini/space/yifq/yifq/code/scripts/jsons/crawl_0923_split", x) for x in os.listdir("/gemini/space/yifq/yifq/code/scripts/jsons/crawl_0923_split")
        ] + [
            os.path.join("/gemini/space/yifq/yifq/code/scripts/jsons/istock", x) for x in os.listdir("/gemini/space/yifq/yifq/code/scripts/jsons/istock")
        ],
        eval_time_steps=[1000]
    ),
    sampler=dict(
        type="BucketSampler",
        shuffle=False,
        seed=42,
        drop_last=True,
        infinite=True,
    ),
    model_config=dict(
        dit=dict(
            type="ParallelTeleaiModel", # ParallelTeleaiModel
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36 + 16, # t2v:16 i2v:36, s2v: 16 * numof_s
                dim=1536, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=8960, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12, # 1.3B:12 10B:40 14B:40
                num_layers=30, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=True,  # fl2v: True
                has_compressor={"use":True, "up_T":True},
                has_quantizer=True,
            ),
        ),
        encoder=dict(
            type="teleai_encoder", # teleai_encoder
            encoder_schema=['context', 'latents', 'img_emb_y', 'img_clip_feature', "latents_canny_images", "latents_masked_images"],
            vae=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
                tiler_kwargs=dict(
                    tiled=False,  # 
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                ),
                torch_compile=False # must be false
            ),
            text_encoder=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/gemini/space/yifq/wrq/model_zoo/PAI/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                torch_compile=False
            ),
        ),
        ddit=dict(
            type="ParallelTeleaiLogitsModel", # ParallelTeleaiModel
            config=dict(
                has_image_input=True, # t2v:False i2v:True i2v Wan2.2:False
                patch_size=[1, 2, 2],
                in_dim=36, # t2v:16 i2v:36, s2v: 16 * numof_s
                dim=1536, # 1.3B:1536 10B:5120 14B:5120
                ffn_dim=8960, # 1.3B:8960 10B:13824 14B:13824
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12, # 1.3B:12 10B:40 14B:40
                num_layers=30, # 1.3B:30 10B:30 14B:40
                eps=1e-6,
                has_image_pos_emb=True,  # fl2v: True
            ),
        ),
    ),
)

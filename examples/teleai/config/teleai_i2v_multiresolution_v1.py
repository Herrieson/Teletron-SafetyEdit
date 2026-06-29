import os
import math

dst_size = (704, 384) # (dst // 8) % 8 == 0
buckets_size = [(704, 384), (832, 448), (1408, 768), (1984, 1088)] 
dst_num_frames = 29 # ((dst_num_frames - 1) // 4 + 1) % 4 == 0

config = dict(
    dataset=dict(
        type="ClipDatasetEasy",
        serialize_data=False,
        enable_bucket_index=True,
        data_path_list = [os.path.join("/gemini/platform/shared/xujy70/trainable_jsons", x) for x in os.listdir("/gemini/platform/shared/xujy70/trainable_jsons")],
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
        data_path_list = [os.path.join("/gemini/platform/shared/xujy70/trainable_jsons", x) for x in os.listdir("/gemini/platform/shared/xujy70/trainable_jsons")],
        eval_time_steps=[200,400,600,800,1000]
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
                in_dim=36 + 16 + 4, # t2v:16 i2v:36, s2v: 16 * numof_s
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
            # latents_canny_images： canny图过vae
            # latents_masked_images: 首尾帧mask后过vae
            encoder_schema=['context', 'latents', 'img_emb_y', 'img_clip_feature', "latents_canny_images", "latents_masked_images"],
            vae=dict(
                # path="/gemini/platform/shared/xujingyu/xujy/data/encoder_weights/Wan2.1_VAE.pth",
                path="/gemini/platform/shared/yifq1/yifq/taew2_1.pth",
                type="TeleaiVideoTAE_2_1",
                tiler_kwargs=dict(
                    tiled=False,  # 
                    tile_size=(34, 34),
                    tile_stride=(18, 16),
                    has_mask=True, # True 会带mask的4维
                ),
                torch_compile=False # must be false
            ),
            text_encoder=dict(
                path="/gemini/platform/shared/xujingyu/xujy/data/encoder_weights/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/gemini/platform/shared/xujingyu/xujy/data/encoder_weights/google/umt5-xxl",
            ),
            image_encoder=dict(
                path="/gemini/platform/shared/xujingyu/xujy/data/encoder_weights/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                torch_compile=False
            ),
        ),
    ),
)

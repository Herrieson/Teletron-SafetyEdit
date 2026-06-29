data_config = dict(
    dataset=dict(
        filter_cfg=dict(
            dst_size=dst_size,
            dst_num_frames=num_frames,
            min_num_frames=num_frames,
            dst_fps=30,
            multiple=16,
            min_area=dst_size[0] * dst_size[1],
            min_size=4,
            aesthetic_th=-1,
            optical_flow_unimatch=0.0, 
            optical_flow=0.0,
            condition_list=["canny"],

        ),
        transforms=[
            dict(
                type="SampleImages",
                num_frames=num_frames,
            ),
            dict(
                type="GenerateFirstAndLastRefImage",
            ),
            dict(
                type="ConditionGenerator",
                cn_keys=["canny"],
                mask_ratios={
                    "random": 0.0,
                },  
            ),
            dict(
                type="PackInputs",
                crop_keys=[
                    "images",
                    "raw_first_image", 
                    "raw_last_image",
                    "masked_images",
                    "canny_images",
                ],
                image_keys=[
                    "images",
                    "masked_images",
                    "canny_images",
                ],
                embedding_keys=[],
                dst_size=dst_size,
            ),  
        ],
    ),
    models=dict(
        text_encoder_path="/data01/model_zoo/Wan2.1-FLF2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        vae_path="/data01/model_zoo/Wan2.1-FLF2V-14B-720P/Wan2.1_VAE.pth",
        image_encoder_path="/data01/model_zoo/Wan2.1-FLF2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        dit_path="/data01/model_zoo/Wan2.1-FLF2V-14B-720P", 
        tiled=True, 
        tile_size=(34, 34),
        tile_stride=(18, 16), 
        pretrained_downsample_model_path = "/gemini/platform/public/aigc/yifq/yfq/vast-fl2v/work_dirs/wanvideo/wanvideo_fl2v_down-up-cn_20250811_vae_train_v3_three-decoder/models/checkpoint_epoch_10_step_6800/downsample_model/downsample_model.pt",
        pretrained_upsample_model_path = "/gemini/platform/public/aigc/yifq/yfq/vast-fl2v/work_dirs/wanvideo/wanvideo_fl2v_down-up-cn_20250811_vae_train_v3_three-decoder/models/checkpoint_epoch_10_step_6800/upsample_model/upsample_model.pt",
        cn_keys=["canny_images"],
    ),
)
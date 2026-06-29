import types
import math

from teletron.models.teleai.models.dit import TeleaiPrompter
from teletron.models.teleai.models.dit import TeleaiTextEncoder
from teletron.models.teleai.models.dit import TeleaiVideoVAE
from teletron.models.teleai.models.dit import TeleaiVideoVAE_2_2
from teletron.models.teleai.models.dit import TeleaiImageEncoder
from teletron.models.teleai.taehv import TAEW2_1DiffusersWrapper
from teletron.models.teleai import TeleaiModel
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop


from teletron.models.flow_match import FlowMatchScheduler
from .base import BasePipeline
import torch, os
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional


from PIL import Image

def resize_and_crop(image, target_size):
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    scale = max(target_width / original_width, target_height / original_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    return cropped_image


class WanVideoI2VPipeline(BasePipeline):

    def __init__(self, pipeline_config):
        torch_dtype = pipeline_config.get("torch_dtype", torch.bfloat16)
        device = pipeline_config.get("device", "cuda")
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        
        # model config
        self.model_config = pipeline_config.get("model_config", None)

        # dit config
        dit_config = pipeline_config.get("model_config", None).get("dit", None)
        self.dit_config = dit_config
        print(f"加载 DiT 模型... {dit_config.get('path')}")
        # with torch.device('meta'):
        self.dit = TeleaiModel(**dit_config.get("config"))
        if "ema" in dit_config.get("path"):
            state_static = torch.load(dit_config.get("path"), map_location='cpu', weights_only=False)[0]
        else:
            state_static = torch.load(dit_config.get("path"), map_location='cpu', weights_only=False)["model"]

        self.dit.load_state_dict(state_static, strict=True, assign=True)
        self.dit.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)

        # encoder config
        self.encoder_model_config = pipeline_config.get("model_config", None).get("encoder", None)
        
        # vae config
        self.vae_path = self.encoder_model_config.get("vae", None).get("path", None)
        self.vae_type = self.encoder_model_config.get("vae", None).get("type", "TeleaiVideoVAE_2_1")
        self.tiler_kwargs = self.encoder_model_config.get("vae", None).get("tiler_kwargs", {})
        if self.tiler_kwargs is None:
            self.tiler_kwargs = dict(
                tiled=False,
                tile_size=(34, 34),
                tile_stride=(18, 16),
            )
        print(f"加载 VAE 模型... {self.vae_path}")
        if self.vae_type == "TeleaiVideoVAE_2_1":
            self.vae = TeleaiVideoVAE().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.compression = (4,8,8)
            self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu', weights_only=False), strict=True)

        elif self.vae_type == "TeleaiVideoTAE_2_1":
            self.compression = (4,8,8)
            self.vae = TAEW2_1DiffusersWrapper(self.vae_path, device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        else:
            self.vae = TeleaiVideoVAE_2_2().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.compression = (4,16,16)
            self.vae.model.load_state_dict(torch.load(self.vae_path, map_location='cpu', weights_only=False), strict=True)
            
        # text encoder config
        text_encoder_path = self.encoder_model_config.get("text_encoder", None).get("path", None)
        tokenizer_path = self.encoder_model_config.get("text_encoder", None).get("tokenizer_path", None)
        print(f"加载 Text Encoder 模型... {text_encoder_path}")
        with torch.device('meta'):
            self.text_encoder = TeleaiTextEncoder()
        self.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location='cpu', weights_only=False), strict=True, assign=True)
        self.text_encoder.to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.prompter = TeleaiPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(tokenizer_path)

        if self.encoder_model_config.get("image_encoder", None) is not None and self.dit.has_image_input:
            image_encoder_path = self.encoder_model_config.get("image_encoder", None).get("path", None)
            print(f"加载 Image Encoder 模型... {image_encoder_path}")
            self.image_encoder = TeleaiImageEncoder().to(device=self.device, dtype=torch.bfloat16).eval().requires_grad_(False)
            self.image_encoder.model.load_state_dict(torch.load(image_encoder_path, map_location='cpu', weights_only=False), strict=False)
        else:
            self.image_encoder = None
        
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder'] if self.image_encoder is not None else ['text_encoder', 'dit', 'vae']

        self.height_division_factor = 16
        self.width_division_factor = 16

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive, device=self.device)
        return {"context": prompt_emb}

    def encode_ref_images(self, ref_images, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        ref_images = [(int(frame_id), self.preprocess_image(resize_and_crop(image, (width, height))).to(self.device)) for frame_id, image in ref_images]
        ref_video = torch.zeros(1, num_frames, 3, height, width)
        for frame_id, ref_image in ref_images:
            ref_video[:, frame_id] = ref_image.unsqueeze(0)
        ref_video = ref_video.to(dtype=self.torch_dtype, device=self.device).permute(0, 2, 1, 3, 4)
        ref_latents = self.vae.encode(
            ref_video, device=self.device, 
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        ).to(dtype=self.torch_dtype, device=self.device)

        msk = torch.zeros(1, num_frames, height//8, width//8, device=self.device)
        for frame_id, _ in ref_images:
            msk[:, frame_id] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, ref_latents], dim=1)

        if self.dit.has_image_input:
            assert ref_images[0][0] == 0 # first frame
            clip_context = self.image_encoder.encode_image(
                [ref_images[0][1].to(dtype=self.torch_dtype, device=self.device)]
            ).to(dtype=self.torch_dtype, device=self.device)

        if self.dit.has_image_input:
            return {"clip_feature": clip_context, "y": y}
        else:
            return {"y": y}


    def encode_image(self, image, end_image, num_frames, height, width, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        # import ipdb;ipdb.set_trace()
        image = self.preprocess_image(image.resize((width, height))).to(device="cuda",dtype=torch.bfloat16) # 训练
        # print(image.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = self.preprocess_image(end_image.resize((width, height))).to(device="cuda",dtype=torch.bfloat16)# 训练
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = self.vae.encode(vae_input.to(dtype=self.torch_dtype, device=self.device).unsqueeze(0), device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=self.torch_dtype, device=self.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        if frames.max().item() > 10: # TAE
            frames = frames.cpu().numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
        else:
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames
    
    def encode_condition(self, condition, num_frames, height, width, has_mask=False, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(condition, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) # [1, 16, T, H, W]
        msk = torch.zeros(1, num_frames, height//8, width//8, device=self.device)
        index = (condition > -0.999).any(dim=(1, 3, 4))[0]
        msk[:, index] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2).to(dtype=self.torch_dtype, device=self.device)
        if has_mask:
            y = torch.concat([msk, latents], dim=1) # [1, 16+4, T, H, W]
        else:
            y = latents
        return y

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        ref_images=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        progress_bar_cmd=tqdm,
        **kwargs
    ):
        # Parameter check
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        target_area = width * height
        if ref_images is not None:
            original_width, original_height = ref_images[0][1].size
            ratio = original_height / original_width
            new_width, new_height = math.sqrt(target_area / ratio), math.sqrt(target_area * ratio)
            width = int((new_width // 16) * 16)
            height = int((new_height // 16) * 16)
        else:
            width = (width // 16) * 16
            height = (height // 16) * 16

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if ref_images is not None:
            self.load_models_to_device(["vae"]) if self.image_encoder is None else self.load_models_to_device(["vae", "image_encoder"])
            image_emb = self.encode_ref_images(ref_images, num_frames, height, width) # without tilling

        # Denoise
        self.load_models_to_device(self.dit)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc='Denoising ...')):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(
                x=latents, timestep=timestep, **prompt_emb_posi, **image_emb,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    x=latents, timestep=timestep, **prompt_emb_nega, **image_emb,
                    )
                noise_pred = cfg_scale * noise_pred_posi + (1 - cfg_scale) * noise_pred_nega
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, denoising_strength=1.)

        # Decode
        self.load_models_to_device(['vae'])
        
        frames = self.decode_video(latents, tiled=False)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
    
    @torch.no_grad
    def recon(
        self,
        input_video, 
        prompt,
        negative_prompt="",
        input_image=None,
        last_image=None, 
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=1.0,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        num_inference_steps=50,
        sigma_shift=5.0,
        progress_bar_cmd=tqdm,
        cn_images=None,
        add_cn_noise=False,
        has_mask=False,
        **kwargs
    ):
        # Parameter check
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            ori_latents = self.encode_video(input_video.to(dtype=self.torch_dtype, device=self.device), **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            downsample_latents1, downsample_latents2, downsample_latents3 = self.dit.compressor_down(ori_latents) # 16x64x64,  16x32x32, 8x16x16

            # quantization
            if not add_cn_noise: # 1.3B
                # compression
                dequantized_downsample_latents3, latent_uint8_3, _ = self.dit.quantizer(downsample_latents3)

            upsample_latents1 = self.dit.compressor_up(dequantized_downsample_latents3) # v3
            mu, logvar = upsample_latents1.chunk(2, dim=1)
            noise = mu * 0.5 + noise * 0.5
            # noise = mu
        latents = noise

        if cn_images is not None:
            bs, c, num_frames, h, w = cn_images.shape
            if add_cn_noise: # 14B
                cn_images = self.encode_condition(cn_images.to(dtype=self.torch_dtype, device=self.device), num_frames=num_frames, height=h, width=w, has_mask=has_mask,**tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
                cn_noise = torch.rand_like(cn_images)
                cn_images = 0.5 * cn_images + 0.5 * cn_noise
                cn_images = cn_images.to(dtype=self.torch_dtype, device=self.device)
            else: # 1.3B
                cn_images = self.encode_condition(cn_images.to(dtype=self.torch_dtype, device=self.device), num_frames=num_frames, height=h, width=w, has_mask=has_mask,**tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
                cn_images = cn_images.to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, last_image, num_frames, height, width)
        else:
            image_emb = {}

        # Denoise
        self.load_models_to_device(self.dit)

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc='Denoising ...')):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.dit(
                x=latents, timestep=timestep, **prompt_emb_posi, **image_emb, cn_images=cn_images,
            )

            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    x=latents, timestep=timestep, **prompt_emb_nega, **image_emb, cn_images=cn_images,
                    )
                noise_pred = cfg_scale * noise_pred_posi + (1 - cfg_scale) * noise_pred_nega
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents, denoising_strength=1.)


        # Decode
        self.load_models_to_device(['vae'])
        
        frames = self.decode_video(latents, tiled=False)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
    



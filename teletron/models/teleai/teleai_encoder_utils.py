import torch
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
import numpy as np
from einops import rearrange
from collections import defaultdict
from teletron.train.utils import get_args
from teletron.utils import set_config
import random
from .keyframe_utils import encoder_keyframe_fun, init_encoder, decoder_keyframe_fun, init_decoder, flf_process

def encode_prompt(prompter,prompt, positive=True):
    prompt_emb = prompter.encode_prompt(
        prompt, positive=positive, device=torch.cuda.current_device()
    )
    return {"context": prompt_emb}

def encode_image(
    vae,
    image_encoder,
    image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    dtype=torch.bfloat16,
    compression=(4,8,8),
):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
    # print("msk create shape:", 1, num_frames, height // 8, width // 8 ) # 1, 81, 56, 98
    msk[:, 1:] = 0 # 1, 1:81, 56, 98
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
    # print("msk view shape:", 1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2]) # 1, 21, 4, 56, 98
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
        dim=1,
    )
    y = vae.encode(
        vae_input.to(dtype=dtype, device=torch.cuda.current_device()).unsqueeze(0),
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

def encode_image_with_mask(
        vae,
        image_encoder, 
        image, 
        num_frames,
        height, 
        width, 
        msk, 
        ref_images, 
        tiled=False, 
        tile_size=(34, 34), 
        tile_stride=(18, 16),
        dtype=torch.bfloat16
    ):
    image = preprocess_image(image.resize((width, height))).to(torch.cuda.current_device())
    clip_context = image_encoder.encode_image([image])
    ref_images = rearrange(ref_images, 'b t c h w -> b c t h w')
    y = encode_video(
        vae, 
        ref_images.to(dtype=dtype, device=torch.cuda.current_device()),
        tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
    y = y.unsqueeze(0)
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    msk = msk.transpose(1, 2).to(torch.cuda.current_device())
    y = torch.concat([msk, y], dim=1)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}


def encode_first_last_image(
    vae,
    image_encoder,
    pil_first_image,
    pil_last_image,
    num_frames,
    height,
    width,
    tiled=False,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    dtype=torch.bfloat16,
    compression=(4,8,8),
):
    first_image = preprocess_image(pil_first_image.resize((width, height))).to(
        torch.cuda.current_device()
    )
    last_image = preprocess_image(pil_last_image.resize((width, height))).to(
        torch.cuda.current_device()
    )
    # if self.dit.has_image_pos_emb:
    #     clip_context = torch.cat([self.image_encoder.encode_image([first_image]),
    #                             self.image_encoder.encode_image([last_image])], dim=1)
    # else:
    #     clip_context = self.image_encoder.encode_image([first_image])
    clip_context = torch.cat(
        [
            image_encoder.encode_image([first_image]),
            image_encoder.encode_image([last_image]),
        ],
        dim=1,
    )
    msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
    msk[:, 1:-1] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
    )
    msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2])
    msk = msk.transpose(1, 2)[0]
    vae_input = torch.concat(
        [
            first_image.transpose(0, 1),
            torch.zeros(3, num_frames - 2, height, width).to(first_image.device),
            last_image.transpose(0, 1),
        ],
        dim=1,
    )
    y = vae.encode(
        vae_input.to(dtype=dtype, device=torch.cuda.current_device()).unsqueeze(0),
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )[0]
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())
    y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return {"clip_feature": clip_context, "y": y}

    
def encode_video(vae, input_video, return_var=False, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    latents = vae.encode(
        input_video,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return latents


def encode_condition(vae, condition, num_frames, height, width, has_mask=False, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    latents = vae.encode(
        condition, 
        device=torch.cuda.current_device(), 
        tiled=tiled, 
        tile_size=tile_size, 
        tile_stride=tile_stride,
    ) # [1, 16, T, H, W]
    msk = torch.zeros(1, num_frames, height//8, width//8, device=torch.cuda.current_device())
    index = (condition > -0.999).any(dim=(1, 3, 4))[0]
    msk[:, index] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
    msk = msk.transpose(1, 2).to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    if has_mask:
        y = torch.concat([msk, latents], dim=1) # [1, 16+4, T, H, W]
    else:
        y = latents
    return y

def decode_video(vae, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
    frames = vae.decode(
        latents,
        device=torch.cuda.current_device(),
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    return frames

def preprocess_image(image):
    image = (
        torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return image

def preprocess_images(images):
    return [preprocess_image(image) for image in images]


def get_encoder_features(batch, prompter, vae, tiler_kwargs, image_encoder, dtype=torch.bfloat16, compression=(4,8,8)):
    with torch.no_grad():
        prompt_emb = encode_prompt(prompter,batch["dense_prompt"][0])
        latents = encode_video(vae,
            rearrange(batch["images"], "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            **tiler_kwargs,
        )[0]
        _, num_frames, _, height, width = batch["images"].shape
        # print("images: ",height, width )
        if 'raw_last_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_first_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            raw_last_image = batch['raw_last_image']
            pil_last_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_first_last_image(
                vae,image_encoder, pil_first_image, pil_last_image, num_frames, height, width, dtype=dtype
            )
        elif 'raw_first_image' in batch:
            raw_first_image = batch["raw_first_image"]
            pil_image = to_pil_image(
                raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            image_emb = encode_image(vae, image_encoder, pil_image, num_frames, height, width, dtype=dtype, compression=compression)
        elif 'ref_images' in batch:
            first_image = (batch['ref_images'] + 1) / 2 * 255
            ref_mask = batch["ref_mask"]
            ref_images = batch["ref_images"]
            pil_image = to_pil_image(first_image[0][0].cpu().permute(1,2,0).numpy().astype(np.uint8))
            image_emb = encode_image_with_mask(vae, image_encoder, pil_image, num_frames,
                                                height, width, ref_mask, ref_images, dtype=dtype)
        
        
        latents = latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())

        # Data
        prompt_emb["context"] = prompt_emb["context"][0].to(
            dtype=dtype, device=torch.cuda.current_device()
        )
        prompt_emb["context"] = prompt_emb["context"].unsqueeze(0)

        if "clip_feature" in image_emb:
            image_emb["clip_feature"] = (
                image_emb["clip_feature"][0]
                .to(dtype=dtype, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
        if "y" in image_emb:
            image_emb["y"] = (
                image_emb["y"][0]
                .to(dtype=dtype, device=torch.cuda.current_device())
                .unsqueeze(0)
            )
    return prompt_emb, image_emb, latents

@torch.no_grad
def get_context(batch, prompter, dtype=torch.bfloat16):
    prompt_emb = encode_prompt(prompter, batch["struct_prompt"][0])
    prompt_emb["context"] = prompt_emb["context"].to(
            dtype=dtype, device=torch.cuda.current_device()
    )
    return prompt_emb["context"]


@torch.no_grad
def get_unprompt_emb(batch, prompter, dtype=torch.bfloat16):
    args = get_args()
    batch_size = args.micro_batch_size
    prompt_emb = encode_prompt(prompter, [args.negative_prompt] * batch_size)
    prompt_emb["context"] = prompt_emb["context"].to(
            dtype=dtype, device=torch.cuda.current_device()
    )
    return prompt_emb["context"]

@torch.no_grad
def get_img_clip_feature(batch, image_encoder, keyframeenc=None, keyframedec=None, dtype=torch.bfloat16):
    _, num_frames, _, height, width = batch["images"].shape
    if 'raw_first_image' in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        clip_context = image_encoder.encode_image([image])
        
        if 'raw_last_image' in batch:
            raw_last_image = batch["raw_last_image"]
            last_pil_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            last_pil_image = preprocess_image(last_pil_image.resize((width, height))).to(torch.cuda.current_device())
            clip_context = torch.concat([clip_context, image_encoder.encode_image([last_pil_image])], dim=1)
        clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())

    else:
        raise NotImplementedError("ref_images is not supported yet")
    return clip_context

@torch.no_grad
def get_img_emb_y(batch, vae, dtype=torch.bfloat16, compression=(4,8,8), keyframeenc=None, keyframedec=None, tiler_kwargs={}):
    _, num_frames, _, height, width = batch["images"].shape
    if 'ref_images' in batch:
        # assert False, "ref_images is not supported yet"
        ref_images = rearrange(batch["ref_images"], "b t c h w -> b c t h w")
        y = vae.encode(
            ref_images.to(dtype=dtype, device=torch.cuda.current_device()),
            device=torch.cuda.current_device(),
            **tiler_kwargs
        )
        msk = batch['ref_mask'].transpose(1, 2).to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y], dim=1)
    
    elif 'raw_first_image' in batch:
        raw_first_image = batch["raw_first_image"]
        pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        image = preprocess_image(pil_image.resize((width, height))).to(torch.cuda.current_device())
        msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
        msk[:, 1:] = 0 

        if 'raw_last_image' in batch:
            raw_last_image = batch["raw_last_image"]
            last_pil_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            last_pil_image = preprocess_image(last_pil_image.resize((width, height))).to(torch.cuda.current_device())
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(torch.cuda.current_device()), last_pil_image.transpose(0,1)],dim=1)
            # clip_context = torch.concat([clip_context, self.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
        msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2]) # 1, 21, 4, 56, 98
        msk = msk.transpose(1, 2)[0]
        
        y = vae.encode(
            vae_input.to(dtype=dtype, device=torch.cuda.current_device()).unsqueeze(0),
            device=torch.cuda.current_device(),
            **tiler_kwargs
        )[0]
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return y


@torch.no_grad
def get_img_clip_feature_ds(batch, image_encoder, keyframeenc=None, keyframedec=None, dtype=torch.bfloat16):
    _, num_frames, _, height, width = batch["images"].shape
    if 'raw_first_image_ds' in batch:
        raw_first_image = batch["raw_first_image_ds"]
        first_pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        
        
        if 'raw_last_image_ds' in batch:
            raw_last_image = batch["raw_last_image_ds"]
            last_pil_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            if keyframeenc is not None:
                keyframeenc.frame_idx = 0
                first_pil_image, last_pil_image, frame_bytes = flf_process(keyframeenc, keyframedec, first_pil_image, last_pil_image, qp_i=random.choice([32, 35, 38, 42, 45]), qp_p=random.choice([32, 35, 38, 42, 45])) 
            
            first_pil_image = preprocess_image(first_pil_image.resize((width, height))).to(torch.cuda.current_device())
            clip_context = image_encoder.encode_image([first_pil_image])
            last_pil_image = preprocess_image(last_pil_image.resize((width, height))).to(torch.cuda.current_device())
            clip_context = torch.concat([clip_context, image_encoder.encode_image([last_pil_image])], dim=1)
        else:
            first_pil_image = preprocess_image(first_pil_image.resize((width, height))).to(torch.cuda.current_device())
            clip_context = image_encoder.encode_image([first_pil_image])

        clip_context = clip_context.to(dtype=dtype, device=torch.cuda.current_device())

    else:
        raise NotImplementedError("ref_images is not supported yet")
    return clip_context

@torch.no_grad
def get_img_emb_y_ds(batch, vae, dtype=torch.bfloat16, compression=(4,8,8), keyframeenc=None, keyframedec=None, tiler_kwargs={}):
    _, num_frames, _, height, width = batch["images"].shape
    if 'ref_images_ds' in batch:
        # assert False, "ref_images is not supported yet"
        ref_images = rearrange(batch["ref_images_ds"], "b t c h w -> b c t h w")
        y = vae.encode(
            ref_images.to(dtype=dtype, device=torch.cuda.current_device()),
            device=torch.cuda.current_device(),
            **tiler_kwargs
        )
        msk = batch['ref_mask_ds'].transpose(1, 2).to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y], dim=1)
    
    elif 'raw_first_image_ds' in batch:
        raw_first_image = batch["raw_first_image_ds"]
        first_pil_image = to_pil_image(
            raw_first_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        )
        msk = torch.ones(1, num_frames, height // compression[1], width // compression[2], device=torch.cuda.current_device())
        msk[:, 1:] = 0 
        

        if 'raw_last_image_ds' in batch:
            raw_last_image = batch["raw_last_image_ds"]
            last_pil_image = to_pil_image(
                raw_last_image[0][0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            )
            if keyframeenc is not None:
                keyframeenc.frame_idx =0
                first_pil_image, last_pil_image, frame_bytes = flf_process(keyframeenc, keyframedec, first_pil_image, last_pil_image, qp_i=random.choice([32, 35, 38, 42, 45]), qp_p=random.choice([32, 35, 38, 42, 45])) 

            first_pil_image = preprocess_image(first_pil_image.resize((width, height))).to(torch.cuda.current_device())
            last_pil_image = preprocess_image(last_pil_image.resize((width, height))).to(torch.cuda.current_device())
            vae_input = torch.concat([first_pil_image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(torch.cuda.current_device()), last_pil_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            first_pil_image = preprocess_image(first_pil_image.resize((width, height))).to(torch.cuda.current_device())
            vae_input = torch.concat([first_pil_image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(first_pil_image.device)], dim=1)

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        ) # 1, 4, 56, 98; # 1, 80, 56, 98 => 1, 84, 56, 98
        msk = msk.view(1, msk.shape[1] // compression[0], compression[0], height // compression[1], width // compression[2]) # 1, 21, 4, 56, 98
        msk = msk.transpose(1, 2)[0]
        
        y = vae.encode(
            vae_input.to(dtype=dtype, device=torch.cuda.current_device()).unsqueeze(0),
            device=torch.cuda.current_device(),
            **tiler_kwargs
        )[0]
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=dtype, device=torch.cuda.current_device())
    return y


@torch.no_grad
def get_latents(batch, vae, dtype=torch.bfloat16, **tiler_kwargs):
    def _get_latents(images):
        # 将视频张量重排为 VAE期望的 (b, c, t, h, w) 格式并进行编码
        latents = encode_video(
            vae,
            rearrange(images, "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            tiler_kwargs
        )
        return latents.to(dtype=dtype, device=torch.cuda.current_device())

    return _get_latents(batch["images"])


@torch.no_grad
def get_latents_ds(batch, vae, dtype=torch.bfloat16, **tiler_kwargs):
    def _get_latents(images):
        # 将视频张量重排为 VAE期望的 (b, c, t, h, w) 格式并进行编码
        latents = encode_video(
            vae,
            rearrange(images, "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            tiler_kwargs
        )
        return latents.to(dtype=dtype, device=torch.cuda.current_device())

    return _get_latents(batch["images_ds"])


@torch.no_grad
def get_latents_canny_images(batch, vae, dtype=torch.bfloat16, tiler_kwargs={}):
    def _get_latents(images):
        bs, num_frames, _, height, width = images.shape
        # 将视频张量重排为 VAE期望的 (b, c, t, h, w) 格式并进行编码
        latents = encode_condition(
            vae,
            rearrange(images, "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            num_frames=num_frames,
            height=height,
            width=width,
            **tiler_kwargs
        )
        return latents.to(dtype=dtype, device=torch.cuda.current_device())
    return _get_latents(batch["canny_images"])

@torch.no_grad
def get_latents_canny_masked_images(batch, vae, dtype=torch.bfloat16, tiler_kwargs={}):
    def _get_latents(images):
        bs, num_frames, _, height, width = images.shape
        # 将视频张量重排为 VAE期望的 (b, c, t, h, w) 格式并进行编码
        latents = encode_condition(
            vae,
            rearrange(images, "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            num_frames=num_frames,
            height=height,
            width=width,
            has_mask=True,
            **tiler_kwargs
        )
        return latents.to(dtype=dtype, device=torch.cuda.current_device())
    return _get_latents(batch["canny_images"])

@torch.no_grad
def get_latents_masked_images(batch, vae, dtype=torch.bfloat16, tiler_kwargs={}):
    '''
        返回均值和方差
    '''
    def _get_latents(images):
        # 将视频张量重排为 VAE期望的 (b, c, t, h, w) 格式并进行编码
        latents_mu, latents_var = vae.single_encode(
            rearrange(images, "b t c h w -> b c t h w").to(
                dtype=dtype, device=torch.cuda.current_device()
            ),
            device=torch.cuda.current_device(),
            return_var=True,
        )
        return torch.cat([latents_mu.to(dtype=dtype, device=torch.cuda.current_device()), latents_var.to(dtype=dtype, device=torch.cuda.current_device())], dim=1)
    
    return _get_latents(batch["masked_images"])



@torch.no_grad
def get_noise(batch, dtype=torch.bfloat16, compression=(4,8,8)):
    if 'latents' in batch:
        return torch.randn_like(batch['latents']).to(dtype=dtype, device=torch.cuda.current_device())
    else:
        bsz, num_frames, _, height, width = batch["images"].shape
        return torch.randn(bsz, 16, (num_frames + 3) // compression[0], height // compression[1], width // compression[2]).to(dtype=dtype, device=torch.cuda.current_device())

@torch.no_grad
def get_fake_latents(batch, vae, dtype=torch.bfloat16, tiler_kwargs={}):
    bsz, num_frames, video_channels, height, width = batch["images"].shape
    
    low_res_video = torch.nn.functional.interpolate(
        rearrange(batch["images"], "b t c h w -> (b t) c h w"),
        size=(height // 2, width // 2),
        mode='bilinear'
    ).reshape(bsz, num_frames, video_channels, height // 2, width // 2)
    
    low_res_latent = encode_video(vae,
        rearrange(low_res_video, "b t c h w -> b c t h w").to(
            dtype=dtype, device=torch.cuda.current_device()
        ),
        **tiler_kwargs,
    ) # b c t h w
    
    bsz, latent_channels, latent_frames, latent_height, latent_width = bsz, 16, (num_frames + 3) // 4, height // 8, width // 8
    fake_latents = torch.nn.functional.interpolate(
        rearrange(low_res_latent, "b c t h w -> (b t) c h w"),
        size=(latent_height, latent_width),
        mode='nearest'
    ).reshape(bsz, latent_frames, latent_channels, latent_height, latent_width)[0] # t, c, h, w
    fake_latents = fake_latents.permute(1, 0, 2, 3) # c, t, h, w
    
    fake_latents = fake_latents.unsqueeze(0).to(dtype=dtype, device=torch.cuda.current_device())

    return fake_latents # b, c, t, h, w

@torch.no_grad
def get_frame_interval(batch, dtype=torch.bfloat16):
    return batch['frame_interval'].to(
        dtype=dtype, device=torch.cuda.current_device()
    )

@torch.no_grad
def get_depth_latents(batch, depth_model, vae, dtype=torch.bfloat16, tiler_kwargs={}):
    global_config = set_config()
    target_fps = global_config.dataset.filter_cfg.dst_fps
    frames = rearrange(batch["images"], "b t c h w -> b t h w c").squeeze(0).numpy()
    depths, fps = depth_model.infer_video_depth(frames, target_fps, device='cuda', fp32=dtype==torch.float32)
    depths = torch.from_numpy(depths).unsqueeze(0).unsqueeze(2).repeat(1, 1, 3, 1, 1).to(dtype=dtype, device=torch.cuda.current_device())

    depths_latents = encode_video(
        vae,
        rearrange(depths, "b t c h w -> b c t h w").to(
            dtype=dtype, device=torch.cuda.current_device()
        ),
        **tiler_kwargs,
    )
    return depths_latents.to(dtype=dtype, device=torch.cuda.current_device())

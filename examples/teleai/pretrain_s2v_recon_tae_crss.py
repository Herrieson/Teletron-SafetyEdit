import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func
from teletron.utils import print_rank_0
import random

import torch.nn.functional as F

def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group.add_argument("--test-with-pseudo-data", action="store_true")
    group.add_argument("--test-resolution", type=str, default="360")
    
    return parser


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def kl_gaussian_safe(mu, logvar, mu_p, logvar_p, eps=1e-6, min_logvar=-10, max_logvar=10):
    logvar   = torch.clamp(logvar,   min_logvar, max_logvar)
    logvar_p = torch.clamp(logvar_p, min_logvar, max_logvar)

    var   = torch.exp(logvar)   + eps
    var_p = torch.exp(logvar_p) + eps

    kl = 0.5 * ((var + (mu - mu_p)**2) / var_p
                - 1
                + torch.log(var_p) - torch.log(var))
    kl = torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=-1e4).mean()
    return kl

def forward_step(data_iterator, model, time_step=None):
    flow_scheduler = FlowMatchScheduler(shift=1, sigma_min=0.0, extra_one_step=True)
    flow_scheduler.set_timesteps(1000, training=True)
    
    batch = next(data_iterator)
    
    latents = batch["latents"]
    p_mu,_ = batch['latents_masked_images'].chunk(2, dim=1)
    cns = batch['latents_canny_images']

    noise = torch.randn_like(latents) 
    timestep_range = [0, flow_scheduler.num_train_timesteps]

    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16, device=torch.cuda.current_device()
    )
    def broadcast_timesteps(input: torch.Tensor):
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if mpu.get_tensor_context_parallel_world_size() > 1:
            dist.broadcast(input, tp_cp_src_rank, group=mpu.get_tensor_context_parallel_group())

    if time_step is not None:
        timestep = torch.tensor([time_step], dtype=torch.bfloat16, device=torch.cuda.current_device())

    broadcast_timesteps(timestep)
    broadcast_timesteps(noise)

    # custom modification
    downsample_latents1, downsample_latents2, downsample_latents3 = model.module.module.compressor_down(latents) # <DistributedDataParallel  < FP16Moudle < TeleAImodel > > > 
    # import pdb; pdb.set_trace()
    # quantization
    dequantized_downsample_latents1, latent_uint8_1, latent_probs1 = model.module.module.quantizer(downsample_latents1)
    dequantized_downsample_latents2, latent_uint8_2, latent_probs2 = model.module.module.quantizer(downsample_latents2)
    dequantized_downsample_latents3, latent_uint8_3, latent_probs3 = model.module.module.quantizer(downsample_latents3)

    #
    tokens = dequantized_downsample_latents3.reshape(1, 16, -1).permute(0, 2, 1)
    tokens = model.module.module.token_emb(tokens)

    
    training_target = flow_scheduler.training_target(latents, noise, timestep)    
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    loss_weight = flow_scheduler.training_weight(timestep)


    output_tensor_list = model(x=noisy_latents, 
                            timestep=timestep, 
                            # context=batch['context'],
                            context=tokens,
                            clip_feature=batch['img_clip_feature'],
                            y=batch['img_emb_y'],
                            cn_images=None,
                        )

    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    
    loss = loss * loss_weight
    return [loss, loss, loss, loss, loss], loss_func



if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .teleai_video_dit import SelfAttention, precompute_freqs_cis_3d


class MyTransformerBlock3D(nn.Module):
    """
    Pre-Norm Transformer:
        x = x + MHA(Norm(x))
        x = x + FFN(Norm(x))
    """
    def __init__(self, dim, heads=4, mlp_ratio=4.0, dropout=0.0, eps=1e-6):
        super().__init__()
        self.attn  = SelfAttention(dim, heads, eps)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, freqs):
        # x: (B, C, T, H, W)  ->  (B, THW, C)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)                # (B, THW, C)
        x = x + self.attn(x, freqs)
        x = x + self.ffn(self.norm2(x))
        x = x.transpose(1, 2).view(B, C, T, H, W)       # 还原
        return x

class Upsample3d_Coder_v3(nn.Module):
    def __init__(self, hidden_dim=96, num_heads=4, eps=1e-6, depth=2, stride=(1,2,2)):
        super().__init__()
        mlp_ratio = 4.0
        dropout = 0.1
        self.blocks = nn.ModuleList([
            MyTransformerBlock3D(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.up = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
            padding=1, output_padding=tuple(s-1 for s in stride))
        
    def forward(self, x, freqs):
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.up(x)
        return x

class Downsample3d_Coder_v3(nn.Module):
    def __init__(self, hidden_dim=96, num_heads=4, eps=1e-6, depth=2, stride=(1,2,2)):
        super().__init__()
        mlp_ratio = 4.0
        dropout = 0.1
        self.blocks = nn.ModuleList([
            MyTransformerBlock3D(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.down = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1)
        
    def forward(self, x, freqs):
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.down(x)
        return x



class EnhancedResidualUpsample3d_v3(nn.Module):
    """
    残差上采样网络
    输入: (B, 16, T/8, H/64, W/64)
    输出: (B, 16, T/4, H/8, W/8)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        self.input_proj1 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj2 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj3 = nn.Conv3d(dim, hidden_dim, 1)

        self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # 三个分辨率级
        self.decoder1 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2)),
        ])

        self.decoder2 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2)),
        ])

        self.decoder3 = nn.ModuleList([Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(2,2,2))])

        self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        
    def forward(self, x1):
        x1 = self.input_proj1(x1)
        for blk in self.decoder1:
            b, c, f, h, w = x1.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x1.device)
            x1 = blk(x1, freqs)
        x1_out = self.output_proj1(x1)
        return x1_out

        x2 = self.input_proj2(x2)
        for blk in self.decoder2:
            b, c, f, h, w = x2.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x2.device)
            x2 = blk(x2, freqs)
        x2_out = self.output_proj2(x2)

        x3 = self.input_proj2(x3)
        for blk in self.decoder3:
            b, c, f, h, w = x3.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x3.device)
            x3 = blk(x3, freqs)
        x3_out = self.output_proj3(x3)
        return x1_out, x2_out, x3_out


class EnhancedResidualUpsample3d_v3_1(EnhancedResidualUpsample3d_v3):
    '''
    Docstring for EnhancedResidualUpsample3d_v3_1
    带pixel shuffle，再做一次上采样
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_proj1 = nn.Conv3d(kwargs["dim"], kwargs["hidden_dim"] * 4, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x1):
        x1 = self.input_proj1(x1)
        # pixel shuffle
        b,c,t,h,w = x1.shape
        x1 = x1.permute(0,2,1,3,4).reshape(b * t, c, h, w)

        x1 = self.pixel_shuffle(x1)
        x1 = x1.reshape(b, t, c//4, h*2, w*2).permute(0,2,1,3,4)
        for blk in self.decoder1:
            b, c, f, h, w = x1.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x1.device)
            x1 = blk(x1, freqs)
        x1_out = self.output_proj1(x1)
        return x1_out


class EnhancedResidualDownsample3d_v3(nn.Module):

    """
    增强版带残差连接、且带self-attn的降采样网络
    输入: (B, 16, T/4, H/8, W/8)
    输出: (B, 16, T/8, H/64, W/64)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        
        # 输入投影层 - 扩展通道数
        self.input_proj = nn.Conv3d(dim, hidden_dim, 1)
        self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # 三个分辨率级
        self.encoder1 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(2,2,2))
        self.encoder2 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2))
        self.encoder3 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2))

        self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        

    def forward(self, x):
        x = self.input_proj(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder1(x, freqs)
        x1_out = self.output_proj1(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder2(x, freqs)
        x2_out = self.output_proj2(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder3(x, freqs)
        x3_out = self.output_proj3(x)
        
        return x1_out, x2_out, x3_out


class EnhancedResidualDownsample3d_v3_1(EnhancedResidualDownsample3d_v3):
    """
    增强版带残差连接、且不带self-attn的降采样网络
    输入: (B, 16, T/4, H/8, W/8)
    输出: (B, 16, T/8, H/64, W/64)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        
        self.encoder1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=(2,2,2), padding=1)
        self.encoder2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=(1,2,2), padding=1)
        # # 输入投影层 - 扩展通道数
        # self.input_proj = nn.Conv3d(dim, hidden_dim, 1)
        # self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # # 三个分辨率级
        # self.encoder1 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(2,2,2))
        # self.encoder2 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2))
        # self.encoder3 = Downsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2))

        # self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        # self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        # self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        

    def forward(self, x):
        x = self.input_proj(x)

        b, c, f, h, w = x.shape
        x = self.encoder1(x)
        x1_out = self.output_proj1(x)

        b, c, f, h, w = x.shape
        x = self.encoder2(x)
        x2_out = self.output_proj2(x)

        b, c, f, h, w = x.shape
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        x = self.encoder3(x, freqs)
        x3_out = self.output_proj3(x)
        
        return x1_out, x2_out, x3_out


class EnhancedResidualUpsample3d_v4(nn.Module):
    """
    残差上采样网络
    输入: (B, 16, T/8, H/64, W/64)
    输出: (B, 16, T/4, H/8, W/8)
    """
    
    def __init__(self, dim=16, out_dim=16, hidden_dim=96, num_heads=4, eps=1e-6, depth=[2,2,2]):
        super().__init__()
        self.input_proj1 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj2 = nn.Conv3d(dim, hidden_dim, 1)
        self.input_proj3 = nn.Conv3d(dim, hidden_dim, 1)

        self.freqs = precompute_freqs_cis_3d(hidden_dim // num_heads)
        
        # 三个分辨率级
        self.decoder1 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[0], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2)),
        ])

        self.decoder2 = nn.ModuleList([
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[1], stride=(1,2,2)),
            Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2)),
        ])

        self.decoder3 = nn.ModuleList([Upsample3d_Coder_v3(hidden_dim, num_heads, eps, depth=depth[2], stride=(1,2,2))])

        self.output_proj1 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj2 = nn.Conv3d(hidden_dim, out_dim, 1)
        self.output_proj3 = nn.Conv3d(hidden_dim, out_dim, 1)
        
    def forward(self, x1, x2, x3):
        x1 = self.input_proj1(x1)
        for blk in self.decoder1:
            b, c, f, h, w = x1.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x1.device)
            x1 = blk(x1, freqs)
        x1_out = self.output_proj1(x1)


        x2 = self.input_proj2(x2)
        for blk in self.decoder2:
            b, c, f, h, w = x2.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x2.device)
            x2 = blk(x2, freqs)
        x2_out = self.output_proj2(x2)

        x3 = self.input_proj2(x3)
        for blk in self.decoder3:
            b, c, f, h, w = x3.shape
            freqs = torch.cat([
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x3.device)
            x3 = blk(x3, freqs)
        x3_out = self.output_proj3(x3)
        return x1_out, x2_out, x3_out
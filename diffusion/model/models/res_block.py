import torch
import torch.nn as nn
import torch.nn.init as init

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始层（无下采样）
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差块部分（包含下采样）
        self.layer1 = self._make_layer(64, 256, stride=2)  # 保持尺寸64x64
        self.layer2 = self._make_layer(256, 512, stride=2)  # 下采样到32x32

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock2D(in_channels, out_channels, stride, downsample),
            ResidualBlock2D(out_channels, out_channels)
        )

    def forward(self, x):
        # 输入尺寸: [batch, 64, 64, 64]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)  # 输出尺寸: [batch, 128, 64, 64]
        x = self.layer2(x)  # 输出尺寸: [batch, 256, 32, 32]
        # x = self.layer3(x)  # 输出尺寸: [batch, 512, 16, 16]
        return x
    
    def _initialize_weights(self):
        """定制化权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming初始化（适配ReLU激活）
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：权重1，偏置0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层（示例，当前网络未使用）
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)

import torch.nn.functional as F

class TokenEmbedder(nn.Module):
    def __init__(self, in_channels=64, out_channels=1024, patch_size=4):
        super(TokenEmbedder, self).__init__()
        self.proj  = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.proj2  = nn.Linear(1024,1152)
        self._initialize_weights()

    def forward(self, x, train=1):
        """
        x: Tensor of shape [B, C_in, H, W] (e.g., [B, 64, 64, 64])
        Returns:
            tokens: [B, N, C_out], where N = (H/patch_size)*(W/patch_size)
        """

        xi = x
        xi = self.proj(xi)
        token = xi.flatten(2).transpose(1, 2)                        # [B, C_out, H//p, W//p]
        token = self.proj2(token)


        return token
    
    def _initialize_weights(self):
        """定制化权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming初始化（适配ReLU激活）
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：权重1，偏置0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层（示例，当前网络未使用）
                init.normal_(m.weight, 0, 0.01)
                init.constant_(m.bias, 0)


class OverlapPatchEmbed(nn.Module):
    """分层 + 重叠 patch embedding，不改变 token 尺寸。"""
    def __init__(self, in_ch, embed_dim, patch_size, overlap):
        super().__init__()
        stride = patch_size - overlap
        self.proj = nn.Sequential(
            # 先缩小两倍，提取局部纹理
            nn.Conv2d(in_ch, in_ch*2, 3, stride=1, padding=1),  # depthwise
            nn.Conv2d(in_ch*2, embed_dim // 2, 1),  # pointwise
            nn.GELU(),
            # 再做重叠 patch
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=stride),
        )

    def forward(self, x):
        return self.proj(x)        # (B, C_embed, H', W')


class MlpTokenAdapter(nn.Module):
    """两层 MLP，通道扩张 r× → GELU → 投射回 dim。"""
    def __init__(self, dim, expansion=4, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return self.drop(x)


class TokenEmbedderV2(nn.Module):
    """
    输入 : [B, C_in, H, W]
    输出 : tokens [B, N, embed_dim]
    """
    def __init__(
        self,
        in_channels: int = 64,
        embed_dim: int = 1024,
        patch_size: int = 4,
        overlap: int = 0,          # 重叠像素
        mlp_expansion: int = 4,
        pos_embed='sine'   # "sine" | "learnable" | None
    ):
        super().__init__()
        self.patch = OverlapPatchEmbed(in_channels, embed_dim, patch_size, overlap)
        self.norm  = nn.LayerNorm(embed_dim)
        self.adapter = MlpTokenAdapter(embed_dim, expansion=mlp_expansion)

        # 可学习 LayerScale，初值 1e‑5 → 渐进放大新分支影响
        self.layer_scale = nn.Parameter(1e-5 * torch.ones(embed_dim))

        # 可选位置编码
        self.pos_embed_type = pos_embed
        self.pos_embed = None
        self.register_buffer("sine_pos", None, persistent=False)

        self._init_weights()

    # ------------------------------ forward ------------------------------
    def forward(self, x):
        """
        x : [B, C_in, H, W]
        return tokens : [B, N, embed_dim]
        """
        x = self.patch(x)                    # [B, C_e, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # → [B, N, C_e]

        if self.pos_embed_type is not None:
            pos = self._get_pos_embed(B, H, W).to(x.device, dtype=x.dtype)
            x = x + pos   # broadcast to [B, N, C_e]

        x = self.norm(x)
        x = x + self.layer_scale * self.adapter(x)
        return x

    # ------------------------------ helpers ------------------------------
    def _get_pos_embed(self, B, H, W):
        N = H * W
        if self.pos_embed_type == "learnable":
            if self.pos_embed is None or self.pos_embed.shape[1] != N:
                self.pos_embed = nn.Parameter(torch.zeros(1, N, self.norm.normalized_shape[0],
                                                   device=self.layer_scale.device))
                init.trunc_normal_(self.pos_embed, std=.02)
            return self.pos_embed
        elif self.pos_embed_type == "sine":
            if self.sine_pos is None or self.sine_pos.shape[1] != N:
                self.sine_pos = self._build_2d_sincos_pos_embed(H, W, self.norm.normalized_shape[0],
                                                                device=self.layer_scale.device)
            return self.sine_pos.repeat(B, 1, 1)
        else:
            return 0.0

    @staticmethod
    def _build_2d_sincos_pos_embed(H, W, C, device):
        grid_w = torch.arange(W, device=device)
        grid_h = torch.arange(H, device=device)
        grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='ij'), dim=0)  # (2, W, H)
        pe = []
        for i in range(C // 4):
            div_term = 1. / (10000 ** (2 * i / C))
            pe.append(torch.sin(grid * div_term))
            pe.append(torch.cos(grid * div_term))
        pe = torch.cat(pe, dim=0).reshape(C, H * W).transpose(0, 1)  # (H*W, C)
        return pe.unsqueeze(0)  # (1, N, C)

    # --------------------------- initialization ---------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

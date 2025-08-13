import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from diffusion.model.nets import PixArt_XL_C

model = PixArt_XL_C()

import math, torch, torch.nn as nn, torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    y_out = gate_base * (x Wᵀ) + gate_lora * α/r * (x Aᵀ Bᵀ)
    gate_base / gate_lora 可设逐通道或全局标量。
    """
    def __init__(self,
                 base: nn.Linear,
                 r: int = 8,
                 alpha: int = 8,
                 dropout: float = 0.0,
                 per_channel: bool = True):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

        # ----- LoRA 权重 -----
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.empty(base.out_features, r))

        # ----- 双门控 -----
        gate_shape = (base.out_features,) if per_channel else (1,)
        self.gate_base = nn.Parameter(torch.ones(gate_shape))
        self.gate_lora = nn.Parameter(torch.ones(gate_shape))

        # 冻结原始 Linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        # 统一初始化
        self._init_parameters()

    # ==================================================================
    #                      权重初始化函数
    # ==================================================================
    def _init_parameters(self):
        """自定义初始化策略：LoRA-A 用 Kaiming，LoRA-B 零初始化"""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        # 如需 LayerScale 风格：self.gate_base.data.mul_(1e-5); self.gate_lora.data.mul_(1e-5)

    # ------------------------------------------------------------------
    def forward(self, x):
        y = self.base(x)
        if self.r:
            dx = F.linear(self.dropout(x), self.A)   # (B, *, r)
            dx = F.linear(dx, self.B)                # (B, *, out)
            return self.gate_base * y + self.gate_lora * self.scale * dx
        return self.gate_base * y
    


def inject_lora_and_freeze(model: nn.Module,
                           keywords=("attn", "mlp"),
                           r=8, alpha=8, dropout=0.05):
    # 1) 全冻结
    # for p in model.parameters():
    #     p.requires_grad_(False)

    # 2) 注入
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in keywords):
            parent = model
            *path, child = name.split(".")
            for p in path: parent = getattr(parent, p)

            lora = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child, lora)

            lora_params += list(lora.parameters())
            print(f"[LoRA] {name:45s}  |  r={r}")
    return lora_params

# # 3) 打印所有子模块名称和类型
# print("=== 全模型模块列表 ===")
# for name, module in model.named_modules():
#     print(f"{name:40s}  ->  {module.__class__.__name__}")

# lora_params = inject_lora_and_freeze(model, keywords=("attn", "mlp"), r=4)
trainable = [n for n,p in model.named_parameters() if p.requires_grad]


untrainable = [n for n,p in model.named_parameters() if not p.requires_grad]
for n in untrainable[:100]:
    print("  •", n)

print("Trainable params:", len(trainable), "/", sum(1 for _ in model.parameters()))
for n in trainable[:100]:
    print("  •", n)
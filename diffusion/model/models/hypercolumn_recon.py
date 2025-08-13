
import torch
from torchvision import transforms
from diffusion.model.models.hypercolumn.vit_pytorch.train_V1_sep_new import Column_trans_rot_lgn
import numpy as np
from einops import rearrange, repeat
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.serialization
from argparse import Namespace


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class HyperColumnLGN(nn.Module):
    def __init__(self,a=1,key=0,hypercond=[2],size=512,restore_ckpt = '/home/easytaker/code/generate/infinity/Infinity/models/hypercolumn/checkpoint/imagenet/equ_nv16_vl4_rn1_Bipolar_norm.pth'):
        super().__init__()
        ckpt = torch.load(restore_ckpt, weights_only=False)
        hc = Column_trans_rot_lgn(ckpt['arg'])
        hc.load_state_dict(ckpt['state_dict'], strict=False)
        self.lgn_ende = hc.lgn_ende[0].eval()
        self.lgn_ende.train = disabled_train
        for param in self.lgn_ende.parameters():
            param.requires_grad = False

        self.resize = transforms.Resize(size)
        if size == 128:
            self.pad = nn.ConstantPad2d((1,1,1,1),0.)
        else:
            self.pad = nn.Identity()
        
        self.groups = [[0,1,4,8,9,15],[2,3],[5,12],[10],[6,7],[11,13,14]]
        # self.groups = [[1],[2,3],[5,12],[10],[6,7],[11,13,14]]

        norm_mean = np.array([0.50705882, 0.48666667, 0.44078431])
        norm_std = np.array([0.26745098, 0.25568627, 0.27607843])
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.cond = hypercond
        self.slct = None

    def forward(self,x,i=0):
        b,_,h,w = x.shape
        c = 16
        r = torch.zeros(b, c, 1, 1, device=x.device, dtype=x.dtype) 
        r[:,self.groups[i],:,:]=1
        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        # out = self.lgn_ende(self.norm(x))
        out = self.lgn_ende(x)*r
        out = self.pad(self.lgn_ende.deconv(out))
        out = F.interpolate(out, size=( h, w), mode='bilinear', align_corners=False)
        out = out - out.min()
        out = out/(out.max() + 0.00001)
        return out*2-1
    def to_embedding(self,x,i=0):
        b,_,h,w = x.shape
        c = 16
        r = torch.zeros(b, c, 1, 1, device=x.device, dtype=x.dtype) 
        r[:,self.groups[i],:,:]=1
        r = repeat(r,'n c h w -> n (c repeat) h w',repeat=4)
        out = self.lgn_ende(x)*r

        return out
    

# def process_and_concatenate(
#     image1_path: str,
#     image2_path: str,
#     model: torch.nn.Module,
#     save_path: str,
#     device: str = "cpu" if torch.cuda.is_available() else "cpu"
# ) -> None:
#     """
#     处理并拼接图像的完整流程
    
#     参数:
#         image1_path: 第一张图像的路径
#         image2_path: 第二张图像的路径
#         model: 预加载的PyTorch模型
#         save_path: 结果保存路径
#         device: 使用的计算设备
#     """
#     # 确保模型在指定设备上
#     model = model.to(device)
#     model.eval()
    
#     # 定义预处理（根据模型需求调整）
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     # 读取并处理两张图像
#     with torch.no_grad():
#         # 处理第一张图像
#         img1 = Image.open(image1_path).convert("RGB")
#         tensor1 = preprocess(img1).unsqueeze(0).to(device)
#         output1 = model(tensor1,1)
        
#         # 处理第二张图像
#         img2 = Image.open(image2_path).convert("RGB")
#         tensor2 = preprocess(img2).unsqueeze(0).to(device)
#         output2 = model(tensor2,1)
    
#     # 后处理（反归一化）
#     postprocess = transforms.Compose([
#         # 若输出需要恢复为[0,255]整型
#         transforms.Lambda(lambda x: (x * 255).byte()),  # 仅当输出在[0,1]时有效
#         transforms.ToPILImage()
#     ])
#     # 转换为PIL图像
#     proc1 = postprocess(output1.squeeze().cpu())
#     proc2 = postprocess(output2.squeeze().cpu())
    
#     # 拼接图像（水平拼接）
#     concatenated = Image.new("RGB", (proc1.width + proc2.width, max(proc1.height, proc2.height)))
#     concatenated.paste(proc1, (0, 0))
#     concatenated.paste(proc2, (proc1.width, 0))
    
#     # 保存结果
#     concatenated.save(save_path)
#     print(f"结果已保存至: {save_path}")


# # ----------------- 使用示例 -----------------
# if __name__ == "__main__":
#     # 示例模型定义（替换为你的实际模型）
#     class ExampleModel(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
            
#         def forward(self, x):
#             return torch.clamp(self.conv(x), 0, 1)
    
#     # 初始化模型
#     model = HyperColumnLGN()
    
#     # 运行处理
#     process_and_concatenate(
#         image1_path="/mnt/data/guochengwang/data/fusion/MSRS/train/vi/00001D.png",
#         image2_path="/mnt/data/guochengwang/data/fusion/MSRS/train/vi/00002D.png",
#         model=model,
#         save_path="/mnt/data/guochengwang/code2/VAR_Hyper_generate/vis/concat_result.jpg"
#     )
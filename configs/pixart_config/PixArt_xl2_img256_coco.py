_base_ = ['../PixArt_xl2_sam.py']
data_root = '/home/easytaker/data/coc2017/sa'
image_list_txt = ['part0.txt']
data = dict(type='SAM_coco', root='SA1B', image_list_txt=image_list_txt, transform='default_train', load_vae_feat=True)
image_size = 256

# model setting
window_block_indexes=[]
window_size=0
use_rel_pos=False
model = 'PixArt_XL_all_C'
fp32_attention = True
load_from = '/home/easytaker/code/generate/dit-base/PixArt-alpha/output/pretrained_models/PixArt-XL-2-SAM-256x256.pth'
vae_pretrained = "/home/easytaker/sd-vae-ft-ema"

# training setting
use_fsdp=False   # if use FSDP mode
num_workers=10

train_batch_size = 12 # 32
num_epochs = 200 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

eval_sampling_steps = 200
log_interval = 200
save_model_epochs=1
save_model_steps=20000
work_dir = '/home/easytaker/code/generate/dit-base/PixArt-alpha/output/debug/PixArt_XL_all_C_1'

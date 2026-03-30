# BrainSee-ModuleNet

- [BrainSee-ModuleNet](#BrainSee-ModuleNet)
Our model is built upon our trained modular Hypercolumn-like features and PixArt. The hypercolumn-like features are trained using a self-supervised method. Based on these features, we train conditional Pixart. In reference time, it can spontaneously exhibit image editing capabilities. Refer to https://github.com/PixArt-alpha/PixArt-alpha.

### installation
```
pip install -r requirements.txt
```

### Training
```
sh train.sh
```

### checkpoints
https://huggingface.co/Wistons/BrainSee-ModuleNet

### Inference
#### Prepare data
Put reference image into file './img_input'
![img](./img_input/000000000030.jpg)

write image path and prompt text into './asset/sample_3.txt'
![text](./asset/samples_3.txt)

#### Run
```
python ./scripts/inference_cross_lora.py
```

#### Result
Generated image is saved in './output/txt-img/vis/2025-08-08_custom_epoch144_step133200_scale4.5_step20_size256_bs1_sampdpm-solver_seed0'

![img](./output/txt-img/vis/2025-08-08_custom_epoch144_step133200_scale4.5_step20_size256_bs1_sampdpm-solver_seed0/A%20flower%20vase%20is%20sitting%20on%20a%20porch%20stand..jpg)
![img](./output/txt-img/vis/2025-08-08_custom_epoch144_step133200_scale4.5_step20_size256_bs1_sampdpm-solver_seed0/A%20flower%20vase%20is%20sitting%20on%20a%20porch%20stand.%20the%20color%20of%20vase%20is%20yellow..jpg)
![img](./output/txt-img/vis/2025-08-08_custom_epoch144_step133200_scale4.5_step20_size256_bs1_sampdpm-solver_seed0/A%20flower%20vase%20is%20sitting%20on%20a%20porch%20stand.%20the%20color%20of%20vase%20is%20blue..jpg)
![img](./output/txt-img/vis/2025-08-08_custom_epoch144_step133200_scale4.5_step20_size256_bs1_sampdpm-solver_seed0/A%20flower%20vase%20is%20sitting%20on%20a%20porch%20stand.%20the%20color%20of%20vase%20is%20red..jpg)


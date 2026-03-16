import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm_eval.models.huggingface import HFLM
from ..quantization.step_aware_linear import load_stepaware_model, StepAwareFakeLinear

# 加载模型路径
model_path = "/share/MY-DAPO/DAPO-WITH-Z/Qwen3-4B-AWQ-w4g128-Soft-Only-freeze-lr_1e-4/global_step_50_hf"

# 加载 HFLM 模型
hflm = HFLM(
    pretrained=model_path,
    batch_size=16,
    parallelize=True,           # 对齐 CLI
    trust_remote_code=True,
)

# 加载 StepAware 模型
hflm._model, _ = load_stepaware_model(model_path, wbits=4, tbits=2, group_size=128)


total = 0
for name, param in hflm._model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

for name, module in hflm._model.named_modules():
    if isinstance(module, StepAwareFakeLinear) and not 'head' in name:
        module.weight.requires_grad = True


for n,p in hflm._model.named_parameters():
    if "weight" in n and p.requires_grad:
        total += p.numel()
        print(n, p.numel())

print("trainable weight:", total)

# 验证所有 buffers
def check_buffers(model):
    for name, module in model.named_modules():
        # 检查 running_xmin 和 running_xmax
        if hasattr(module, 'running_xmin') and hasattr(module, 'running_xmax'):
            print(f"{name}.running_xmin: {module.running_xmin}")
            print(f"{name}.running_xmax: {module.running_xmax}")
        
        # 检查量化状态相关的 buffers
        if hasattr(module, 'soft_round_enable') and hasattr(module, 'progressive_enable'):
            print(f"{name}.soft_round_enable: {module.soft_round_enable}")
            print(f"{name}.progressive_enable: {module.progressive_enable}")
            print(f"{name}.progressive_ratio: {module.progressive_ratio}")

# 执行检查
check_buffers(hflm._model)

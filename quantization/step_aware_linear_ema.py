import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import gc
import os
from tqdm import tqdm
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from .utils import get_named_linears, set_op_by_name


CLIPMIN=1e-6

TRAINING_MODE   = 0
INFER_MODE      = 1

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def round_soft(x: torch.Tensor, qmax: int):
    t = x.new_tensor(1000.0)

    k = torch.arange(
        qmax,
        device=x.device,
        dtype=x.dtype,
    ).view(-1, *([1] * x.dim()))

    # ---- 主计算 ----
    term1 = torch.tanh(t * (x.unsqueeze(0) - (k + 0.5)))
    term2 = torch.tanh(t * (k + 0.5))
    denom = 2.0 * torch.tanh(t)

    delta = (term1 + term2).sum(dim=0) / denom

    return delta

def round_soft_aligned_general(
    x: torch.Tensor,
    *,
    s_c: int, 
    t_c: int, 
    t: float = 1000.0,
):
    """
    Low-bit soft rounding embedded into full-bit integer lattice.
    """
    device = x.device
    dtype = x.dtype

    qmax_s = s_c - 1
    qmax_t = t_c - 1

    assert qmax_s >= qmax_t
    assert qmax_t > 1

    # ---- compute full-lattice indices to keep ----
    # k_i = floor((i + 0.5) * qmax_s / qmax_t)
    i = torch.arange(
        qmax_t,
        device=device,
        dtype=dtype,
    )

    k = torch.floor((i + 0.5) * qmax_s / qmax_t)

    # ensure valid range
    k = k.clamp(min=0, max=qmax_s-1)

    # shape: (num_steps, 1, ..., 1)
    k = k.view(-1, *([1] * x.dim()))

    t = x.new_tensor(t)

    term1 = torch.tanh(t * (x.unsqueeze(0) - (k + 0.5)))
    term2 = torch.tanh(t * (k + 0.5))
    denom = 2.0 * torch.tanh(t)

    delta = (qmax_s / qmax_t) * (term1 + term2).sum(dim=0) / denom
    return delta


class WeightQuantizer(nn.Module):
    def __init__(self, s_bits=4, t_bits=2, group_size=128):
        super().__init__()

        self.s_bits = s_bits
        self.t_bits = t_bits

        self.qmin_full = 0
        self.qmax_full = (1 << s_bits) - 1
        self.qmax_target = (1 << t_bits) - 1

        self.group_size = group_size
        self.ema_decay = 0.99

        self.register_buffer("running_xmin",torch.empty(1,1))
        self.register_buffer("running_xmax",torch.empty(1,1))

        self.initialized = False
        self.mode = TRAINING_MODE

    def fake_quant(
        self,
        x: torch.Tensor,
        *,
        progressive_enable: bool = False,
        progressive_ratio: float = 0.0,
        soft_round_enable: bool = True,
    ):
        dim1, dim2 = x.shape
        if self.group_size != -1:
            assert dim2 % self.group_size == 0
        x = x.reshape(-1, self.group_size)

        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)

        if self.mode == TRAINING_MODE:
            if not self.initialized:
                self.running_xmin = xmin.detach().clone()
                self.running_xmax = xmax.detach().clone()
                self.initialized = True
            else:
                self.running_xmin.mul_(self.ema_decay).add_(xmin.detach() * (1 - self.ema_decay))
                self.running_xmax.mul_(self.ema_decay).add_(xmax.detach() * (1 - self.ema_decay))

        xmin = self.running_xmin
        xmax = self.running_xmax

        scale = (xmax - xmin) / (self.qmax_full - self.qmin_full)
        scale = scale.clamp(min=CLIPMIN)
        zero = - xmin / scale

        # ---- base rounding (full bits) ----
        z_int = torch.round(zero).detach()
        if soft_round_enable:
            x_fp = x / scale + z_int
            x_int = round_soft(x_fp, math.ceil(self.qmax_full))
            if progressive_enable:
                x_int_t = round_soft_aligned_general(
                    x_fp,
                    s_c=self.qmax_full+1,
                    t_c=self.qmax_target+1,
                    t=1000.0,
                )
                # ΔW = Q2 - Q4
                delta = (x_int_t - x_int).detach()
                # Q4 + r ΔW
                x_int = x_int + progressive_ratio * delta
        else:
            x_fp = x / scale
            x_int = round_ste(x_fp) + z_int
            x_int = x_int.clamp(self.qmin_full, self.qmax_full)


        x_dequant = (x_int - z_int) * scale
        return x_dequant.reshape(dim1, dim2)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.s_bits == 16:
            return x
        return self.fake_quant(x, **kwargs)


class StepAwareFakeLinear(nn.Module):
    def __init__(self, org_module: nn.Linear, s_bits=4, t_bits=2, group_size=128):
        super().__init__()
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features

        self.register_parameter("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_parameter("bias", org_module.bias)
        else:
            self.bias = None

        self.fake_quant_weight = WeightQuantizer(
            s_bits=s_bits,
            t_bits=t_bits,
            group_size=group_size,
        )

        # QuantState cache (由 worker / trainer 注入)
        self.register_buffer("soft_round_enable", torch.zeros((), dtype=torch.int8))
        self.register_buffer("progressive_enable", torch.zeros((), dtype=torch.int8))
        self.register_buffer("progressive_ratio", torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def set_quant_state(self, state):
        """
        state: QuantState
        """
        device = self.weight.device

        self.soft_round_enable = self.soft_round_enable.to(device)
        self.progressive_enable = self.progressive_enable.to(device)
        self.progressive_ratio = self.progressive_ratio.to(device)

        self.soft_round_enable.fill_(int(state.soft_round_enable))
        self.progressive_enable.fill_(int(state.progressive_enable))
        self.progressive_ratio.fill_(float(state.progressive_ratio))

    def forward(self, input):
        weight_q = self.fake_quant_weight(
            self.weight,
            soft_round_enable=bool(self.soft_round_enable.item()),
            progressive_enable=bool(self.progressive_enable.item()),
            progressive_ratio=self.progressive_ratio.item(),
        )
        weight_q = weight_q.to(input.dtype)
        return F.linear(input, weight_q, self.bias)    
    

@torch.no_grad()
def convert_to_fake_quant(model):
    print("Converting StepAwareFakeLinear → FakeQuant Linear")

    replace_ops = []

    for name, module in model.named_modules():
        if isinstance(module, StepAwareFakeLinear):
            replace_ops.append((name, module))

    for name, module in tqdm(replace_ops):
        weight = module.weight.data

        # ---------- fake quant ----------
        qweight = module.fake_quant_weight(
            weight,
            soft_round_enable=bool(module.soft_round_enable.item()),
            progressive_enable=bool(module.progressive_enable.item()),
            progressive_ratio=module.progressive_ratio.item()
        )

        # ---------- build new Linear ----------
        new_linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=weight.device,
            dtype=qweight.dtype,
        )

        new_linear.weight.data.copy_(qweight)

        if module.bias is not None:
            new_linear.bias.data.copy_(module.bias.data)

        # ---------- replace module ----------
        set_op_by_name(model, name, new_linear)

    print("Conversion finished.")


def load_running_stats(model_path: str):
    print("Inferring running stats shape from checkpoint...")

    safetensor_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    
    if not safetensor_files:
        raise ValueError("No .safetensors files found in the model path.")

    running_shapes = {}

    # 遍历所有safetensor文件
    for safetensor_file in safetensor_files:
        safetensor_path = os.path.join(model_path, safetensor_file)

        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "running_xmin" in k or "running_xmax" in k:
                    running_shapes[k] = f.get_tensor(k).shape
                    print(f"matched: {k} {f.get_tensor(k).shape}")

    return running_shapes

def update_buffers_from_checkpoint(model, running_shapes):
    # ---------- allocate buffers from checkpoint ----------
    for ckpt_key, shape in running_shapes.items():

        if ckpt_key.endswith("running_xmin"):
            module_path = ckpt_key[:-len(".running_xmin")]
            try:
                module = model.get_submodule(module_path)
                module._buffers["running_xmin"] = torch.zeros(shape)
                # print(f"set xmin: {module_path} {shape}")
            except Exception:
                continue

        if ckpt_key.endswith("running_xmax"):
            module_path = ckpt_key[:-len(".running_xmax")]
            try:
                module = model.get_submodule(module_path)
                module._buffers["running_xmax"] = torch.zeros(shape)
                # print(f"set xmax: {module_path} {shape}")
            except Exception:
                continue

def load_stepaware_model(model_path: str, wbits: int = 4, tbits: int = 2, group_size: int = 128):
    print(f"Loading StepAware model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)

    # ---------- build empty model ----------
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    layers = model.model.layers

    # ---------- replace Linear BEFORE loading ----------
    print("Replacing Linear with StepAwareFakeLinear...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = StepAwareFakeLinear(module, s_bits=wbits, t_bits=tbits, group_size=group_size)
            q_linear.fake_quant_weight.mode = INFER_MODE
            
            set_op_by_name(layer, name, q_linear)

    model.tie_weights()

    device_map = infer_auto_device_map(model)

    # ---------- load running stats (xmin, xmax) ----------
    running_shapes = load_running_stats(model_path)

    # ---------- update buffers based on running stats ----------
    update_buffers_from_checkpoint(model, running_shapes)

    # ---------- load checkpoint ----------
    print("Loading checkpoint...")
    load_checkpoint_in_model(
        model,
        checkpoint=model_path,
        device_map=device_map,
        offload_state_dict=True,
    )
    print("Checkpoint loaded successfully.")

    return model, tokenizer
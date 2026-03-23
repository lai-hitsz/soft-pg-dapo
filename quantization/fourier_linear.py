import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from .utils import get_named_linears, set_op_by_name

CLIPMIN = 1e-6


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x


def round_fourier(x, N=4, t=1.0, k=1.0):

    series = torch.zeros_like(x)

    sign = 1.0
    w = 2 * math.pi * x / k

    for n in range(1, N + 1):
        series += sign * torch.sin(n * w) / n
        sign = -sign

    return x - k * series / (t * math.pi)


def get_t_k(progress,
            k_base,
            k_target,
            t_min=1.0,
            t_max=100.0):

    if progress < 0.5:
        # Phase A: 变软（去量化）
        alpha = progress / 0.5
        t = t_min + alpha * (t_max - t_min)
        k = k_base

    elif progress == 0.5:
        # Phase B: 切 k（在 identity 附近）
        t = t_max
        k = k_target

    else:
        # Phase C: 重新变硬（恢复量化）
        gamma = (progress - 0.5) / 0.5
        t = t_max - gamma * (t_max - t_min)
        k = k_target

    return t, k

class WeightQuantizer(nn.Module):

    def __init__(self, s_bits=4, t_bits=2, group_size=128):
        super().__init__()

        self.s_bits = s_bits
        self.t_bits = t_bits

        self.qmin_full = 0
        self.qmax_full = (1 << s_bits) - 1
        self.qmax_target = (1 << t_bits) - 1

        self.group_size = group_size

        # Fourier parameters
        self.fourier_terms = 6
        self.temperature = 1.0
        self.k_value = 1

    def fake_quant(
        self,
        x: torch.Tensor,
        *,
        progressive_enable: bool = True,
        progressive_ratio: float = 0.0,
        soft_round_enable: bool = True,
    ):

        dim1, dim2 = x.shape

        if self.group_size != -1:
            assert dim2 % self.group_size == 0

        x = x.reshape(-1, self.group_size)

        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)

        scale = (xmax - xmin) / (self.qmax_full - self.qmin_full)
        scale = scale.clamp(min=CLIPMIN)

        zero = -xmin / scale
        z_int = torch.round(zero).detach()
        x_fp = x / scale + z_int

        if soft_round_enable:
            k = self.k_value
            t = self.temperature

            if progressive_enable:
                # TODO: progressive quantization
                # pass
                # k = self.k_value + progressive_ratio * 1.0
                t, k = get_t_k(progressive_ratio, self.k_value, 2)

            x_int = round_fourier(
                x_fp,
                N=self.fourier_terms,
                t=t,
                k=k
            )
        else:
            x_int = round_ste(x_fp)
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

        self.register_buffer("soft_round_enable", torch.zeros((), dtype=torch.int8))
        self.register_buffer("progressive_enable", torch.zeros((), dtype=torch.int8))
        self.register_buffer("progressive_ratio", torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def set_quant_state(self, state):

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
def convert_to_fake_quant(model, args):

    print("Converting StepAwareFakeLinear → FakeQuant Linear")

    replace_ops = []

    for name, module in model.named_modules():
        if isinstance(module, StepAwareFakeLinear):
            replace_ops.append((name, module))

    for name, module in tqdm(replace_ops):

        weight = module.weight.data

        qweight = module.fake_quant_weight(
            weight,
            soft_round_enable=bool(module.soft_round_enable.item()),
            # soft_round_enable=True,
            progressive_enable=bool(module.progressive_enable.item()),
            # progressive_enable=args.use_pg,
            progressive_ratio=module.progressive_ratio.item(),
            # progressive_ratio=0.0,
        )

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

        set_op_by_name(model, name, new_linear)

    print("Conversion finished.")


def load_stepaware_model(model_path: str, wbits: int = 4, tbits: int = 2, group_size: int = 128):

    print(f"Loading StepAware model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)

    with init_empty_weights():

        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    layers = model.model.layers

    print("Replacing Linear with StepAwareFakeLinear...")

    for i in tqdm(range(len(layers))):

        layer = layers[i]

        named_linears = get_named_linears(layer, torch.nn.Linear)

        for name, module in named_linears.items():

            q_linear = StepAwareFakeLinear(
                module,
                s_bits=wbits,
                t_bits=tbits,
                group_size=group_size,
            )

            set_op_by_name(layer, name, q_linear)

    model.tie_weights()
    tokenizer.model_max_length = 1024

    device_map = infer_auto_device_map(model)

    print("Loading checkpoint...")

    load_checkpoint_in_model(
        model,
        checkpoint=model_path,
        device_map=device_map,
        offload_state_dict=True,
    )

    print("Checkpoint loaded successfully.")

    return model, tokenizer
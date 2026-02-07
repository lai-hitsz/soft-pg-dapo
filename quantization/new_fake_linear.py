import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIPMIN = 1e-6

def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding."""
    return (x.round() - x).detach() + x

def round_soft(x: torch.Tensor, qmax: int):
    """Soft rounding for differentiable quantization."""
    t = x.new_tensor(1000.0)
    k = torch.arange(qmax, device=x.device, dtype=x.dtype).view(-1, *([1] * x.dim()))
    term1 = torch.tanh(t * (x.unsqueeze(0) - (k + 0.5)))
    term2 = torch.tanh(t * (k + 0.5))
    denom = 2.0 * torch.tanh(t)
    delta = (term1 + term2).sum(dim=0) / denom
    return delta

class WeightQuantizer(nn.Module):
    def __init__(self, s_bits: int = 4, t_bits: int = 2, group_size: int = 128):
        super().__init__()
        self.s_bits = s_bits
        self.t_bits = t_bits
        self.group_size = group_size

        self.qmin_full = 0
        self.qmax_full = (1 << s_bits) - 1
        self.qmax_target = (1 << t_bits) - 1

        # rollout-only full-view buffers (FSDP2: not sharded)
        self.register_buffer("cached_scale", torch.empty(0))
        self.register_buffer("cached_zero", torch.empty(0))
        self.register_buffer("cached_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("cached_eff", torch.zeros((), dtype=torch.int8))

    @torch.no_grad()
    def update_scale_zero(
        self,
        weight: torch.Tensor,
        progressive_enable: bool = False,
        progressive_ratio: float = 0.0,
    ):
        if progressive_enable:
            qmax_eff = self.qmax_full - progressive_ratio * (
                self.qmax_full - self.qmax_target
            )
        else:
            qmax_eff = self.qmax_full

        out_dim, in_dim = weight.shape
        group_size = self.group_size if self.group_size != -1 else in_dim
        assert in_dim % group_size == 0

        x = weight.reshape(-1, group_size)

        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)

        scale = (xmax - xmin) / (qmax_eff - self.qmin_full)
        scale = scale.clamp(min=CLIPMIN)
        zero = -xmin / scale

        # in-place update buffers (important for FSDP / state_dict safety)
        self.cached_scale.resize_as_(scale).copy_(scale)
        self.cached_zero.resize_as_(zero).copy_(zero)
        self.cached_eff.fill_(math.ceil(qmax_eff))
        self.cached_ready.fill_(True)

    def fake_quant(
        self,
        x: torch.Tensor,
        soft_round_enable: bool = True,
        progressive_enable: bool = False,
        progressive_ratio: float = 0.0,
    ):
        out_dim, in_dim = x.shape
        group_size = self.group_size if self.group_size != -1 else in_dim
        x_ = x.reshape(-1, group_size)

        if self.cached_ready.item():
            scale = self.cached_scale
            zero = self.cached_zero
            qmax_eff = self.cached_eff
        else:
            xmin = x_.amin(dim=-1, keepdim=True)
            xmax = x_.amax(dim=-1, keepdim=True)
            scale = (xmax - xmin) / (self.qmax_full - self.qmin_full)
            scale = scale.clamp(min=CLIPMIN)
            zero = -xmin / scale
            qmax_eff = self.qmax_full

        z_int = torch.round(zero)

        if soft_round_enable:
            x_fp = x_ / scale + z_int
            x_int = round_soft(x_fp, math.ceil(qmax_eff))
        else:
            x_fp = x_ / scale
            x_int = round_ste(x_fp) + z_int
            x_int = x_int.clamp(self.qmin_full, qmax_eff)

        x_dequant = (x_int - z_int) * scale
        return x_dequant.reshape(out_dim, in_dim)


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

        self.fake_quant_weight = WeightQuantizer(s_bits=s_bits, t_bits=t_bits, group_size=group_size)

        # Quantization state
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

    @torch.no_grad()
    def prepare_shard_quant(
        self,
        weight: torch.Tensor,
    ):
        self.fake_quant_weight.update_scale_zero(
            weight,
            progressive_enable=bool(self.progressive_enable.item()),
            progressive_ratio=float(self.progressive_ratio.item()),
        )

    def forward(self, input):
        weight_q = self.fake_quant_weight(
            self.weight,
            soft_round_enable=bool(self.soft_round_enable.item()),
            progressive_enable=bool(self.progressive_enable.item()),
            progressive_ratio=self.progressive_ratio.item()
        )
        return F.linear(input, weight_q, self.bias)

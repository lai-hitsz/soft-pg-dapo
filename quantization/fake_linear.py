import torch
import torch.nn as nn
import torch.nn.functional as F


CLIPMIN=1e-4


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


class WeightQuantizer(nn.Module):
    def __init__(self, s_bits: int=4, t_bits: int=0, group_size: int=128):
        super().__init__()
        assert 2 <= s_bits <= 4 or s_bits == 16
        self.s_bits = s_bits
        self.t_bits = t_bits
        self.qmin = 0
        self.qmax_full = (1 << s_bits) - 1
        self.qmax_final = (1 << t_bits) - 1
        self.group_size = group_size

        self.duration = None
        if t_bits >= 2 and t_bits <= s_bits:
            self.duration = self.qmax_full / self.qmax_final

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
        scale = (xmax - xmin) / (self.qmax_full - self.qmin)
        scale = scale.clamp(min=CLIPMIN)
        zero_point = -(xmin / scale)

        x_fp = (x - xmin) / scale

        # ---- base rounding (full bits) ----
        if soft_round_enable:
            x_int = round_soft(x_fp, self.qmax_full)
        else:
            x_int = round_ste(x_fp)

        x_int = x_int.clamp(self.qmin, self.qmax_full)

        # ---- progressive quantization ----
        if progressive_enable:
            assert self.duration
            delta = 1.0 + progressive_ratio * (self.duration - 1.0)

            qmax_eff = torch.floor(
                x.new_tensor(self.qmax_full / delta)
            )

            x_c = x_int / delta

            if soft_round_enable:
                x_c = round_soft(x_c, int(qmax_eff.item()))
            else:
                x_c = round_ste(x_c)

            x_c = x_c.clamp(self.qmin, qmax_eff)
            x_int = x_c * delta

        x_dequant = x_int * scale + xmin
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
        if self.soft_round_enable.item() == 0:
            return F.linear(input, self.weight, self.bias)

        weight_q = self.fake_quant_weight(
            self.weight,
            soft_round_enable=bool(self.soft_round_enable.item()),
            progressive_enable=bool(self.progressive_enable.item()),
            progressive_ratio=self.progressive_ratio.item(),
        )
        return F.linear(input, weight_q, self.bias)


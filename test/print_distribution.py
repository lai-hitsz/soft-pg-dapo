import torch
from transformers import AutoModelForCausalLM
from typing import Dict, List


# =========================
# 量化函数
# =========================

def fake_quant_affine(
    w: torch.Tensor,
    qmin: int,
    qmax: int,
):
    xmin = w.amin(dim=-1, keepdim=True)
    xmax = w.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin) / (qmax - qmin)
    scale = scale.clamp(min=1e-6)
    zero = -xmin / scale

    z_int = torch.round(zero)
    x_int = torch.round(w / scale) + z_int
    x_int = x_int.clamp(qmin, qmax)
    x_dequant = (x_int - z_int) * scale
    return x_int, x_dequant, scale, z_int


# =========================
# HF 加载
# =========================

def load_hf_model(model_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()
    return model


# =========================
# 核心分析函数（只打印）
# =========================

@torch.no_grad()
def analyze_param_diff_across_steps(
    ckpt_dirs: Dict[str, str],      # {"step0": path, "step100": path, ...}
    param_names: List[str],         # list of parameter names
    qmin: int = 0,
    qmax: int = 15,
):
    steps = list(ckpt_dirs.keys())
    assert len(steps) >= 2

    # ---------- load all params ----------
    params = {name: {} for name in param_names}

    for step, ckpt in ckpt_dirs.items():
        model = load_hf_model(ckpt)
        sd = model.state_dict()
        for name in param_names:
            if name not in sd:
                print(f"[WARN] {name} not found in {step}")
                continue
            params[name][step] = sd[name].float()
        del model

    # ---------- analyze ----------
    for name in param_names:
        print("\n" + "=" * 80)
        print(f"Parameter: {name}")
        print("=" * 80)

        for i in range(1, len(steps)):
            s0, s1 = steps[i - 1], steps[i]
            if s0 not in params[name] or s1 not in params[name]:
                continue

            w0 = params[name][s0]
            w1 = params[name][s1]

            # ---- FP diff ----
            dfp = w1 - w0

            # ---- Quant diff (if tensor is matrix-like) ----
            if w0.dim() >= 2:
                w0r = w0.reshape(-1, 128)
                w1r = w1.reshape(-1, 128)

                wi0, _, sc0, z0 = fake_quant_affine(w0r, qmin, qmax)
                wi1, _, sc1, z1 = fake_quant_affine(w1r, qmin, qmax)

                dint = wi1 - wi0
                dscale = sc1 - sc0
                dz = z1 - z0

                mean_dint = dint.abs().mean().item()
                max_dint = dint.abs().max().item()
                mean_dscale = dscale.abs().mean().item()
                mean_dz = dz.abs().mean().item()

                # FP 是否足以跨 bin
                mean_scale = sc0.mean().item()
                fp_over_scale = (dfp.abs() / mean_scale).mean().item()
            else:
                mean_dint = max_dint = mean_dscale = mean_dz = float("nan")
                fp_over_scale = float("nan")

            print(f"[{s0} → {s1}]")
            print(f"  FP   mean |Δw| : {dfp.abs().mean().item():.3e}")
            print(f"  FP   max  |Δw| : {dfp.abs().max().item():.3e}")
            print(f"  INT  mean |Δw| : {mean_dint:.3e}")
            print(f"  INT  max  |Δw| : {max_dint:.3e}")
            print(f"  mean |Δscale| : {mean_dscale:.3e}")
            print(f"  mean |Δz_int| : {mean_dz:.3e}")
            print(f"  mean |Δw_fp| / scale : {fp_over_scale:.3e}")

        print("\n")


# =========================
# 使用示例
# =========================

if __name__ == "__main__":
    ckpt_dirs = {
        "step0": "/root/lai-code/quant_models/qwen3-4b-instruct/fake_quant_model",
        "step100": "/share/MY-DAPO/Qwen3-4B-AWQ-w4g128-Soft-Only/global_step_100_hf",
        "step200": "/share/MY-DAPO/Qwen3-4B-AWQ-w4g128-Soft-Only/global_step_200_hf",
    }

    param_names = [
        # MLP
        "model.layers.10.mlp.up_proj.weight",
        "model.layers.10.mlp.gate_proj.weight",
        "model.layers.10.mlp.down_proj.weight",

        # Attention
        "model.layers.10.self_attn.o_proj.weight",

        # Norm
        "model.layers.10.input_layernorm.weight",
    ]

    analyze_param_diff_across_steps(
        ckpt_dirs=ckpt_dirs,
        param_names=param_names,
        qmin=0,
        qmax=15,
    )

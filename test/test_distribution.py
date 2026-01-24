import os
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from transformers import AutoModelForCausalLM


# =========================
# 量化函数
# =========================

def fake_quant_affine(
    w: torch.Tensor,
    qmin: int,
    qmax: int,
):
    x = w

    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin) / (qmax - qmin)
    scale = scale.clamp(min=1e-6)
    zero = - xmin / scale

    # ---- base rounding (full bits) ----
    z_int = torch.round(zero)
    x_fp = x / scale
    x_int = torch.round(x_fp) + z_int
    x_int = x_int.clamp(qmin, qmax)

    x_dequant = (x_int - z_int) * scale
    return x_int, x_dequant, scale, z_int


# =========================
# 绘图（只保存，不 show）
# =========================

def plot_hist_save(
    tensors: List[torch.Tensor],
    labels: List[str],
    title: str,
    save_path: str,
    bins: int = 200,
):
    plt.figure(figsize=(6, 4))
    for t, label in zip(tensors, labels):
        plt.hist(
            t.flatten().cpu().numpy(),
            bins=bins,
            density=True,
            alpha=0.5,
            label=label,
            histtype="step"
        )
    plt.legend()
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_int_hist_save(
    tensors: List[torch.Tensor],
    labels: List[str],
    qmin: int,
    qmax: int,
    title: str,
    save_path: str,
):
    plt.figure(figsize=(6, 3))
    for t, label in zip(tensors, labels):
        t = t.flatten().long().clamp(qmin, qmax)
        hist = torch.bincount(
            t - qmin,
            minlength=(qmax - qmin + 1),
        ).float()
        hist /= hist.sum()
        plt.plot(range(qmin, qmax + 1), hist.cpu(), label=label)

    plt.legend()
    plt.xlabel("quant index")
    plt.ylabel("probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# HF 模型加载（safetensors）
# =========================

def load_hf_model(
    model_dir: str,
    device: str = "cpu",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()
    return model


# =========================
# 主分析逻辑
# =========================

def analyze_layer_across_steps_hf(
    ckpt_dirs: Dict[str, str],  # {"step2": "checkpoint-2", ...}
    weight_name: str,
    qmin: int,
    qmax: int,
    out_root: str = "figs",
):
    steps = list(ckpt_dirs.keys())

    # ---------- 输出目录 ----------
    safe_name = weight_name.replace(".", "_")
    out_dir = os.path.join(out_root, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load fp weights ----------
    w_fp = {}
    for step, ckpt_dir in ckpt_dirs.items():
        model = load_hf_model(ckpt_dir)
        w_fp[step] = model.state_dict()[weight_name].float().reshape(-1, 128)
        del model

    # ---------- quant ----------
    w_int, w_dequant = dict(), dict()
    scale_temp, zint_temp = dict(), dict()
    for step in steps:
        wi, wd, scale, z_int = fake_quant_affine(
            w_fp[step],
            qmin,
            qmax,
        )
        w_int[step] = wi
        w_dequant[step] = wd

        scale_temp[step] = scale
        zint_temp[step] = z_int

    # ======================
    # pre-round FP
    # ======================
    plot_hist_save(
        [w_fp[s] for s in steps],
        steps,
        f"{weight_name} | Pre-round FP",
        os.path.join(out_dir, "pre_round_fp.png"),
    )

    # ======================
    # round 后 index
    # ======================
    plot_int_hist_save(
        [w_int[s] for s in steps],
        steps,
        qmin,
        qmax,
        f"{weight_name} | Quant Index",
        os.path.join(out_dir, "quant_index.png"),
    )

    # ======================
    #  dequant
    # ======================
    plot_hist_save(
        [w_dequant[s] for s in steps],
        steps,
        f"{weight_name} | Dequant FP",
        os.path.join(out_dir, "dequant_fp.png"),
    )

    # ======================
    # ΔW
    # ======================
    # delta_tensors, delta_labels = [], []
    # for i in range(1, len(steps)):
    #     s0, s1 = steps[i - 1], steps[i]
    #     # delta = w_fp[s1] - w_fp[s0]
    #     # print("mean |ΔW|      :", delta.abs().mean().item())
    #     # print("nonzero ratio  :", (delta.abs() > 0).float().mean().item())
    #     # delta_tensors.append(delta)
    #     # delta_labels.append(f"Δ({s1}-{s0})")

    #     delta = w_int[s1] - w_int[s0]
    #     print("mean |ΔW|      :", delta.abs().mean().item())
    #     delta_tensors.append(delta)
    #     delta_labels.append(f"Δ({s1}-{s0})")

    # plot_hist_save(
    #     delta_tensors,
    #     delta_labels,
    #     f"{weight_name} | FP Delta",
    #     os.path.join(out_dir, "delta_fp.png"),
    # )

    delta_residuals = []
    delta_labels = []

    for i in range(1, len(steps)):
        s0, s1 = steps[i - 1], steps[i]
        dscale = scale_temp[s1] - scale_temp[s0]
        dz = zint_temp[s1] - zint_temp[s0]
        dint = w_int[s1] - w_int[s0]
        dfp = w_fp[s1] - w_fp[s0]

        delta_residuals.append(dfp)
        delta_labels.append(f"Δw({s1}-{s0})")

        print(f"[{s0}->{s1}]")
        print("  mean |Δscale| :", dscale.abs().mean().item())
        print("  max  |Δscale| :", dscale.abs().max().item())

        print("  mean |Δz_int| :", dz.abs().mean().item())
        print("  max  |Δz_int| :", dz.abs().max().item())
        
        print("  mean |Δw_int| :", dint.abs().mean().item())
        print("  max  |Δw_int| :", dint.abs().max().item())

        print("  mean |Δw_fp| :", dfp.abs().mean().item())
        print("  max  |Δw_fp| :", dfp.abs().max().item())


    plot_hist_save(
        delta_residuals,
        delta_labels,
        f"{weight_name} | Δ Residual",
        os.path.join(out_dir, "delta_residual.png"),
    )



# =========================
# 使用示例
# =========================

if __name__ == "__main__":
    ckpt_dirs = {
        "step0": "/root/lai-code/quant_models/qwen3-4b-instruct/fake_quant_model",
        "step100": "/share/MY-DAPO/Qwen3-4B-AWQ-w4g128-Soft-Only/global_step_100_hf",
        "step200": "/share/MY-DAPO/Qwen3-4B-AWQ-w4g128-Soft-Only/global_step_200_hf",
        # "step6": "/root/lai-code/verl/ckpts/DAPO-TEST/Qwen3-4B-AWQ-w4g128-Soft-Only/global_step_6_hf",
    }

    weight_name = "model.layers.10.mlp.up_proj.weight"

    qmin, qmax = 0, 15

    analyze_layer_across_steps_hf(
        ckpt_dirs=ckpt_dirs,
        weight_name=weight_name,
        qmin=qmin,
        qmax=qmax,
    )

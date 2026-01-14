import torch
import torch.nn as nn
from collections import Counter
from ..quantization.utils import replace_modules, inject_to_modules
from ..quantization.fake_linear import StepAwareFakeLinear


class MockQuantState:
    def __init__(
        self,
        *,
        soft_round_enable=True,
        progressive_enable=False,
        progressive_ratio=0.0
    ):
        self.soft_round_enable = soft_round_enable
        self.progressive_enable = progressive_enable
        self.progressive_ratio = progressive_ratio


def test_inject_quant_state(model):

    qs = MockQuantState(
        soft_round_enable=True,
        progressive_enable=True,
        progressive_ratio=0.5
    )

    inject_to_modules(
        model,
        module_type=StepAwareFakeLinear,
        attr_name="_quant_state",
        value=qs,
        verbose=False,
    )

    # ---- assert 注入成功 ----
    for m in model.modules():
        if isinstance(m, StepAwareFakeLinear):
            assert hasattr(m, "_quant_state")
            assert m._quant_state is qs

    print("[OK] QuantState injected correctly")


@torch.no_grad()
def test_forward(model, device="cuda"):
    model.eval()
    model.to(device)

    # 随机 token
    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(2, 16),
        device=device,
    )

    outputs = model(input_ids)
    logits = outputs.logits

    assert torch.isfinite(logits).all()
    print("[OK] Forward pass successful:", logits.shape)


def test_local_grad_flow(model, device="cuda"):
    model.train().to(device)

    # 只取一个 FakeLinear
    fake_linear = None
    for m in model.modules():
        if isinstance(m, StepAwareFakeLinear):
            fake_linear = m
            break
    assert fake_linear is not None

    # 构造一个最小输入
    x = torch.randn(2, fake_linear.in_features, device=device, requires_grad=True)

    # 直接 forward 这个模块
    y = fake_linear(x)

    # 人工构造 loss
    loss = y.pow(2).mean()
    loss.backward()

    # ---- 关键检查 ----
    grad = fake_linear.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0

    print("[OK] Local grad flow works")

def test_model_quant_levels(model: nn.Module, soft_round_enable=True, progressive_enable=True, progressive_ratio=1.0, verbose=True):
    """
    对整个模型的 StepAwareFakeLinear 模块进行 group-level 量化统计
    输出每个 level 数的 group 占比
    
    Args:
        model: nn.Module, 已替换 StepAwareFakeLinear
        soft_round_enable: 是否使用 soft round
        progressive_enable: 是否使用渐进量化
        progressive_ratio: 渐进比例
        verbose: 是否打印每个模块统计
    """
    total_counter = Counter()
    total_groups = 0

    for name, module in model.named_modules():
        if isinstance(module, StepAwareFakeLinear):
            w = module.weight.detach().clone()
            fq = module.fake_quant_weight
            dim1, dim2 = w.shape
            group_size = fq.group_size if fq.group_size != -1 else dim2
            assert dim2 % group_size == 0
            w_reshaped = w.reshape(-1, group_size)

            group_level_counts = []
            for g in w_reshaped:
                g = g.unsqueeze(0)
                g_q = fq.fake_quant(
                    g,
                    soft_round_enable=soft_round_enable,
                    progressive_enable=progressive_enable,
                    progressive_ratio=progressive_ratio
                )
                levels = torch.unique(g_q).numel()
                group_level_counts.append(levels)

            counter = Counter(group_level_counts)
            total_counter.update(counter)
            total_groups += len(group_level_counts)

            if verbose:
                print(f"[Module: {name}] group-level counts: {dict(counter)}")

    print("\n[Global Group-Level Distribution]")
    for levels, count in sorted(total_counter.items()):
        print(f"  Groups with {levels} levels: {count} / {total_groups} ({count/total_groups*100:.2f}%)")

    return total_counter, total_groups


def run_all_tests(model):
    print("=== Test: inject ===")
    test_inject_quant_state(model)

    print("=== Test: forward ===")
    test_forward(model)

    print("=== Test: backward ===")
    test_local_grad_flow(model)



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("/share/Llama-3.2-1B")
    dst_kwargs = {"s_bits": 4, "t_bits": 2, "group_size": 128}
    replace_modules(
        model,
        src_type=nn.Linear,
        dst_type=StepAwareFakeLinear,
        dst_kwargs=dst_kwargs,
        skip_names=("lm_head",),
        verbose=False,
    )

    # run_all_tests(model)
    test_model_quant_levels(model)
import torch
import torch.nn as nn
from collections.abc import Callable


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def replace_modules(
    model: nn.Module,
    *,
    src_type: type,
    dst_type: type,
    dst_kwargs: dict | None = None,
    skip_names=(),
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
    verbose: bool = True,
):
    if dst_kwargs is None:
        dst_kwargs = {}

    named_modules = dict(model.named_modules())

    if verbose:
        print(
            f"[replace_modules] Found {sum(isinstance(m, src_type) for m in named_modules.values())} "
            f"{src_type.__name__} modules"
        )

    for name, module in named_modules.items():
        if not isinstance(module, src_type):
            continue

        if any(s in name for s in skip_names):
            continue

        if filter_fn is not None and not filter_fn(name, module):
            continue

        new_module = dst_type(
            org_module=module,
            **dst_kwargs,
        )

        set_op_by_name(model, name, new_module)

        if verbose:
            print(f"[replace_modules] Replaced: {name}")

    return model


def inject_to_modules(
    model: nn.Module,
    *,
    module_type: type | tuple[type, ...] | None = None,
    module_name_contains: tuple[str, ...] = (),
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
    attr_name: str,
    value,
    use_setter: bool = True,
    verbose: bool = False,
):
    for name, module in model.named_modules():

        if module_type is not None and not isinstance(module, module_type):
            continue

        if module_name_contains and not any(s in name for s in module_name_contains):
            continue

        if filter_fn is not None and not filter_fn(name, module):
            continue

        # 支持 lazy value
        v = value(module) if callable(value) else value

        setter = None
        if use_setter:
            setter_name = f"set_{attr_name.lstrip('_')}"
            setter = getattr(module, setter_name, None)

        if setter is not None and callable(setter):
            setter(v)
        else:
            setattr(module, attr_name, v)

        if verbose:
            print(f"[Inject] {name}.{attr_name} <- {type(v).__name__}")



def pseudo_quantize_tensor(
    w, n_bit=4, zero_point=True, q_group_size=128, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        w.div_(scales)
        w.round_()
        w.add_(zeros).clamp_(min_int, max_int).sub_(zeros)
        w.mul_(scales)
    else:
        w = torch.clamp(
            torch.round(w / scales) + zeros,
            min_int,
            max_int
        )
        w = (w - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

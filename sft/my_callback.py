import torch
from transformers import TrainerCallback


class QuantizationCallback(TrainerCallback):
    def __init__(self, quant_controller):
        """
        quant_controller: QuantizationController 实例
        """
        self.controller = quant_controller

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """
        在每个 optimizer update 开始前调用。
        global_step = 当前 update index (从0开始)
        """

        if model is None:
            print("model is null!")
            return

        if not self.controller.enable:
            return

        step = state.global_step  # 第 k 次 update
        quant_state = self.controller.get_state(step)

        for module in model.modules():
            if hasattr(module, "set_quant_state"):
                module.set_quant_state(quant_state)
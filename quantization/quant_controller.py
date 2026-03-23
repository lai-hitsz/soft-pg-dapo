from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class QuantState:
    # soft-round 参数
    soft_round_enable: bool
    
    # progressive 参数
    progressive_enable: bool
    progressive_ratio: Optional[float] = None  # ∈ [0, 1]


class QuantizationController:
    """
    Step-based quantization scheduler.
    Progressive capacity shrink is continuous and delegated to QLinear.
    """

    def __init__(self, quant_cfg):
        # ===== global =====
        self.enable = bool(quant_cfg.get("enable", False))

        # ===== nominal bits =====
        bits_cfg = quant_cfg.get("bits", {})
        self.begin_pg = int(bits_cfg.get("begin_pg", 0))
        self.start = int(bits_cfg.get("start", 4))
        self.target = int(bits_cfg.get("target", 2))
        self.pg_duration = int(bits_cfg.get("pg_duration", 200))  # duration for ratio 0->1
        self.enable_progressive = bool(bits_cfg.get("enable_progressive", True))

        # ===== soft-round =====
        soft_cfg = quant_cfg.get("soft_round", {})
        self.soft_round_enable = bool(soft_cfg.get("enable", True))

        if self.enable_progressive:
            assert self.begin_pg >= 0, "begin_pg must be >= 0 when progressive is enabled"


    # ------------------------------------------------------------------

    @property
    def _is_active(self) -> bool:
        return self.enable

    def _compute_progressive_ratio(self, global_step: int) -> float:
        """
        Continuous shrink ratio inside current stage: ∈ [0, 1].
        Used to shrink level_max / delta / clip range in QLinear.
        """
        if not self.enable_progressive or global_step < self.begin_pg:
            return 0.0

        offset = global_step - self.begin_pg
        if self.pg_duration <= 0:
            return 1.0  # immediately saturated
        return min(max(offset / self.pg_duration, 0.0), 1.0)
    

    def ratio_from_capacity_level(
        self,
        progressive_ratio: float
    ) -> float:
        # lazy init
        if not hasattr(self, "_last_level"):
            self._last_level = 0

        if not hasattr(self, "_level_ratio"):
            self._level_ratio = 0.0

        L_max = (2 ** self.start) - 1
        L_eff = (2 ** self.target) - 1

        # compute current level
        level = math.ceil(L_max - progressive_ratio * (L_max - L_eff))

        # first call or level changed
        if self._last_level is None or level != self._last_level:
            self._last_level = level
            self._level_ratio = progressive_ratio

        return self._level_ratio
    

    def ratio_from_num_gen_batch(
        self,
        num_gen_epochs: int,
        update_threshold: int = 10,
        delta: float = 0.01,
        max_ratio: float = 1.0,
    ) -> float:
        # lazy init
        if not hasattr(self, "_batch_driven_ratio"):
            self._batch_driven_ratio = 0.0

        if num_gen_epochs <= update_threshold and num_gen_epochs != 0:
            self._batch_driven_ratio = min(
                self._batch_driven_ratio + delta,
                max_ratio,
            )

        return self._batch_driven_ratio

    # ------------------------------------------------------------------

    def get_state(self, global_step: int) -> QuantState:
        prog_ratio = self._compute_progressive_ratio(global_step)
        soft_active = self.soft_round_enable
        prog_active = self.enable_progressive and global_step >= self.begin_pg


        return QuantState(
            soft_round_enable=soft_active,
            progressive_enable=prog_active,
            progressive_ratio=prog_ratio
        )
    
    def get_level_state(self, global_step: int) -> QuantState:
        prog_ratio = self._compute_progressive_ratio(global_step)
        prog_ratio = self.ratio_from_capacity_level(prog_ratio)
        soft_active = self.soft_round_enable
        prog_active = self.enable_progressive and global_step >= self.begin_pg


        return QuantState(
            soft_round_enable=soft_active,
            progressive_enable=prog_active,
            progressive_ratio=prog_ratio
        )

    def get_batch_state(self, num_gen_epochs: int) -> QuantState:
        prog_ratio = self.ratio_from_num_gen_batch(num_gen_epochs)
        soft_active = self.soft_round_enable
        prog_active = self.enable_progressive


        return QuantState(
            soft_round_enable=soft_active,
            progressive_enable=prog_active,
            progressive_ratio=prog_ratio
        )

from dataclasses import dataclass
from typing import Optional


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
        self.pg_duration = int(bits_cfg.get("pg_duration", 200))  # duration for ratio 0->1
        self.enable_progressive = bool(bits_cfg.get("enable_progressive", False))

        # ===== soft-round =====
        soft_cfg = quant_cfg.get("soft_round", {})
        self.soft_round_enable = bool(soft_cfg.get("enable", False))

        if self.enable_progressive:
            assert self.begin_pg >= 0, "begin_pg must be >= 0 when progressive is enabled"

    # ------------------------------------------------------------------

    def _is_active(self, global_step: int) -> bool:
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

    # ------------------------------------------------------------------

    def get_state(self, global_step: int) -> QuantState:
        prog_ratio = self._compute_progressive_ratio(global_step)
        soft_active = self.soft_round_enable
        prog_active = self.enable_progressive and global_step >= self.begin_pg

        # ---- 00: progressive 0, soft 0 ----
        if not prog_active and not soft_active:
            return QuantState(
                soft_round_enable=False,
                progressive_enable=False,
                progressive_ratio=prog_ratio
            )

        # ---- 10: progressive 1, soft 0 ----
        elif prog_active and not soft_active:
            return QuantState(
                soft_round_enable=False,
                progressive_enable=True,
                progressive_ratio=prog_ratio,
            )

        # ---- 01: progressive 0, soft 1 ----
        elif not prog_active and soft_active:
            # soft_step = self._compute_soft_round_step(global_step)
            return QuantState(
                soft_round_enable=True,
                progressive_enable=False,
                progressive_ratio=prog_ratio
            )

        # ---- 11: progressive 1, soft 1 ----
        elif prog_active and soft_active:
            # soft_step = self._compute_soft_round_step(global_step)
            return QuantState(
                soft_round_enable=True,
                progressive_enable=True,
                progressive_ratio=prog_ratio,
            )

        else:
            raise RuntimeError("Invalid quantization state combination")

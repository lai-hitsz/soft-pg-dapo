import inspect
import logging
import os
import time
from collections import OrderedDict

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from dataclasses import asdict

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, layered_summon_lora_params, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_device_is_available
from verl.utils.vllm_utils import TensorLoRARequest, VLLMHijack, is_version_ge, patch_vllm_moe_model_weight_loader

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


from verl.workers.sharding_manager.base import BaseShardingManager


class FSDPHFShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(self, module: FSDP, model_config, full_params: bool = False,
                 device_mesh: DeviceMesh = None, offload_param: bool = False,
                 load_format: str = "hf", layered_summon: bool = True):
        self.module = module
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.load_format = load_format
        self.layered_summon = layered_summon

        # Full params vs sharded params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig()
            )

        # Tensor parallel info
        if self.device_mesh is not None:
            self.tp_size = self.device_mesh["infer_tp"].size()
            self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()
        else:
            self.tp_size = 1
            self.tp_rank = 0

        # 保存原始随机状态
        self.torch_random_states = get_torch_device().get_rng_state()
        if self.device_mesh is not None:
            # Ray 多卡环境下，保证不同 dp rank 有相同初始化
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        self.base_sync_done: bool = "dummy" not in load_format

    def _params_on_cpu(self) -> bool:
        """
        判断 FSDP flat_param 是否还在 CPU
        """
        for handle in self.module._all_handles:
            if handle._offload_params:
                continue
            flat = handle.flat_param
            if flat.device.type == "cpu":
                return True
        return False


    @GPUMemoryLogger(role="fsdp hf sharding_manager", logger=logger)
    def __enter__(self):
        """
        HF rollout 的 sharding manager
        """

        # rollout 前清理显存碎片（可选，但安全）
        get_torch_device().empty_cache()

        if self.offload_param and self._params_on_cpu():
            load_fsdp_model_to_gpu(self.module)

        # 多卡 RNG 对齐（如果 rollout 用 sampling）
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

        return self


    @GPUMemoryLogger(role="fsdp hf sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        """
        HF + FSDP 上下文退出
        - offload_param 时将 flat_param 回 CPU
        - 保持 train 模式
        - 恢复多卡随机状态
        """
        self.module.train()

        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp hf sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        HF 版本多卡输入预处理：
        - 在 tensor parallel group 内做 all_gather，保证每个 rank 的输入一致
        """
        if self.tp_size == 1:
            return data

        group = self.device_mesh["infer_tp"].group
        all_gather_data_proto(data=data, process_group=group)
        return data


    @GPUMemoryLogger(role="fsdp hf sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        HF 版本多卡输出后处理：
        - 从 all_gather 的数据中取本 rank 对应的 chunk
        """
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]


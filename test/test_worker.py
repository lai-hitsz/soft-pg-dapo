import torch
from omegaconf import OmegaConf
from ..worker.my_workers import QuantActorRolloutRefWorker
from verl.utils.fs import copy_to_local
import os

# 单卡 FSDP 环境
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

# -----------------------------
# Minimal config
# -----------------------------
config_dict = {
    "model": {
        "path": "/share/Llama-3.2-1B",  # 替换成你本地可用的模型路径
        "override_config": {},
        "use_remove_padding": False,
        "use_shm": False,
        "use_fused_kernels": False,
        "enable_gradient_checkpointing": False,
        "trust_remote_code": False,
        "use_liger": False,
        "enable_activation_offload": False,
    },
    "actor": {
        "strategy": "fsdp",
        "ppo_mini_batch_size": 128,
        "ppo_micro_batch_size": None,
        "ppo_micro_batch_size_per_gpu": None,
        "use_dynamic_bsz": False,
        "ppo_max_token_len_per_gpu": 16384,
        "grad_clip": 1.0,
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0,
        "loss_agg_mode": "token-mean",
        "entropy_coeff": 0.0,
        "use_kl_loss": False,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "ppo_epochs": 1,
        "shuffle": False,
        "ulysses_sequence_parallel_size": 1,
        "checkpoint": {"contents": ["model", "optimizer", "extra"]},
        "optim": {
            "lr": 1e-6,
            "lr_warmup_steps": -1,
            "lr_warmup_steps_ratio": 0.0,
            "min_lr_ratio": 0.0,
            "num_cycles": 0.5,
            "warmup_style": "constant",
            "total_training_steps": -1,
            "weight_decay": 0.01,
        },
        "fsdp_config": {
            "wrap_policy": {"min_num_params": 0},
            "param_offload": False,
            "optimizer_offload": False,
            "offload_policy": False,
            "reshard_after_forward": True,
            "fsdp_size": -1,
        },
    },
    "rollout": {
        "n": 1,
        "name": "hf",
        "mode": "sync",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": True,
        "max_prompt_length": 512,
        "max_response_length": 512,
    },
    "quant": {
        "enable": True,
        "group_size": 128,
        "bits": {"start": 4, "enable_progressive": False, "target": 2},
        "soft_round": {"enable": True},
        "modules": {"linear": True, "lm_head": False},
    },
}

config = OmegaConf.create(config_dict)


# -----------------------------
# 初始化 worker
# -----------------------------
worker = QuantActorRolloutRefWorker(config=config, role="actor")

# -----------------------------
# 构建模型
# -----------------------------
local_path = copy_to_local(config.model.path, use_shm=config.model.get("use_shm", False))
actor_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config = worker._build_model_optimizer(
    model_path=local_path,
    quant_config=config.quant,
    fsdp_config=config.actor.fsdp_config,
    optim_config=config.actor.optim,
    override_model_config=config.model.override_config,
    use_remove_padding=config.model.get("use_remove_padding", False),
    use_fused_kernels=config.model.get("use_fused_kernels", False),
    enable_gradient_checkpointing=config.model.get("enable_gradient_checkpointing", False),
    trust_remote_code=config.model.get("trust_remote_code", False),
    use_liger=config.model.get("use_liger", False),
    role="actor",
    enable_activation_offload=config.model.get("enable_activation_offload", False),
)
print("=== FSDP model built successfully ===")
print(actor_fsdp)

# -----------------------------
# 简单前向测试
# -----------------------------
print("=== Running forward pass test ===")
tokenizer = worker.tokenizer
text = "Hello world!"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
actor_fsdp.eval()
with torch.no_grad():
    outputs = actor_fsdp(**inputs)
print("Forward pass successful, logits shape:", outputs.logits.shape)

torch.distributed.destroy_process_group()
print("=== Test complete ===")
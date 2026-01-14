"""
FSDP-only main for DAPO-style recipe.
"""


import hydra
import ray

from verl.trainer.ppo.reward import get_custom_reward_fn
from .my_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_quant", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # print resolved config
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # download checkpoint
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # tokenizer / processor
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path)
        processor = hf_processor(local_path, use_fast=True)

        # ============================
        # FSDP ONLY
        # ============================
        assert config.actor_rollout_ref.actor.strategy == "fsdp"
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

        from verl.single_controller.ray import RayWorkerGroup
        from .worker.my_workers import QuantActorRolloutRefWorker
        from verl.workers.fsdp_workers import CriticWorker

        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(QuantActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node]
            * config.trainer.nnodes
        }

        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # reward model
        if config.reward_model.enable:
            from verl.workers.fsdp_workers import RewardModelWorker

            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference policy (for KL / DAPO)
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(QuantActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # reward manager
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)

        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

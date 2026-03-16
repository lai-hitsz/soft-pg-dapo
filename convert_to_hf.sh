# 转换为hf模型
python /root/lai-code/verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /share/MY-DAPO/AWQ-w4g128-Sofe-Only/global_step_100/actor \
    --target_dir /share/MY-DAPO/AWQ-w4g128-Sofe-Only/global_step_100_hf

# 转换为伪量化模型并保存
CUDA_VISIBLE_DEVICES=0 python -m soft_gq_dapo.convert_to_quant \      
--model_path /share/MY-DAPO/QWEN3-4B-DAPO/global_step_600_hf \                                          
--wbits 4 \ 
--tbits 3 \
--use_pg \
--save_path /share/MY-DAPO/QWEN3-4B-DAPO/global_step_600_hf_quant


# 评测
HF_ALLOW_CODE_EVAL=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval \
--model hf \
--model_args pretrained="/share/Qwen3-4B-Instruct-2507",parallelize=True \
--tasks gsm8k \
--batch_size 64 \
--confirm_run_unsafe_code


# 转换并评测
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m soft_gq_dapo.convert_and_eval \
--model_path /share/Qwen3-4B-Instruct-2507 \
--eval_tasks gsm8k \
--eval_batch_size 64 \
--wbits 16 \
--shots 5
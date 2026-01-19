python /root/lai-code/verl/scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /share/MY-DAPO/AWQ-w4g128-Sofe-Only/global_step_100/actor \
    --target_dir /share/MY-DAPO/AWQ-w4g128-Sofe-Only/global_step_100_hf
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model gpt \
    --model_args model_version=gpt-4o-mini \
    --tasks tcm_sd_rc_five_upperbound \
    --batch_size auto \
    --log_samples \
    --output_path ./logs/ \
    --wandb_args project=tcm_sd_rc_five,name=upperbound \
    # --limit 5
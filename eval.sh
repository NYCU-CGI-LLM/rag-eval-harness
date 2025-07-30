export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model gpt \
    --model_args model_version=gpt-4o-mini,api_url=http://localhost:8004/v1/chat/completions \
    --tasks tcm_sd_multiple_choice \
    --batch_size auto \
    --log_samples \
    --output_path ./logs/ \
    --wandb_args project=test_tcm_sd,name=simple_rag_new \
    --limit 5
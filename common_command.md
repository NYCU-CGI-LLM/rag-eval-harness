

accelerate launch --num_processes 1 --main_process_port 30000 -m lmms_eval \
    --model gpt \
    --model_args model_version=<your_model>,api_url=http://localhost:8000/v1/chat/completions \
    --tasks <your_task> \
    --batch_size auto \
    --log_samples \
    --output_path ./logs/ \
    --wandb_args project=<your_project_name>,name=<your_method>


task: 
- medqa
- tcmeval_sdt_task1
- tcmeval_sdt_task2
- tcmeval_sdt_task3
- tcmeval_sdt_task4
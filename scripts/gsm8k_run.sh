
TOKENIZERS_PARALLELISM=false WANDB_MODE=offline accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port=1234 gsm8k_train.py --learning_rate 2e-5 --epochs 5 --lora 64 --per_device_batch_size 2

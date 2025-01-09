TOKENIZERS_PARALLELISM=false accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port=1234 gsm8k_train.py --learning_rate 2e-5 --epochs 5 --per_device_batch_size 2 --model mistralai/Mistral-7B-v0.1


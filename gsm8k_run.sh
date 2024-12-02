
evaluate_checkpoints_gsm8k() {
    local RUN_NAME=$1

    # counter=0
    # for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    # do
    #     CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
    #     echo $CKPT
    #     CUDA_VISIBLE_DEVICES=$counter python gsm8k_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
    #     ((counter++))
    # done 
    # wait
    counter=0
    for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    do
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        echo $CKPT
        CUDA_VISIBLE_DEVICES=$counter python gsm8k_eval_samples.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
        ((counter++))
    done 
    wait
    counter=0
    for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    do
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        echo $CKPT
        CUDA_VISIBLE_DEVICES=$counter python gsm8k_eval_samples.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
        ((counter++))
    done 
    wait
}

#CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=1234 gsm8k_train.py --learning_rate 2e-5 --epochs 3 --lora 64
TOKENIZERS_PARALLELISM=false WANDB_MODE=offline accelerate launch --multi_gpu --mixed_precision=fp16 --main_process_port=1234 gsm8k_train.py --learning_rate 2e-5 --epochs 5 --lora 64 --per_device_batch_size 2
# --model meta-llama/Llama-2-7b-hf
#RUN_NAME=gsm8k_orig_3epochs_full_lr2e-05_bs128_llama-3-8b
#evaluate_checkpoints_gsm8k $RUN_NAME #only use this if num epochs is less than num GPUs

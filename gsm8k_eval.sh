evaluate_checkpoints_gsm8k() {
    local RUN_NAME=$1
    local gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

    counter=0
    for CKPT_DIR in /data/locus/large_training_datasets/kbaek/ckpts/$RUN_NAME/checkpoint-*/
    do  
        utilization=$(nvidia-smi -i $counter --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [[ "$utilization" -gt 5 ]]; then wait
        fi
        echo $CKPT_DIR
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        CUDA_VISIBLE_DEVICES=$counter python gsm8k_eval_samples.py --ckpt_dir ${CKPT_DIR::-1} --eval_type test --num_samples 5 &
        counter=$(((counter+1)%$gpu_count))
    done 
}

evaluate_checkpoints_gsm8k $1 #only use this if num epochs is less than num GPUs
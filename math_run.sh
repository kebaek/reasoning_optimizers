evaluate_checkpoints_math() {
    local RUN_NAME=$1

    counter=0
    for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    do
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        echo $CKPT
        CUDA_VISIBLE_DEVICES=$counter python math_eval_perplexity.py --ckpt_dir $RUN_NAME --ckpt $CKPT &
        ((counter++))
    done 
    wait
    counter=0
    for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    do
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        echo $CKPT
        CUDA_VISIBLE_DEVICES=$counter python math_eval_samples.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type test --num_samples 5 &
        ((counter++))
    done 
    wait
    counter=0
    for CKPT_DIR in ckpts/$RUN_NAME/checkpoint-*/
    do
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        echo $CKPT
        CUDA_VISIBLE_DEVICES=$counter python math_eval_samples.py --ckpt_dir ckpts/$RUN_NAME/checkpoint-$CKPT --eval_type train --num_samples 5 &
        ((counter++))
    done 
    wait
}


torchrun --nproc_per_node=4 --master_port=1234 math_train.py --train_type full --learning_rate 2e-5 --epochs 3
RUN_NAME=math_orig_3epochs_full_lr2e-05_bs128
evaluate_checkpoints_math $RUN_NAME #only run this if num epochs is less than num GPUs
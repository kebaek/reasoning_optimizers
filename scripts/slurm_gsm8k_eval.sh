evaluate_checkpoints_gsm8k() {
    local RUN_NAME=$1
    local TEMP=$2
    #local CKPT_DIR_BASE="/data/locus/large_training_datasets/kbaek/ckpts/$RUN_NAME"
    local CKPT_DIR_BASE="/data/locus/llm_weights/kbaek/ckpts/$RUN_NAME"

    # Loop through all checkpoint directories
    for CKPT_DIR in ${CKPT_DIR_BASE}/checkpoint-*/
    do  
        # # Check if the directory contains a file starting with "test_answers"
        # if ls "${CKPT_DIR}"test_answers* 1> /dev/null 2>&1; then
        #     echo "Checkpoint ${CKPT_DIR} already contains test_answers files. Skipping."
        #     continue
        # fi
        
        # Create a unique job name for this checkpoint
        CKPT=$(basename $CKPT_DIR | sed 's/checkpoint-//')
        JOB_NAME="gsm8k_eval_${RUN_NAME}_${CKPT}"
        echo $CKPT 
        # Submit the job using sbatch
        sbatch --mem=70GB --gres=gpu:A6000:1 --partition=preempt --job-name=$JOB_NAME --requeue --time=1-00:00:00 --output=logs/${JOB_NAME}.out --error=logs/${JOB_NAME}.err --wrap="
        source ~/.bashrc
        conda activate sphere
        cd ~/reasoning_optimizers
        CUDA_VISIBLE_DEVICES=0 python gsm8k_eval_samples.py --ckpt_dir ${CKPT_DIR::-1} --eval_type test --num_samples 4 --temp $TEMP
        "
    done
}

# Ensure logs directory exists.
mkdir -p logs

# Call the function with the argument
evaluate_checkpoints_gsm8k $1 $2

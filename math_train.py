import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import os
from datasets import load_dataset
import numpy as np
import argparse
import json
from huggingface_params import cache_dir, use_auth_token
from utils import *
from tokenizers.processors import TemplateProcessing
import torch.distributed as dist


def make_supervised_data_module(train_type, output_dir, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset = load_dataset("hendrycks/competition_math", cache_dir=cache_dir, trust_remote_code=True)
    
    train_questions = np.array(dataset["train"]["problem"])
    train_answers = np.array(dataset["train"]['solution'])
    
    if train_type == "full":
        subsample_idxs = np.arange(len(train_questions))
    else:
        if dist.get_rank() == 0:
            if train_type == "half":
                subsample_idxs = np.arange(len(train_questions))
                subsample_idxs = np.random.choice(subsample_idxs, (len(train_questions)+1)//2, replace=False)
            elif train_type == "quarter":
                subsample_idxs = np.arange(len(train_questions))
                subsample_idxs = np.random.choice(subsample_idxs, (len(train_questions)+1)//4, replace=False)
            np.random.shuffle(subsample_idxs)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            np.save(os.path.join(output_dir, f"subsample_idxs.npy"), subsample_idxs)
            dist.barrier()
        else:
            dist.barrier()
            subsample_idxs = np.load(os.path.join(output_dir, f"subsample_idxs.npy"))
    print(subsample_idxs)
                                
    train_questions  = train_questions[subsample_idxs]
    train_answers = train_answers[subsample_idxs]

    train_dataset = SupervisedDataset(train_questions, train_answers, tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    model_name_or_path="meta-llama/Meta-Llama-3-8B"

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_type", type=str, default="full")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dont_save_intermediate", action='store_true')

    args = parser.parse_args()
    train_type = args.train_type
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_intermediate = not args.dont_save_intermediate


    project_name = "math_orig"
    run_name  = f"{num_epochs}epochs_{train_type}_lr{learning_rate}_bs{batch_size}"


    num_devices = torch.cuda.device_count()


    per_device_batch_size = 1
    gradient_accumulation_steps = int((batch_size/1)/num_devices)
    print("Num devices: ", num_devices)
    print("Per device batch: ", per_device_batch_size)
    print("Grad accum steps: ", gradient_accumulation_steps)

    assert(gradient_accumulation_steps*per_device_batch_size*num_devices==batch_size)

    if save_intermediate:
        save_strategy = "epoch"
        save_steps = None 
    else:
        save_strategy = "no"
        save_steps = None

    output_dir = f"ckpts/{project_name}_{run_name}"
    training_args = TrainingArguments(
        num_train_epochs = num_epochs, 
        per_device_train_batch_size = per_device_batch_size,
        per_device_eval_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr_scheduler_type = "linear",
        warmup_steps = 20,
        learning_rate = learning_rate,
        max_grad_norm = 2,
        optim = "adamw_torch",
        output_dir = output_dir,
        evaluation_strategy = "no",
        logging_strategy = "no",
        report_to = "none",
        save_strategy = save_strategy,
        save_steps = save_steps,
        save_only_model = True,
        run_name=run_name,
        bf16 = True,
        fsdp= "full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap= 'LlamaDecoderLayer',
    )
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        cache_dir=cache_dir
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        model_max_length=1024,
        padding_side="right",
        cache_dir=cache_dir)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(train_type, output_dir, tokenizer=tokenizer)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if not save_intermediate:
        trainer.save_state()
        trainer.save_model(output_dir=f"ckpts/{project_name}_{run_name}")


if __name__ == "__main__":
    train()

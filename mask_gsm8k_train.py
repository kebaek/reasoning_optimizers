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
from peft import LoraConfig, get_peft_model
from huggingface_params import cache_dir, use_auth_token
from utils import *
import datasets
import torch.distributed as dist
from accelerate import Accelerator

accelerator = Accelerator()

os.environ["WANDB_PROJECT"] = "reasoning_optimizer"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

def average_metric_across_devices(metric_value):
    # Ensure that the metric_value is a tensor
    tensor_value = torch.tensor(metric_value).cuda()
    # Reduce across all processes (sum the metric values)
    dist.reduce(tensor_value, dst=0, op=dist.ReduceOp.SUM)
    # Divide by the world size to get the average
    if dist.get_rank() == 0:
        tensor_value /= dist.get_world_size()
    return tensor_value.item()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs[1]
            
            
            

            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
                    
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
            
            loss_per_example = loss.view(labels.size(0), -1).sum(dim=1) / (shift_labels != -100).sum(dim=1)
            
            mask = (loss.view(labels.size(0), -1) > 0).to(torch.float)
            
            # if dist.get_rank() == 0:
            #     import IPython; IPython.embed()
            # else:
            #     dist.barrier()
            
            
            total_epochs = self.args.num_train_epochs
            current_epoch = self.state.epoch
            
            if current_epoch < total_epochs-1:
                ratio_masked = 0
                for example_idx in range(len(mask)):
                    if (torch.log(loss_per_example[example_idx])) < -2:
                        mask[example_idx]*=0.01
                        # print("Masked")
                        # print(torch.log(loss_per_example))
                        ratio_masked+=1
                ratio_masked=ratio_masked/len(mask)
                avg_ratio_masked = average_metric_across_devices(ratio_masked)
                    # else:
                    #     print("Not masked")
            else:
                ratio_masked = 0
                avg_ratio_masked = 0


            loss = (loss*mask.view(-1)).sum()
            
            avg_loss = average_metric_across_devices(loss.item())
            if dist.get_rank() == 0:
                self.log({'ratio_masked': avg_ratio_masked,
                          "loss": avg_loss})
                
            return (loss, outputs) if return_outputs else loss

def make_supervised_data_module(output_dir, train_type, tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    train_questions_orig = np.array(dataset["train"]["question"])
    train_answers_orig = np.array(dataset["train"]['answer'])

    if train_type == "full":
        subsample_idxs = np.arange(len(train_questions_orig))
    elif train_type == "half":
        subsample_idxs= np.arange(len(train_questions_orig))
        if dist.get_rank() == 0:
            subsample_idxs = np.random.choice(subsample_idxs, len(subsample_idxs)//2, replace=False)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)
            dist.barrier()
        else:
            dist.barrier()
            subsample_idxs = np.load(os.path.join(output_dir, f"subsample_idxs.npy"))
        
    elif train_type == "quarter":
        subsample_idxs= np.arange(len(train_questions_orig))
        if dist.get_rank() == 0:
            subsample_idxs = np.random.choice(subsample_idxs, len(subsample_idxs)//4, replace=False)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(os.path.join(output_dir, "subsample_idxs.npy"), subsample_idxs)
            dist.barrier()
        else:
            dist.barrier()
            subsample_idxs = np.load(os.path.join(output_dir, f"subsample_idxs.npy"))
            
    else:
        1/0
    
    train_dataset = SupervisedDataset(train_questions_orig[subsample_idxs], train_answers_orig[subsample_idxs], tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_type", type=str, default="full")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--lora", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0)
    parser.add_argument("--dont_save_intermediate", action='store_true')

    args = parser.parse_args()
    train_type = args.train_type
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    use_lora = args.lora
    model_name_or_path = args.model
    save_intermediate = not args.dont_save_intermediate
    
    
    project_name = "gsm8k_orig"
    model_name = args.model.split('/')[-1].lower()
    if use_lora != 0:
        run_name  = f"{epochs}epochs_{train_type}_lr{learning_rate}_rho0_bs{batch_size}_lora{use_lora}_{model_name}"
    else:
        run_name  = f"{epochs}epochs_{train_type}_lr{learning_rate}_rho0_bs{batch_size}_{model_name}"
    
    
    batch_size = batch_size
    num_devices = torch.cuda.device_count()
    per_device_batch_size = args.per_device_batch_size
    gradient_accumulation_steps = int((batch_size/per_device_batch_size)/num_devices)
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
    
    output_dir = f"/data/locus/large_training_datasets/kbaek/ckpts/{project_name}_{run_name}"
    
    
    training_args = TrainingArguments(
        num_train_epochs = epochs, 
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
        report_to = "wandb",
        logging_strategy = "steps",
        logging_steps = 25,
        save_strategy = save_strategy,
        save_steps=save_steps,
        save_only_model = True,
        run_name=run_name,
        fsdp= "full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap= 'LlamaDecoderLayer',
        fp16=True,
        weight_decay=0.01,
        dataloader_num_workers=accelerator.num_processes,
    )
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        cache_dir=cache_dir,
        trust_remote_code=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_auth_token = use_auth_token,
        model_max_length=1024,
        padding_side="right",
        cache_dir=cache_dir,
        trust_remote_code=True)


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if use_lora != 0:
        lora_config = LoraConfig(
            r=use_lora,  # LoRA rank
            lora_alpha=32,  # Scaling factor for LoRA weights
            target_modules=[
                "up_proj",
                "o_proj",
                "q_proj",
                "k_proj",
                "down_proj",
                "gate_proj",
                "v_proj"
            ],  # Target attention layers
            lora_dropout=0.1,  # LoRA dropout
            bias="none",
            task_type="CAUSAL_LM"  # Task type
        )
        model = get_peft_model(model, lora_config)

    data_module = make_supervised_data_module(output_dir, train_type, tokenizer=tokenizer)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if not save_intermediate:
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()
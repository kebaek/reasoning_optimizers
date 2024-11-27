# SAM optimizer using Trainer with FSDP 

import os
from typing import Dict
import numpy as np
import argparse
from utils import *
from contextlib import contextmanager
from collections import defaultdict
# ---- Torch ----- # 
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP 
from torch.distributed.fsdp._runtime_utils import _lazy_init
from torch.distributed.fsdp._common_utils import TrainingState
from torch.distributed.fsdp.sharded_grad_scaler import _refresh_per_optimizer_state as refresh_optim
import torch.distributed as dist
# ---- Huggingface ---- # 
import transformers
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from huggingface_params import cache_dir, use_auth_token


accelerator = Accelerator()
os.environ["WANDB_PROJECT"] = "reasoning_optimizer"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "false"  # log all model checkpoints

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

## [Speed up SAM] Ensures that the gradients are sharded for FSDP
    ## (if gradients are not synced across devices, only the parameters are sharded.)
@contextmanager
def sync(module):
    _lazy_init(module, module)
    if not module._is_root:
        raise RuntimeError(
            "`no_sync()` on inner FSDP instances is not supported. Please call `no_sync()` on root FSDP module."
        )
    module._assert_state(TrainingState.IDLE)
    old_flags = []
    for m in module.modules():
        if isinstance(m, FSDP):
            old_flags.append((m, m._sync_gradients))
            m._sync_gradients = True
    try:
        yield
    finally:
        for m, old_flag in old_flags:
            assert m._sync_gradients, (
                "`_sync_gradients` was incorrectly set to "
                "`False` while in the `sync()` context manager"
            )
            m._sync_gradients = old_flag

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
    parser.add_argument("--dont_save_intermediate", action='store_true')
    parser.add_argument("--rho", type=float, default=5e-6)
    parser.add_argument("--split_gpus", action='store_true')

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
        run_name  = f"{epochs}epochs_{train_type}_lr{learning_rate}_rho{args.rho}_bs{batch_size}_lora{use_lora}_{model_name}"
    else:
        run_name  = f"{epochs}epochs_{train_type}_lr{learning_rate}_rho{args.rho}__bs{batch_size}_{model_name}"
    
    
    batch_size = batch_size
    if args.split_gpus:
        num_devices = torch.cuda.device_count() // 2
    else:
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
    
    class CustomTrainer(Trainer):
        sam_rho = args.rho
        split_gpus = args.split_gpus
        custom_num_devices = num_devices            

        def training_step(self, model, inputs, num_items_in_batch=None):
            torch.distributed.barrier()

            ## Save Old Parameters by Cloning
            model_old_params = {}
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        if self.split_gpus:
                            model_old_params[name] = p.clone().to(f"cuda:{p.device.index + self.custom_num_devices}")
                        else:
                            model_old_params[name] = p.clone()
            
            ## Get Nabla W
            with sync(model): 
                # To keep gradients sharded. Trainer by default runs training_step
                # with no_sync context manager for all grad accumulation step before last one.
                model.train()
                model.zero_grad()

                old_optim_stats = self.accelerator.scaler._per_optimizer_states.copy()
                self.accelerator.scaler._per_optimizer_states = defaultdict(refresh_optim)

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss)

                ## Clip Grad Norm
                norm = self.accelerator.clip_grad_norm_(model.parameters(), self.sam_rho)

                # Adjust Norm of Grad
                if norm < self.sam_rho:
                    lr = self.sam_rho/norm 
                else:
                    lr = 1

                ## Perturb Weights
                with torch.no_grad():
                    # self.custom_apply(model, lambda m: self.perturb(m, lr))
                    for p in model.parameters():
                        if p.grad is None: continue 
                        if p.data.shape != p.grad.shape:
                            print('before FSDP summon', p.data.shape, p.grad.shape)
                        p.data += lr * p.grad

                self.accelerator.scaler._per_optimizer_states = old_optim_stats
                
            ## Get Training Step at Perturbed Model
            model.zero_grad()
            super().training_step(model, inputs, num_items_in_batch)

            # append current gradient to list of gradients 
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        old_params = model_old_params[name]
                        if old_params.grad is not None and p.grad is not None:
                            if self.split_gpus:
                                p.data = old_params.data.to(f"cuda:{p.device.index}")
                                p.grad += old_params.grad.to(f"cuda:{p.device.index}")
                            else:
                                p.data = old_params.data
                                p.grad += old_params.grad
                        else: 
                            if self.split_gpus:
                                p.data = old_params.data.to(f"cuda:{p.device.index}")
                            else:
                                p.data = old_params.data

            del model_old_params
            return loss.detach()

    
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
        print_trainable_parameters(model)

    data_module = make_supervised_data_module(output_dir, train_type, tokenizer=tokenizer)
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    
    if not save_intermediate:
        trainer.save_state()
        trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    train()
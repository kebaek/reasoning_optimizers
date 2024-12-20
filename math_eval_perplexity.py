from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from datasets import load_dataset
import tqdm
import torch
import argparse
from utils import *

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy
from typing import Sequence, Dict
import logging
from huggingface_params import cache_dir

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--ckpt", type=str)

args = parser.parse_args()


model_name = f"ckpts/{args.ckpt_dir}/checkpoint-{args.ckpt}"  
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

model.to("cuda")

model.eval()


def calculate_output_perplexity(input_text, output_text, model, tokenizer):
    # Combine input and output text
    full_text = input_text + output_text

    # Tokenize input and output
    inputs = tokenizer(full_text, return_tensors='pt').to('cuda')
    input_ids = inputs.input_ids

    # Get the index where the output text starts
    input_length = len(tokenizer.encode(input_text, add_special_tokens=False))
    output_length = len(tokenizer.encode(output_text, add_special_tokens=False))
    
    with torch.no_grad():
        # Get model outputs (logits)
        outputs = model(**inputs)
        logits = outputs.logits

    # Shift logits and labels to focus only on the output text portion
    shift_logits = logits[:, input_length:-1, :].contiguous()
    shift_labels = input_ids[:, input_length + 1:].contiguous()

    # Calculate the cross-entropy loss between predicted logits and actual tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Calculate perplexity
    perplexity = torch.exp(loss.mean()).item()
    num_tokens = shift_labels.ne(tokenizer.pad_token_id).sum().item()
    return perplexity, num_tokens



dataset = load_dataset("hendrycks/competition_math", cache_dir=cache_dir, trust_remote_code=True)

train_questions = np.array(dataset["train"]["problem"])
train_answers = np.array(dataset["train"]['solution'])

    
eval_questions = train_questions
eval_questions = [question + "\nAnswer:" for question in train_questions]
eval_answers = [" "+train_answers[i]+"\nDone." for i in range(len(train_answers))] 



perplexities = []
num_tokens_all = []
for i in tqdm.tqdm(range(len(eval_questions))):
    input_text = eval_questions[i]
    output_text = eval_answers[i]
    perplexity,num_tokens = calculate_output_perplexity(input_text, output_text, model, tokenizer)
    perplexities.append(perplexity)
    num_tokens_all.append(num_tokens)

np.save(model_name+"/train_perplexities.npy", perplexities)
np.save(model_name+"/train_num_tokens.npy", num_tokens_all)
from datasets import load_dataset
import numpy as np
import os
import argparse
from peft import PeftModel
from utils import *
#from huggingface_params import cache_dir, use_auth_token
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 
import torch

use_auth_token = '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'
cache_dir = '/tmp/kbaek/hf_cache'

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--eval_type", type=str, default="test")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--temp", type=float, default=0.8)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir, 
                                          cache_dir=cache_dir,
                                          use_auth_token = use_auth_token,
                                          trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir, 
                                            cache_dir=cache_dir,
                                            use_auth_token = use_auth_token,
                                            trust_remote_code=True)
special_tokens_dict = dict()
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)
if 'lora' in args.ckpt_dir:
    model = PeftModel.from_pretrained(model, args.ckpt_dir)
    
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.to("cuda")

model.eval()

if args.eval_type == "test":
    dataset = load_dataset("gsm8k", "main")

    test_questions = dataset["test"]["question"]
    test_answers = dataset["test"]['answer']
    eval_questions = test_questions
    eval_questions = [question + "\nAnswer:" for question in eval_questions]
    eval_answers = test_answers

elif args.eval_type == "train":
    dataset = load_dataset("gsm8k", "main")
    train_questions = dataset["train"]["question"]
    train_answers = dataset["train"]['answer']
    eval_questions = train_questions
    eval_questions = [question + "\nAnswer:" for question in train_questions]
    eval_answers = train_answers


def get_aug_answer(full_answer):
    idx = full_answer.rfind("The answer is")
    if idx == -1:
        return None
    else:
        answer = full_answer[idx + len("The answer is: "):]
        answer = answer.replace(":", "").replace("$", "").strip()
        if len(answer)> 0:
            if answer[-1] == ".":
                answer = answer[:-1]
            left = "\\boxed{"
            if answer[:len(left)] == left and answer[-1] == "}":
                answer = answer[len(left):-1]
        return answer.replace(",", "")

def extract_latex(text):
    start = text.find("#### ") + len("#### ")
    return text[start:].replace(",", "")

def answer_type_individial(output , answer):
    answer = extract_latex(answer)
    output_answer = extract_latex(output)
    if output_answer is not None and answer is not None:
        eqiv = is_equiv(answer, output_answer, verbose=False)

        if eqiv:
            answer_type = 0
        else:
            answer_type = 1
    else:
        answer_type = 2
    return answer_type


n = args.num_samples
temperature=args.temp
max_tokens=512
top_p=0.95
seed=args.seed
stop="\nDone."
bad_words_ids = [tokenizer(stop).input_ids]

TOTAL_EXAMPLES = len(eval_questions)
print('What is sample', n)

# if raw_output files exist, load it
if os.path.exists(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy")):
    output = np.load(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), allow_pickle=True).tolist()
else:
    output = [] 

for i in tqdm(range(len(output), TOTAL_EXAMPLES)):
    np.random.seed(seed)
    torch.manual_seed(seed)

    x = eval_questions[i] 
    tokens = tokenizer(x, return_tensors='pt').to('cuda')
    predictions = model.generate(**tokens, do_sample=True, temperature=temperature, max_new_tokens=max_tokens, top_p=top_p, bad_words_ids=bad_words_ids, num_return_sequences=n)
    samples = [tokenizer.decode(o) for o in predictions]
    output.append(samples)

    if i % 20 == 0:
        np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), np.array(output))
np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), np.array(output))

answer_types_all = []
# answers_all = []
for i in range(len(output)):
    answer_types = []
    # answers = []
    for item in output[i]:
        # answers.append(item)
        answer_type = answer_type_individial(item, eval_answers[i])
        answer_types.append(answer_type)
    answer_types_all.append(answer_types)
    # answers_all.append(answers)

answer_types_all = np.array(answer_types_all)
# answers_all = np.array(answers_all)
print((answer_types_all==0).mean(axis=-1).mean()) #ratio correct
print((answer_types_all==1).mean(axis=-1).mean()) #ratio incorrect
print((answer_types_all==2).mean(axis=-1).mean()) #ratio weird formatting

# np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answers{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answers_all)
np.save(os.path.join(args.ckpt_dir, f"{args.eval_type}_answer_types{args.num_samples}_seed{args.seed}_temp{args.temp}.npy"), answer_types_all)